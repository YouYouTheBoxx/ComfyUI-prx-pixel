import html
import json
import math
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn
from transformers import AutoTokenizer

from .fp8_converter import FP8_STORAGE_DTYPE, FP8_STORAGE_MAX, make_fp8_scale_key, read_prxpixel_single_file_metadata
from .hf_download import resolve_clip_model_directory, resolve_transformer_model_directory

try:
    import ftfy
except Exception:  # pragma: no cover - optional dependency
    ftfy = None

try:
    from transformers import Qwen3VLTextModel
except Exception:  # pragma: no cover - older transformers
    Qwen3VLTextModel = None

from diffusers import FlowMatchEulerDiscreteScheduler


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    max_period: int = 10000,
    flip_sin_to_cos: bool = True,
    downscale_freq_shift: float = 0.0,
    scale: float = 1.0,
) -> torch.Tensor:
    if timesteps.ndim == 0:
        timesteps = timesteps[None]
    timesteps = timesteps.float()

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        half_dim,
        dtype=torch.float32,
        device=timesteps.device,
    )
    exponent = exponent / max(half_dim - downscale_freq_shift, 1)
    emb = timesteps[:, None] * torch.exp(exponent)[None, :]
    emb = emb * scale
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))

    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    return emb


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        if self.weight is not None:
            hidden_states = hidden_states * self.weight.float()
        return hidden_states.to(input_dtype)


def basic_clean(text: str) -> str:
    text = "" if text is None else str(text)
    if ftfy is not None:
        text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def get_image_ids(batch_size: int, height: int, width: int, patch_size: int, device: torch.device) -> torch.Tensor:
    img_ids = torch.zeros(height // patch_size, width // patch_size, 2, device=device)
    img_ids[..., 0] = torch.arange(height // patch_size, device=device)[:, None]
    img_ids[..., 1] = torch.arange(width // patch_size, device=device)[None, :]
    return img_ids.reshape((height // patch_size) * (width // patch_size), 2).unsqueeze(0).repeat(batch_size, 1, 1)


def apply_rope(xq: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    freqs_cis = freqs_cis.to(device=xq.device, dtype=xq_.dtype)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq)


def patchify(img: torch.Tensor, patch_size: int) -> torch.Tensor:
    b, c, h, w = img.shape
    p = patch_size
    img = img.reshape(b, c, h // p, p, w // p, p)
    img = torch.einsum("nchpwq->nhwcpq", img)
    return img.reshape(b, -1, c * p * p)


def unpatchify(seq: torch.Tensor, patch_size: int, height: int, width: int) -> torch.Tensor:
    b, _, d = seq.shape
    p = patch_size
    c = d // (p * p)
    seq = seq.reshape(b, height // p, width // p, c, p, p)
    seq = torch.einsum("nhwcpq->nchpwq", seq)
    return seq.reshape(b, c, height, width)


class PRXEmbedND(nn.Module):
    def __init__(self, theta: int, axes_dim: list[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def rope(self, pos: torch.Tensor, dim: int) -> torch.Tensor:
        if dim % 2 != 0:
            raise ValueError(f"RoPE dimension must be even, got {dim}.")

        scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
        omega = 1.0 / (self.theta**scale)
        out = pos.unsqueeze(-1) * omega.unsqueeze(0)
        out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
        return out.reshape(*out.shape[:-1], 2, 2).float()

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        emb = torch.cat([self.rope(ids[:, :, i], self.axes_dim[i]) for i in range(ids.shape[-1])], dim=-3)
        return emb.unsqueeze(1)


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class FP8WeightLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, compute_dtype: torch.dtype, bias: bool = False):
        super().__init__()
        if bias:
            raise ValueError("FP8WeightLinear currently expects bias=False for PRXPixel fp8 layers.")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.compute_dtype = compute_dtype
        self.register_buffer("weight", torch.zeros((out_features, in_features), dtype=FP8_STORAGE_DTYPE))
        self.register_buffer("weight_scale", torch.ones((1,), dtype=torch.float32))
        self.register_buffer("bias", None)
        self.is_fp8_weight = True

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"weight_dtype={self.weight.dtype}, compute_dtype={self.compute_dtype}"
        )

    def _quantize_input(self, x_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = x_2d.float().abs().amax().clamp(min=1e-12) / FP8_STORAGE_MAX
        x_fp8 = (x_2d.float() / scale).to(FP8_STORAGE_DTYPE)
        return x_fp8.contiguous(), scale.to(dtype=torch.float32)

    def _dequantized_weight(self, dtype: torch.dtype) -> torch.Tensor:
        return (self.weight.float() * self.weight_scale.reshape(())).to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16, torch.float32) else self.compute_dtype

        if x.device.type == "cuda" and hasattr(torch, "_scaled_mm"):
            x_2d = x.reshape(-1, x.shape[-1]).contiguous()
            x_fp8, x_scale = self._quantize_input(x_2d)
            weight_t = self.weight.t()
            out_2d = torch._scaled_mm(
                x_fp8,
                weight_t,
                x_scale,
                self.weight_scale.reshape(()),
                out_dtype=output_dtype,
            )
            return out_2d.reshape(*x.shape[:-1], self.out_features)

        return F.linear(x.to(dtype=output_dtype), self._dequantized_weight(output_dtype), None)


class ResolutionEmbedder(nn.Module):
    def __init__(self, hidden_dim: int, max_period: int = 10000):
        super().__init__()
        self.max_period = max_period
        self.mlp = MLPEmbedder(256, hidden_dim)

    def forward(self, height: int, width: int, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        h = torch.full((batch_size,), float(height), device=device, dtype=torch.float32)
        w = torch.full((batch_size,), float(width), device=device, dtype=torch.float32)
        res = torch.cat(
            [
                get_timestep_embedding(
                    h,
                    128,
                    max_period=self.max_period,
                    flip_sin_to_cos=True,
                    downscale_freq_shift=0.0,
                    scale=1.0,
                ),
                get_timestep_embedding(
                    w,
                    128,
                    max_period=self.max_period,
                    flip_sin_to_cos=True,
                    downscale_freq_shift=0.0,
                    scale=1.0,
                ),
            ],
            dim=-1,
        )
        return self.mlp(res.to(dtype))


class Modulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Linear(dim, 6 * dim, bias=True)
        nn.init.constant_(self.lin.weight, 0)
        nn.init.constant_(self.lin.bias, 0)

    def forward(
        self, vec: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        out = self.lin(F.silu(vec))[:, None, :].chunk(6, dim=-1)
        return tuple(out[:3]), tuple(out[3:])


class PRXAttention(nn.Module):
    def __init__(self, query_dim: int, heads: int, dim_head: int, eps: float = 1e-6):
        super().__init__()
        self.heads = heads
        self.head_dim = dim_head
        self.inner_dim = dim_head * heads

        self.img_qkv_proj = nn.Linear(query_dim, query_dim * 3, bias=False)
        self.norm_q = RMSNorm(self.head_dim, eps=eps, elementwise_affine=True)
        self.norm_k = RMSNorm(self.head_dim, eps=eps, elementwise_affine=True)
        self.txt_kv_proj = nn.Linear(query_dim, query_dim * 2, bias=False)
        self.norm_added_k = RMSNorm(self.head_dim, eps=eps, elementwise_affine=True)
        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, query_dim, bias=False), nn.Dropout(0.0)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        image_rotary_emb: torch.Tensor,
    ) -> torch.Tensor:
        img_qkv = self.img_qkv_proj(hidden_states)
        bsz, img_len, _ = img_qkv.shape
        img_qkv = img_qkv.reshape(bsz, img_len, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        img_q, img_k, img_v = img_qkv[0], img_qkv[1], img_qkv[2]

        img_q = self.norm_q(img_q)
        img_k = self.norm_k(img_k)

        txt_kv = self.txt_kv_proj(encoder_hidden_states)
        _, txt_len, _ = txt_kv.shape
        txt_kv = txt_kv.reshape(bsz, txt_len, 2, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        txt_k, txt_v = txt_kv[0], txt_kv[1]
        txt_k = self.norm_added_k(txt_k)

        img_q = apply_rope(img_q, image_rotary_emb)
        img_k = apply_rope(img_k, image_rotary_emb)

        key = torch.cat([txt_k, img_k], dim=2)
        value = torch.cat([txt_v, img_v], dim=2)

        attn_mask_tensor = None
        if attention_mask is not None:
            txt_mask = attention_mask.to(device=hidden_states.device, dtype=torch.bool)
            img_mask = torch.ones((bsz, img_len), device=hidden_states.device, dtype=torch.bool)
            joint = torch.cat([txt_mask, img_mask], dim=-1)
            attn_mask_tensor = joint[:, None, None, :].expand(-1, self.heads, img_len, -1)

        attn_output = F.scaled_dot_product_attention(
            img_q,
            key,
            value,
            attn_mask=attn_mask_tensor,
            dropout_p=0.0,
            is_causal=False,
        )
        attn_output = attn_output.transpose(1, 2).reshape(bsz, img_len, self.inner_dim)
        attn_output = self.to_out[0](attn_output)
        return self.to_out[1](attn_output)


class PRXBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.head_dim = hidden_size // num_heads
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)

        self.img_pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attention = PRXAttention(hidden_size, num_heads, self.head_dim)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.gate_proj = nn.Linear(hidden_size, self.mlp_hidden_dim, bias=False)
        self.up_proj = nn.Linear(hidden_size, self.mlp_hidden_dim, bias=False)
        self.down_proj = nn.Linear(self.mlp_hidden_dim, hidden_size, bias=False)
        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        mod_attn, mod_mlp = self.modulation(temb)
        attn_shift, attn_scale, attn_gate = mod_attn
        mlp_shift, mlp_scale, mlp_gate = mod_mlp

        hidden_states_mod = (1 + attn_scale) * self.img_pre_norm(hidden_states) + attn_shift
        attn_out = self.attention(
            hidden_states=hidden_states_mod,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )
        hidden_states = hidden_states + attn_gate * attn_out

        mlp_in = (1 + mlp_scale) * self.post_attention_layernorm(hidden_states) + mlp_shift
        mlp_out = self.down_proj(self.mlp_act(self.gate_proj(mlp_in)) * self.up_proj(mlp_in))
        return hidden_states + mlp_gate * mlp_out


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, output_dim: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, output_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(temb).chunk(2, dim=1)
        hidden_states = (1 + scale[:, None, :]) * self.norm_final(hidden_states) + shift[:, None, :]
        return self.linear(hidden_states)


class PRXPixelTransformer2DModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        patch_size: int,
        context_in_dim: int,
        hidden_size: int,
        mlp_ratio: float,
        num_heads: int,
        depth: int,
        axes_dim: list[int],
        theta: int,
        time_factor: float,
        time_max_period: int,
        bottleneck_size: int,
        resolution_embeds: bool,
    ):
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}.")

        self.in_channels = int(in_channels)
        self.patch_size = int(patch_size)
        self.context_in_dim = int(context_in_dim)
        self.hidden_size = int(hidden_size)
        self.mlp_ratio = float(mlp_ratio)
        self.num_heads = int(num_heads)
        self.depth = int(depth)
        self.axes_dim = list(axes_dim)
        self.theta = int(theta)
        self.time_factor = float(time_factor)
        self.time_max_period = int(time_max_period)
        self.bottleneck_size = int(bottleneck_size)
        self.resolution_embeds = bool(resolution_embeds)

        patch_dim = self.in_channels * self.patch_size * self.patch_size
        pe_dim = self.hidden_size // self.num_heads
        if sum(self.axes_dim) != pe_dim:
            raise ValueError(f"axes_dim sum must equal {pe_dim}, got {self.axes_dim}.")

        self.pe_embedder = PRXEmbedND(theta=self.theta, axes_dim=self.axes_dim)
        self.img_in = nn.Sequential(
            nn.Linear(patch_dim, self.bottleneck_size, bias=True),
            nn.Linear(self.bottleneck_size, self.hidden_size, bias=True),
        )
        self.time_in = MLPEmbedder(256, self.hidden_size)
        self.txt_in = nn.Linear(self.context_in_dim, self.hidden_size, bias=True)
        self.resolution_embedder = (
            ResolutionEmbedder(self.hidden_size, max_period=self.time_max_period) if self.resolution_embeds else None
        )
        self.blocks = nn.ModuleList(
            [PRXBlock(self.hidden_size, self.num_heads, self.mlp_ratio) for _ in range(self.depth)]
        )
        self.final_layer = FinalLayer(self.hidden_size, patch_dim)

    def _compute_timestep_embedding(self, timestep: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        embedding = get_timestep_embedding(
            timesteps=timestep,
            embedding_dim=256,
            max_period=self.time_max_period,
            flip_sin_to_cos=True,
            downscale_freq_shift=0.0,
            scale=self.time_factor,
        ).to(dtype)
        return self.time_in(embedding)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        txt = self.txt_in(encoder_hidden_states)
        bsz, _, height, width = hidden_states.shape
        img = patchify(hidden_states, self.patch_size)
        img = self.img_in(img)

        img_ids = get_image_ids(bsz, height, width, self.patch_size, hidden_states.device)
        rotary = self.pe_embedder(img_ids)

        vec = self._compute_timestep_embedding(timestep, img.dtype)
        if self.resolution_embedder is not None:
            vec = vec + self.resolution_embedder(height, width, bsz, hidden_states.device, img.dtype)

        for block in self.blocks:
            img = block(
                hidden_states=img,
                encoder_hidden_states=txt,
                temb=vec,
                image_rotary_emb=rotary,
                attention_mask=attention_mask,
            )

        img = self.final_layer(img, vec)
        return unpatchify(img, self.patch_size, height, width)


def _read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_model_directory(model_source: str, local_files_only: bool) -> str:
    return resolve_transformer_model_directory(model_source=model_source, local_files_only=local_files_only)


def _model_named_tensors(module: nn.Module) -> dict[str, torch.Tensor]:
    tensors = {}
    tensors.update(dict(module.named_parameters()))
    tensors.update(dict(module.named_buffers()))
    return tensors


def _set_submodule(root: nn.Module, module_name: str, new_module: nn.Module) -> None:
    parent_name, child_name = module_name.rsplit(".", 1)
    parent_module = root.get_submodule(parent_name)
    setattr(parent_module, child_name, new_module)


def _enable_fp8_single_file_layers(
    transformer: PRXPixelTransformer2DModel,
    checkpoint_path: str,
    compute_dtype: torch.dtype,
) -> set[str]:
    with safe_open(checkpoint_path, framework="pt", device="cpu") as handle:
        checkpoint_keys = set(handle.keys())

    fp8_weight_names = {
        key[len("__prx_internal__.fp8_scale.") :]
        for key in checkpoint_keys
        if key.startswith("__prx_internal__.fp8_scale.")
    }

    replaced_weight_names: set[str] = set()
    for weight_name in sorted(fp8_weight_names):
        if not weight_name.endswith(".weight"):
            continue
        module_name = weight_name[: -len(".weight")]
        original_module = transformer.get_submodule(module_name)
        if not isinstance(original_module, nn.Linear):
            raise RuntimeError(f"Expected {module_name} to be nn.Linear, got {type(original_module).__name__}.")
        replacement = FP8WeightLinear(
            in_features=original_module.in_features,
            out_features=original_module.out_features,
            compute_dtype=compute_dtype,
            bias=original_module.bias is not None,
        )
        replacement.to(device=original_module.weight.device)
        _set_submodule(transformer, module_name, replacement)
        replaced_weight_names.add(weight_name)

    return replaced_weight_names


def load_sharded_safetensors_into_module(module: nn.Module, base_dir: str, index_filename: str) -> None:
    index_path = os.path.join(base_dir, index_filename)
    index_data = _read_json(index_path)
    weight_map = index_data["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for tensor_name, shard_name in weight_map.items():
        by_shard.setdefault(shard_name, []).append(tensor_name)

    module_tensors = _model_named_tensors(module)
    missing = set(module_tensors.keys())

    with torch.no_grad():
        for shard_name, tensor_names in by_shard.items():
            shard_path = os.path.join(base_dir, shard_name)
            with safe_open(shard_path, framework="pt", device="cpu") as shard:
                for tensor_name in tensor_names:
                    if tensor_name not in module_tensors:
                        continue
                    target = module_tensors[tensor_name]
                    tensor = shard.get_tensor(tensor_name)
                    if list(target.shape) != list(tensor.shape):
                        raise RuntimeError(
                            f"Shape mismatch for {tensor_name}: expected {list(target.shape)}, got {list(tensor.shape)}."
                        )
                    target.copy_(tensor.to(device=target.device, dtype=target.dtype))
                    missing.discard(tensor_name)

    if missing:
        sample = ", ".join(sorted(list(missing))[:10])
        raise RuntimeError(f"Missing {len(missing)} tensors while loading transformer: {sample}")


def load_single_file_safetensors_into_module(module: nn.Module, checkpoint_path: str) -> None:
    module_tensors = _model_named_tensors(module)
    direct_tensor_names = {name for name in module_tensors if not name.endswith(".weight_scale")}
    scale_tensor_names = {
        make_fp8_scale_key(name[: -len(".weight_scale")] + ".weight"): name
        for name in module_tensors
        if name.endswith(".weight_scale")
    }
    missing = set(direct_tensor_names)

    with safe_open(checkpoint_path, framework="pt", device="cpu") as handle:
        key_set = set(handle.keys())
        with torch.no_grad():
            for tensor_name in sorted(direct_tensor_names):
                if tensor_name not in key_set:
                    continue

                target = module_tensors[tensor_name]
                tensor = handle.get_tensor(tensor_name)
                scale_key = make_fp8_scale_key(tensor_name)
                if scale_key in key_set and target.dtype != FP8_STORAGE_DTYPE:
                    scale = handle.get_tensor(scale_key).reshape(()).float()
                    tensor = tensor.float() * scale

                if list(target.shape) != list(tensor.shape):
                    raise RuntimeError(
                        f"Shape mismatch for {tensor_name}: expected {list(target.shape)}, got {list(tensor.shape)}."
                    )
                target.copy_(tensor.to(device=target.device, dtype=target.dtype))
                missing.discard(tensor_name)

            for scale_key, target_name in scale_tensor_names.items():
                if scale_key not in key_set:
                    continue
                target = module_tensors[target_name]
                scale_tensor = handle.get_tensor(scale_key)
                if list(target.shape) != list(scale_tensor.shape):
                    raise RuntimeError(
                        f"Shape mismatch for {target_name}: expected {list(target.shape)}, got {list(scale_tensor.shape)}."
                    )
                target.copy_(scale_tensor.to(device=target.device, dtype=target.dtype))

    if missing:
        sample = ", ".join(sorted(list(missing))[:10])
        raise RuntimeError(f"Missing {len(missing)} tensors while loading single-file transformer: {sample}")


@dataclass
class LoadedPRXPixelModel:
    source_id: str
    transformer: PRXPixelTransformer2DModel
    scheduler: FlowMatchEulerDiscreteScheduler
    default_sample_size: int
    noise_scale: float
    prediction_type: str
    dtype: torch.dtype
    storage_dtype: torch.dtype | None = None
    fp8_layer_count: int = 0
    preferred_device: torch.device | None = None
    cache_key: Any = None


@dataclass
class LoadedPRXPixelClip:
    source_id: str
    tokenizer: Any
    text_encoder: Any
    prompt_max_tokens: int
    skip_text_cleaning: bool
    dtype: torch.dtype
    preferred_device: torch.device | None = None
    cache_key: Any = None


def _build_transformer_from_config(
    transformer_config: dict[str, Any],
    dtype: torch.dtype,
    device: torch.device | None = None,
) -> PRXPixelTransformer2DModel:
    transformer = PRXPixelTransformer2DModel(
        in_channels=transformer_config["in_channels"],
        patch_size=transformer_config["patch_size"],
        context_in_dim=transformer_config["context_in_dim"],
        hidden_size=transformer_config["hidden_size"],
        mlp_ratio=transformer_config["mlp_ratio"],
        num_heads=transformer_config["num_heads"],
        depth=transformer_config["depth"],
        axes_dim=transformer_config["axes_dim"],
        theta=transformer_config["theta"],
        time_factor=transformer_config["time_factor"],
        time_max_period=transformer_config["time_max_period"],
        bottleneck_size=transformer_config["bottleneck_size"],
        resolution_embeds=transformer_config.get("resolution_embeds", True),
    )
    if device is None:
        return transformer.to(dtype=dtype)
    return transformer.to(device=device, dtype=dtype)


def load_prxpixel_model_from_repo(
    model_dir: str,
    dtype: torch.dtype,
    device: torch.device | None = None,
) -> LoadedPRXPixelModel:
    model_index = _read_json(os.path.join(model_dir, "model_index.json"))
    transformer_config = _read_json(os.path.join(model_dir, "transformer", "config.json"))
    scheduler_config = _read_json(os.path.join(model_dir, "scheduler", "scheduler_config.json"))

    transformer = _build_transformer_from_config(transformer_config, dtype=dtype, device=device)
    load_sharded_safetensors_into_module(
        transformer,
        os.path.join(model_dir, "transformer"),
        "diffusion_pytorch_model.safetensors.index.json",
    )
    transformer.eval()

    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=scheduler_config["num_train_timesteps"],
        shift=scheduler_config["shift"],
    )

    return LoadedPRXPixelModel(
        source_id=model_dir,
        transformer=transformer,
        scheduler=scheduler,
        default_sample_size=int(model_index.get("default_sample_size", 1024)),
        noise_scale=float(model_index.get("noise_scale", 2.0)),
        prediction_type=str(model_index.get("prediction_type", "x_prediction_flow_matching")),
        dtype=dtype,
        storage_dtype=dtype,
    )


def load_prxpixel_model_from_single_file(
    checkpoint_path: str,
    dtype: torch.dtype,
    device: torch.device | None = None,
) -> LoadedPRXPixelModel:
    metadata = read_prxpixel_single_file_metadata(checkpoint_path)
    model_info = metadata["model_info"]
    transformer_config = metadata["transformer_config"]
    scheduler_config = metadata["scheduler_config"]

    transformer = _build_transformer_from_config(transformer_config, dtype=dtype, device=device)
    fp8_weight_names = _enable_fp8_single_file_layers(transformer, checkpoint_path=checkpoint_path, compute_dtype=dtype)
    load_single_file_safetensors_into_module(transformer, checkpoint_path=checkpoint_path)
    transformer.eval()

    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=scheduler_config["num_train_timesteps"],
        shift=scheduler_config["shift"],
    )

    return LoadedPRXPixelModel(
        source_id=checkpoint_path,
        transformer=transformer,
        scheduler=scheduler,
        default_sample_size=int(model_info.get("default_sample_size", 1024)),
        noise_scale=float(model_info.get("noise_scale", 2.0)),
        prediction_type=str(model_info.get("prediction_type", "x_prediction_flow_matching")),
        dtype=dtype,
        storage_dtype=FP8_STORAGE_DTYPE if fp8_weight_names else dtype,
        fp8_layer_count=len(fp8_weight_names),
    )


def load_prxpixel_clip_from_repo(model_dir: str, dtype: torch.dtype) -> LoadedPRXPixelClip:
    if Qwen3VLTextModel is None:
        raise RuntimeError("This node needs transformers>=4.57 because Qwen3VLTextModel is required.")

    model_index = _read_json(os.path.join(model_dir, "model_index.json"))
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"), use_fast=True)
    text_encoder = Qwen3VLTextModel.from_pretrained(
        os.path.join(model_dir, "text_encoder"),
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    text_encoder.eval()

    return LoadedPRXPixelClip(
        source_id=model_dir,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        prompt_max_tokens=int(model_index.get("prompt_max_tokens", 256)),
        skip_text_cleaning=bool(model_index.get("skip_text_cleaning", True)),
        dtype=dtype,
    )


class PRXPixelInference:
    def __init__(self, model: LoadedPRXPixelModel, clip: LoadedPRXPixelClip):
        self.model = model
        self.clip = clip

    def encode_prompts(
        self,
        prompt: str | list[str],
        negative_prompt: str,
        do_classifier_free_guidance: bool,
        num_images_per_prompt: int,
        device: torch.device,
        text_encoder_device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if isinstance(prompt, str):
            prompt_list = [prompt]
        else:
            prompt_list = list(prompt)

        tokenizer = self.clip.tokenizer
        max_length = int(self.clip.prompt_max_tokens or 256)
        if self.clip.skip_text_cleaning:
            prompt_list = [basic_clean(text) for text in prompt_list]
        else:
            prompt_list = [str(text).strip() for text in prompt_list]

        prompt_tokens = tokenizer(
            prompt_list,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )
        prompt_tokens = {key: value.to(text_encoder_device) for key, value in prompt_tokens.items()}

        with torch.inference_mode():
            prompt_outputs = self.clip.text_encoder(
                input_ids=prompt_tokens["input_ids"],
                attention_mask=prompt_tokens["attention_mask"],
                output_hidden_states=True,
            )
        prompt_embeds = prompt_outputs["last_hidden_state"].to(device=device, dtype=self.model.dtype)
        prompt_attention_mask = prompt_tokens["attention_mask"].to(device=device, dtype=torch.bool)

        neg_embeds = None
        neg_mask = None
        if do_classifier_free_guidance:
            if self.clip.skip_text_cleaning:
                negative_prompt_value = basic_clean(negative_prompt)
            else:
                negative_prompt_value = str(negative_prompt).strip()
            negative_prompt_list = [negative_prompt_value] * len(prompt_list)
            neg_tokens = tokenizer(
                negative_prompt_list,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_attention_mask=True,
                return_tensors="pt",
            )
            neg_tokens = {key: value.to(text_encoder_device) for key, value in neg_tokens.items()}
            with torch.inference_mode():
                neg_outputs = self.clip.text_encoder(
                    input_ids=neg_tokens["input_ids"],
                    attention_mask=neg_tokens["attention_mask"],
                    output_hidden_states=True,
                )
            neg_embeds = neg_outputs["last_hidden_state"].to(device=device, dtype=self.model.dtype)
            neg_mask = neg_tokens["attention_mask"].to(device=device, dtype=torch.bool)

        if num_images_per_prompt > 1:
            prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            prompt_attention_mask = prompt_attention_mask.repeat_interleave(num_images_per_prompt, dim=0)

            if neg_embeds is not None and neg_mask is not None:
                neg_embeds = neg_embeds.repeat_interleave(num_images_per_prompt, dim=0)
                neg_mask = neg_mask.repeat_interleave(num_images_per_prompt, dim=0)

        return prompt_embeds, prompt_attention_mask, neg_embeds, neg_mask

    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        height: int,
        width: int,
        num_inference_steps: int,
        guidance_scale: float,
        generator: torch.Generator,
        device: torch.device,
        text_encoder_device: torch.device,
        after_prompt_encoding_callback=None,
        progress_callback=None,
    ) -> torch.Tensor:
        if guidance_scale < 1.0:
            raise RuntimeError("guidance_scale must be >= 1.0 for PRXPixel.")
        if height % self.model.transformer.patch_size != 0 or width % self.model.transformer.patch_size != 0:
            raise RuntimeError(
                f"height and width must be divisible by patch_size={self.model.transformer.patch_size}."
            )

        do_classifier_free_guidance = guidance_scale > 1.0

        prompt_embeds, prompt_mask, neg_embeds, neg_mask = self.encode_prompts(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            num_images_per_prompt=1,
            device=device,
            text_encoder_device=text_encoder_device,
        )

        if after_prompt_encoding_callback is not None:
            after_prompt_encoding_callback()

        batch_size = prompt_embeds.shape[0]
        latents = torch.randn(
            (batch_size, self.model.transformer.in_channels, height, width),
            generator=generator,
            device=device,
            dtype=self.model.dtype,
        ) * float(self.model.noise_scale)

        self.model.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.model.scheduler.timesteps
        if do_classifier_free_guidance:
            ca_embed = torch.cat([neg_embeds, prompt_embeds], dim=0)
            ca_mask = torch.cat([neg_mask, prompt_mask], dim=0)
        else:
            ca_embed = prompt_embeds
            ca_mask = prompt_mask

        for step_index, t in enumerate(timesteps):
            if do_classifier_free_guidance:
                latents_in = torch.cat([latents, latents], dim=0)
                t_cont = (t.float() / self.model.scheduler.config.num_train_timesteps).view(1).repeat(2).to(device)
            else:
                latents_in = latents
                t_cont = (t.float() / self.model.scheduler.config.num_train_timesteps).view(1).to(device)

            with torch.inference_mode():
                noise_pred = self.model.transformer(
                    hidden_states=latents_in,
                    timestep=t_cont,
                    encoder_hidden_states=ca_embed,
                    attention_mask=ca_mask,
                )

            if do_classifier_free_guidance:
                noise_uncond, noise_text = noise_pred.chunk(2, dim=0)
                noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

            if self.model.prediction_type == "x_prediction_flow_matching":
                t_x = torch.clamp(t.float() / self.model.scheduler.config.num_train_timesteps, min=0.05)
                noise_pred = (latents - noise_pred) / t_x

            latents = self.model.scheduler.step(noise_pred, t, latents, generator=generator).prev_sample

            if progress_callback is not None:
                progress_callback(step_index + 1, len(timesteps))

        image = ((latents.float() / 2.0) + 0.5).clamp(0.0, 1.0)
        return image.permute(0, 2, 3, 1).contiguous().cpu()
