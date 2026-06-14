import argparse
import json
import os
import struct
from dataclasses import dataclass
from typing import Any

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open


PRXPIXEL_SINGLE_FILE_FORMAT = "lumina_prx_pixel.transformer_single_file.v1"
PRXPIXEL_SINGLE_FILE_FORMAT_KEY = "prxpixel_format"
PRXPIXEL_SINGLE_FILE_INFO_KEY = "prxpixel_model_info"
PRXPIXEL_TRANSFORMER_CONFIG_KEY = "prxpixel_transformer_config"
PRXPIXEL_SCHEDULER_CONFIG_KEY = "prxpixel_scheduler_config"
PRXPIXEL_CONVERSION_INFO_KEY = "prxpixel_conversion_info"
FP8_SCALE_KEY_PREFIX = "__prx_internal__.fp8_scale."
FP8_STORAGE_DTYPE = torch.float8_e4m3fn
FP8_STORAGE_DTYPE_CODE = "F8_E4M3"
FP8_STORAGE_MAX = float(torch.finfo(FP8_STORAGE_DTYPE).max)

SAFE_DTYPE_NBYTES = {
    "BF16": 2,
    "BOOL": 1,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "I8": 1,
    "U8": 1,
}

FP8_ALLOWED_SUFFIXES = (
    ".attention.img_qkv_proj.weight",
    ".attention.txt_kv_proj.weight",
    ".gate_proj.weight",
    ".up_proj.weight",
    ".down_proj.weight",
)


@dataclass(frozen=True)
class ConversionPlanItem:
    shard_name: str
    tensor_name: str
    shape: tuple[int, ...]
    output_dtype_code: str
    use_fp8: bool


def _read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_model_directory(model_source: str, local_files_only: bool = False) -> str:
    if os.path.isdir(model_source):
        return model_source

    allow_patterns = [
        "model_index.json",
        "scheduler/*",
        "text_encoder/*",
        "tokenizer/*",
        "transformer/*",
        "README.md",
        "LICENSE",
        "NOTICE",
    ]
    return snapshot_download(
        repo_id=model_source,
        allow_patterns=allow_patterns,
        local_files_only=local_files_only,
    )


def make_fp8_scale_key(tensor_name: str) -> str:
    return f"{FP8_SCALE_KEY_PREFIX}{tensor_name}"


def is_fp8_scale_key(tensor_name: str) -> bool:
    return tensor_name.startswith(FP8_SCALE_KEY_PREFIX)


def is_prxpixel_single_file(path: str) -> bool:
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
        return metadata.get(PRXPIXEL_SINGLE_FILE_FORMAT_KEY) == PRXPIXEL_SINGLE_FILE_FORMAT
    except Exception:
        return False


def read_prxpixel_single_file_metadata(path: str) -> dict[str, Any]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}

    if metadata.get(PRXPIXEL_SINGLE_FILE_FORMAT_KEY) != PRXPIXEL_SINGLE_FILE_FORMAT:
        raise RuntimeError(f"{path} is not a PRX Pixel single-file transformer checkpoint.")

    def parse_json_field(key: str) -> dict[str, Any]:
        raw_value = metadata.get(key, "{}")
        if not raw_value:
            return {}
        return json.loads(raw_value)

    return {
        "format": metadata.get(PRXPIXEL_SINGLE_FILE_FORMAT_KEY),
        "model_info": parse_json_field(PRXPIXEL_SINGLE_FILE_INFO_KEY),
        "transformer_config": parse_json_field(PRXPIXEL_TRANSFORMER_CONFIG_KEY),
        "scheduler_config": parse_json_field(PRXPIXEL_SCHEDULER_CONFIG_KEY),
        "conversion_info": parse_json_field(PRXPIXEL_CONVERSION_INFO_KEY),
    }


def should_store_tensor_as_fp8(tensor_name: str, shape: tuple[int, ...]) -> bool:
    if len(shape) != 2 or not tensor_name.endswith(".weight"):
        return False
    return tensor_name.endswith(FP8_ALLOWED_SUFFIXES)


def _tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()


def _build_conversion_plan(model_dir: str) -> list[ConversionPlanItem]:
    transformer_dir = os.path.join(model_dir, "transformer")
    index_data = _read_json(os.path.join(transformer_dir, "diffusion_pytorch_model.safetensors.index.json"))
    weight_map = index_data["weight_map"]

    by_shard: dict[str, list[str]] = {}
    for tensor_name, shard_name in weight_map.items():
        by_shard.setdefault(shard_name, []).append(tensor_name)

    plan: list[ConversionPlanItem] = []
    for shard_name in sorted(by_shard):
        shard_path = os.path.join(transformer_dir, shard_name)
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for tensor_name in sorted(by_shard[shard_name]):
                shape = tuple(handle.get_slice(tensor_name).get_shape())
                source_dtype = str(handle.get_slice(tensor_name).get_dtype())
                use_fp8 = should_store_tensor_as_fp8(tensor_name, shape)
                output_dtype_code = FP8_STORAGE_DTYPE_CODE if use_fp8 else source_dtype
                if output_dtype_code not in SAFE_DTYPE_NBYTES:
                    raise RuntimeError(
                        f"Unsupported tensor dtype {output_dtype_code} for {tensor_name}. "
                        "This converter only handles standard PRXPixel safetensors dtypes."
                    )
                plan.append(
                    ConversionPlanItem(
                        shard_name=shard_name,
                        tensor_name=tensor_name,
                        shape=shape,
                        output_dtype_code=output_dtype_code,
                        use_fp8=use_fp8,
                    )
                )
    return plan


def _header_for_plan(plan: list[ConversionPlanItem], metadata: dict[str, str]) -> dict[str, Any]:
    header: dict[str, Any] = {"__metadata__": metadata}
    offset = 0

    for item in plan:
        numel = 1
        for dim in item.shape:
            numel *= int(dim)
        tensor_nbytes = numel * SAFE_DTYPE_NBYTES[item.output_dtype_code]
        header[item.tensor_name] = {
            "dtype": item.output_dtype_code,
            "shape": list(item.shape),
            "data_offsets": [offset, offset + tensor_nbytes],
        }
        offset += tensor_nbytes

        if item.use_fp8:
            scale_name = make_fp8_scale_key(item.tensor_name)
            scale_nbytes = SAFE_DTYPE_NBYTES["F32"]
            header[scale_name] = {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [offset, offset + scale_nbytes],
            }
            offset += scale_nbytes

    return header


def _build_metadata(
    model_dir: str,
    model_index: dict[str, Any],
    transformer_config: dict[str, Any],
    scheduler_config: dict[str, Any],
    fp8_tensor_count: int,
) -> dict[str, str]:
    model_info = {
        "source_model": model_dir,
        "default_sample_size": int(model_index.get("default_sample_size", 1024)),
        "noise_scale": float(model_index.get("noise_scale", 2.0)),
        "prediction_type": str(model_index.get("prediction_type", "x_prediction_flow_matching")),
        "prompt_max_tokens": int(model_index.get("prompt_max_tokens", 256)),
        "skip_text_cleaning": bool(model_index.get("skip_text_cleaning", True)),
    }
    conversion_info = {
        "recipe": "mixed_v1",
        "fp8_dtype": str(FP8_STORAGE_DTYPE),
        "fp8_tensor_count": int(fp8_tensor_count),
        "fp8_scale_key_prefix": FP8_SCALE_KEY_PREFIX,
    }
    return {
        PRXPIXEL_SINGLE_FILE_FORMAT_KEY: PRXPIXEL_SINGLE_FILE_FORMAT,
        PRXPIXEL_SINGLE_FILE_INFO_KEY: json.dumps(model_info, separators=(",", ":")),
        PRXPIXEL_TRANSFORMER_CONFIG_KEY: json.dumps(transformer_config, separators=(",", ":")),
        PRXPIXEL_SCHEDULER_CONFIG_KEY: json.dumps(scheduler_config, separators=(",", ":")),
        PRXPIXEL_CONVERSION_INFO_KEY: json.dumps(conversion_info, separators=(",", ":")),
    }


def _quantize_tensor_to_fp8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    tensor_f32 = tensor.float()
    max_value = tensor_f32.abs().amax()
    if not torch.isfinite(max_value) or float(max_value) == 0.0:
        scale = torch.ones((1,), dtype=torch.float32)
    else:
        scale = (max_value / FP8_STORAGE_MAX).reshape(1).to(torch.float32)
    quantized = (tensor_f32 / scale.item()).to(FP8_STORAGE_DTYPE)
    return quantized, scale


def convert_transformer_repo_to_fp8_mixed_file(
    model_dir: str,
    output_path: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    transformer_dir = os.path.join(model_dir, "transformer")
    if not os.path.isdir(transformer_dir):
        raise RuntimeError(f"Could not find transformer directory inside {model_dir}.")

    if os.path.exists(output_path) and not overwrite:
        raise RuntimeError(f"Output file already exists: {output_path}")

    model_index = _read_json(os.path.join(model_dir, "model_index.json"))
    transformer_config = _read_json(os.path.join(transformer_dir, "config.json"))
    scheduler_config = _read_json(os.path.join(model_dir, "scheduler", "scheduler_config.json"))
    plan = _build_conversion_plan(model_dir)
    fp8_tensor_count = sum(1 for item in plan if item.use_fp8)
    metadata = _build_metadata(model_dir, model_index, transformer_config, scheduler_config, fp8_tensor_count)
    header = _header_for_plan(plan, metadata)
    header_bytes = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temp_output_path = f"{output_path}.tmp"
    items_by_shard: dict[str, list[ConversionPlanItem]] = {}
    for item in plan:
        items_by_shard.setdefault(item.shard_name, []).append(item)

    with open(temp_output_path, "wb") as output_handle:
        output_handle.write(struct.pack("<Q", len(header_bytes)))
        output_handle.write(header_bytes)

        for shard_name in sorted(items_by_shard):
            shard_path = os.path.join(transformer_dir, shard_name)
            with safe_open(shard_path, framework="pt", device="cpu") as shard_handle:
                for item in items_by_shard[shard_name]:
                    tensor = shard_handle.get_tensor(item.tensor_name)
                    if item.use_fp8:
                        quantized, scale = _quantize_tensor_to_fp8(tensor)
                        output_handle.write(_tensor_to_bytes(quantized))
                        output_handle.write(_tensor_to_bytes(scale))
                    else:
                        output_handle.write(_tensor_to_bytes(tensor))

    os.replace(temp_output_path, output_path)
    return {
        "output_path": output_path,
        "tensor_count": len(plan),
        "fp8_tensor_count": fp8_tensor_count,
        "preserved_tensor_count": len(plan) - fp8_tensor_count,
    }


def convert_prxpixel_source_to_fp8_mixed_file(
    model_source: str,
    output_path: str,
    local_files_only: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    model_dir = resolve_model_directory(model_source, local_files_only=local_files_only)
    return convert_transformer_repo_to_fp8_mixed_file(
        model_dir=model_dir,
        output_path=output_path,
        overwrite=overwrite,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PRX Pixel transformer shards into one fp8 mixed safetensors file.")
    parser.add_argument("model_source", help="Hugging Face repo id or local diffusers model directory.")
    parser.add_argument("output_path", help="Output .safetensors file path.")
    parser.add_argument("--local-files-only", action="store_true", help="Do not hit the network for model resolution.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output file if it already exists.")
    args = parser.parse_args()

    result = convert_prxpixel_source_to_fp8_mixed_file(
        model_source=args.model_source,
        output_path=args.output_path,
        local_files_only=args.local_files_only,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
