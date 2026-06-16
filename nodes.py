import gc
import os
import threading

import comfy.model_management as mm
import comfy.model_patcher
import comfy.utils
import folder_paths
import torch
from torch import nn

from .fp8_converter import convert_prxpixel_source_to_fp8_mixed_file, is_prxpixel_single_file
from .hf_download import resolve_clip_model_directory
from .prxpixel_runtime import (
    PRXPixelInference,
    load_prxpixel_clip_from_repo,
    load_prxpixel_model_from_repo,
    load_prxpixel_model_from_single_file,
    resolve_model_directory,
)


PRX_PIXEL_MODEL_TYPE = "PRX_PIXEL_MODEL"
PRX_PIXEL_CLIP_TYPE = "PRX_PIXEL_CLIP"
LOCAL_MODEL_FOLDER_KEY = "prx_pixel_transformers"
MODEL_CACHE = {}
CLIP_CACHE = {}
CACHE_LOCK = threading.Lock()


def _register_combined_folder_key(key: str, targets: list[str], extensions: set[str]) -> None:
    existing_paths, _existing_extensions = folder_paths.folder_names_and_paths.get(key, ([], set()))
    combined_paths = []

    if isinstance(existing_paths, (list, tuple, set)):
        for path in existing_paths:
            if path not in combined_paths:
                combined_paths.append(path)

    for target in targets:
        target_paths, _ = folder_paths.folder_names_and_paths.get(target, ([], set()))
        if isinstance(target_paths, (list, tuple, set)):
            for path in target_paths:
                if path not in combined_paths:
                    combined_paths.append(path)

    folder_paths.folder_names_and_paths[key] = (combined_paths, set(extensions))


_register_combined_folder_key(LOCAL_MODEL_FOLDER_KEY, ["diffusion_models", "unet"], {".safetensors"})


def _resolve_device(device_name: str) -> torch.device:
    name = (device_name or "auto").strip().lower()
    if name in {"auto", "cuda", "gpu"}:
        if torch.cuda.is_available():
            try:
                return mm.get_torch_device()
            except Exception:
                return torch.device("cuda")
        return torch.device("cpu")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device(name)


def _resolve_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    name = (dtype_name or "auto").strip().lower()
    if name == "auto":
        if device.type == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    if name == "bfloat16":
        return torch.float32 if device.type == "cpu" else torch.bfloat16
    if name == "float16":
        return torch.float32 if device.type == "cpu" else torch.float16
    return torch.float32


def _is_torch_oom_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda out of memory" in message


def _clear_memory_after_load_failure() -> None:
    gc.collect()
    try:
        mm.soft_empty_cache()
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _normalize_cache_source(source: str) -> str:
    source = (source or "").strip()
    if not source:
        return source
    if os.path.exists(source):
        return os.path.abspath(source)
    return source


def _model_cache_key(
    loader_kind: str,
    source: str,
    resolved_dtype: torch.dtype,
    device_name: str,
    local_files_only: bool | None,
) -> tuple[str, str, str, str, bool | None]:
    return (loader_kind, _normalize_cache_source(source), str(resolved_dtype), device_name.strip().lower(), local_files_only)


def _clip_cache_key(
    source: str,
    resolved_dtype: torch.dtype,
    device_name: str,
    local_files_only: bool,
) -> tuple[str, str, str, bool]:
    return (_normalize_cache_source(source), str(resolved_dtype), device_name.strip().lower(), bool(local_files_only))


class _ComfyManagedModule(nn.Module):
    def __init__(self, wrapped_module: nn.Module, initial_device: torch.device):
        super().__init__()
        self.wrapped_module = wrapped_module
        self.device = initial_device

    def forward(self, *args, **kwargs):
        return self.wrapped_module(*args, **kwargs)

    def to(self, *args, **kwargs):
        self.wrapped_module.to(*args, **kwargs)
        return self

    def eval(self):
        self.wrapped_module.eval()
        return self

    def train(self, mode: bool = True):
        self.wrapped_module.train(mode)
        return self

    def __getattr__(self, name):
        try:
            return nn.Module.__getattr__(self, name)
        except AttributeError:
            return getattr(self.wrapped_module, name)


def _module_size(module) -> int:
    try:
        return int(mm.module_size(module))
    except Exception:
        total = 0
        for tensor in list(module.parameters()) + list(module.buffers()):
            total += tensor.nelement() * tensor.element_size()
        return int(total)


def _offload_device_for(load_device: torch.device, is_text_encoder: bool) -> torch.device:
    if load_device.type == "cpu":
        return torch.device("cpu")
    try:
        return mm.text_encoder_offload_device() if is_text_encoder else mm.unet_offload_device()
    except Exception:
        return torch.device("cpu")


def _ensure_comfy_patcher(container, attr_name: str, module, load_device: torch.device, is_text_encoder: bool):
    offload_device = _offload_device_for(load_device, is_text_encoder=is_text_encoder)
    patcher = getattr(container, attr_name, None)
    if patcher is None:
        managed_attr_name = f"{attr_name}_module"
        managed_module = _ComfyManagedModule(module, offload_device)
        try:
            mm.archive_model_dtypes(managed_module)
        except Exception:
            pass
        patcher = comfy.model_patcher.ModelPatcher(
            managed_module,
            load_device=load_device,
            offload_device=offload_device,
            size=_module_size(managed_module),
        )
        setattr(container, managed_attr_name, managed_module)
        setattr(container, attr_name, patcher)
    else:
        patcher.load_device = load_device
        patcher.offload_device = offload_device
        managed_module = getattr(container, f"{attr_name}_module", None)
        if managed_module is not None:
            managed_module.device = offload_device
    return patcher


def _ensure_model_patcher(loaded_model, load_device: torch.device):
    return _ensure_comfy_patcher(
        loaded_model,
        "transformer_patcher",
        loaded_model.transformer,
        load_device=load_device,
        is_text_encoder=False,
    )


def _ensure_clip_patcher(loaded_clip, load_device: torch.device):
    return _ensure_comfy_patcher(
        loaded_clip,
        "text_encoder_patcher",
        loaded_clip.text_encoder,
        load_device=load_device,
        is_text_encoder=True,
    )


def _free_loaded_model(loaded_model) -> None:
    patcher = getattr(loaded_model, "transformer_patcher", None)
    if patcher is not None:
        try:
            mm.unload_model_and_clones(patcher, unload_additional_models=False, all_devices=True)
        except Exception:
            pass
    try:
        loaded_model.transformer.to("cpu")
    except Exception:
        pass
    gc.collect()
    try:
        mm.soft_empty_cache()
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _free_loaded_clip(loaded_clip) -> None:
    patcher = getattr(loaded_clip, "text_encoder_patcher", None)
    if patcher is not None:
        try:
            mm.unload_model_and_clones(patcher, unload_additional_models=False, all_devices=True)
        except Exception:
            pass
    try:
        loaded_clip.text_encoder.to("cpu")
    except Exception:
        pass
    gc.collect()
    try:
        mm.soft_empty_cache()
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _get_or_load_repo_model(
    model_source: str,
    dtype_name: str,
    device_name: str,
    local_files_only: bool,
    force_reload: bool,
):
    model_device = _resolve_device(device_name)
    resolved_dtype = _resolve_dtype(dtype_name, model_device)
    key = _model_cache_key("repo", model_source, resolved_dtype, device_name, local_files_only)

    with CACHE_LOCK:
        if force_reload and key in MODEL_CACHE:
            _free_loaded_model(MODEL_CACHE.pop(key))
        cached = MODEL_CACHE.get(key)
        if cached is not None:
            cached.preferred_device = model_device
            cached.cache_key = key
            return cached

    model_dir = resolve_model_directory(model_source, local_files_only=local_files_only)
    storage_device = model_device
    fallback_device = _offload_device_for(model_device, is_text_encoder=False)

    try:
        loaded_model = load_prxpixel_model_from_repo(
            model_dir=model_dir,
            dtype=resolved_dtype,
            device=storage_device,
        )
    except RuntimeError as exc:
        if model_device.type == "cpu" or fallback_device == storage_device or not _is_torch_oom_error(exc):
            raise
        _clear_memory_after_load_failure()
        loaded_model = load_prxpixel_model_from_repo(
            model_dir=model_dir,
            dtype=resolved_dtype,
            device=fallback_device,
        )

    loaded_model.preferred_device = model_device
    loaded_model.cache_key = key

    with CACHE_LOCK:
        MODEL_CACHE[key] = loaded_model
    return loaded_model


def _get_or_load_single_file_model(
    model_path: str,
    dtype_name: str,
    device_name: str,
    force_reload: bool,
):
    model_device = _resolve_device(device_name)
    resolved_dtype = _resolve_dtype(dtype_name, model_device)
    key = _model_cache_key("single_file", model_path, resolved_dtype, device_name, None)

    with CACHE_LOCK:
        if force_reload and key in MODEL_CACHE:
            _free_loaded_model(MODEL_CACHE.pop(key))
        cached = MODEL_CACHE.get(key)
        if cached is not None:
            cached.preferred_device = model_device
            cached.cache_key = key
            return cached

    storage_device = model_device
    fallback_device = _offload_device_for(model_device, is_text_encoder=False)

    try:
        loaded_model = load_prxpixel_model_from_single_file(
            checkpoint_path=model_path,
            dtype=resolved_dtype,
            device=storage_device,
        )
    except RuntimeError as exc:
        if model_device.type == "cpu" or fallback_device == storage_device or not _is_torch_oom_error(exc):
            raise
        _clear_memory_after_load_failure()
        loaded_model = load_prxpixel_model_from_single_file(
            checkpoint_path=model_path,
            dtype=resolved_dtype,
            device=fallback_device,
        )

    loaded_model.preferred_device = model_device
    loaded_model.cache_key = key

    with CACHE_LOCK:
        MODEL_CACHE[key] = loaded_model
    return loaded_model


def _get_or_load_clip(
    model_source: str,
    dtype_name: str,
    device_name: str,
    local_files_only: bool,
    force_reload: bool,
):
    clip_device = _resolve_device(device_name)
    resolved_dtype = _resolve_dtype(dtype_name, clip_device)
    key = _clip_cache_key(model_source, resolved_dtype, device_name, local_files_only)

    with CACHE_LOCK:
        if force_reload and key in CLIP_CACHE:
            _free_loaded_clip(CLIP_CACHE.pop(key))
        cached = CLIP_CACHE.get(key)
        if cached is not None:
            cached.preferred_device = clip_device
            cached.cache_key = key
            return cached

    model_dir = resolve_clip_model_directory(model_source, local_files_only=local_files_only)
    loaded_clip = load_prxpixel_clip_from_repo(model_dir=model_dir, dtype=resolved_dtype)
    loaded_clip.preferred_device = clip_device
    loaded_clip.cache_key = key

    with CACHE_LOCK:
        CLIP_CACHE[key] = loaded_clip
    return loaded_clip


def _evict_model_cache(cache_key) -> None:
    if cache_key is None:
        return
    with CACHE_LOCK:
        loaded_model = MODEL_CACHE.pop(cache_key, None)
    if loaded_model is not None:
        _free_loaded_model(loaded_model)


def _evict_clip_cache(cache_key) -> None:
    if cache_key is None:
        return
    with CACHE_LOCK:
        loaded_clip = CLIP_CACHE.pop(cache_key, None)
    if loaded_clip is not None:
        _free_loaded_clip(loaded_clip)


def _list_local_prxpixel_model_names() -> list[str]:
    try:
        candidates = folder_paths.get_filename_list(LOCAL_MODEL_FOLDER_KEY)
    except Exception:
        candidates = []

    names = []
    for candidate in candidates:
        try:
            full_path = folder_paths.get_full_path(LOCAL_MODEL_FOLDER_KEY, candidate)
        except Exception:
            full_path = None
        if not full_path or not full_path.lower().endswith(".safetensors"):
            continue
        if is_prxpixel_single_file(full_path):
            names.append(candidate)
    names = sorted(set(names))
    return names if names else ["<no prx pixel single-file models found>"]


class LoadPRXPixelModel:
    CATEGORY = "Lumina"
    FUNCTION = "load_model"
    RETURN_TYPES = (PRX_PIXEL_MODEL_TYPE,)
    RETURN_NAMES = ("model",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (_list_local_prxpixel_model_names(),),
                "dtype": (["auto", "bfloat16", "float16", "float32"], {"default": "auto"}),
                "device": ("STRING", {"default": "cuda", "multiline": False}),
                "force_reload": ("BOOLEAN", {"default": False}),
            }
        }

    def load_model(self, model_name: str, dtype: str, device: str, force_reload: bool):
        if not model_name or model_name.startswith("<no prx pixel"):
            raise RuntimeError("No PRX Pixel single-file transformer models were found in diffusion_models or unet.")

        model_path = folder_paths.get_full_path(LOCAL_MODEL_FOLDER_KEY, model_name)
        if not model_path:
            raise RuntimeError(f"Could not resolve model path for {model_name}.")

        loaded_model = _get_or_load_single_file_model(
            model_path=model_path,
            dtype_name=dtype,
            device_name=device,
            force_reload=force_reload,
        )
        return (loaded_model,)


class LoadPRXPixelModelHFRepo:
    CATEGORY = "Lumina"
    FUNCTION = "load_model"
    RETURN_TYPES = (PRX_PIXEL_MODEL_TYPE,)
    RETURN_NAMES = ("model",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_source": ("STRING", {"default": "Photoroom/prxpixel-t2i", "multiline": False}),
                "dtype": (["auto", "bfloat16", "float16", "float32"], {"default": "auto"}),
                "device": ("STRING", {"default": "cuda", "multiline": False}),
                "local_files_only": ("BOOLEAN", {"default": False}),
                "force_reload": ("BOOLEAN", {"default": False}),
            }
        }

    def load_model(self, model_source: str, dtype: str, device: str, local_files_only: bool, force_reload: bool):
        model_source = (model_source or "").strip()
        if not model_source:
            raise RuntimeError("Model source cannot be empty.")

        loaded_model = _get_or_load_repo_model(
            model_source=model_source,
            dtype_name=dtype,
            device_name=device,
            local_files_only=local_files_only,
            force_reload=force_reload,
        )
        return (loaded_model,)


class LoadPRXClipModelOnly:
    CATEGORY = "Lumina"
    FUNCTION = "load_clip"
    RETURN_TYPES = (PRX_PIXEL_CLIP_TYPE,)
    RETURN_NAMES = ("clip",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_source": ("STRING", {"default": "Photoroom/prxpixel-t2i", "multiline": False}),
                "dtype": (["auto", "bfloat16", "float16", "float32"], {"default": "auto"}),
                "device": ("STRING", {"default": "cuda", "multiline": False}),
                "local_files_only": ("BOOLEAN", {"default": False}),
                "force_reload": ("BOOLEAN", {"default": False}),
            }
        }

    def load_clip(self, model_source: str, dtype: str, device: str, local_files_only: bool, force_reload: bool):
        model_source = (model_source or "").strip()
        if not model_source:
            raise RuntimeError("Model source cannot be empty.")

        loaded_clip = _get_or_load_clip(
            model_source=model_source,
            dtype_name=dtype,
            device_name=device,
            local_files_only=local_files_only,
            force_reload=force_reload,
        )
        return (loaded_clip,)


class PRXPixelFP8Converter:
    CATEGORY = "Lumina"
    FUNCTION = "convert"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_source": ("STRING", {"default": "Photoroom/prxpixel-t2i", "multiline": False}),
                "output_folder": (["diffusion_models", "unet"], {"default": "diffusion_models"}),
                "output_filename": (
                    "STRING",
                    {"default": "prxpixel_transformer_fp8_mixed.safetensors", "multiline": False},
                ),
                "local_files_only": ("BOOLEAN", {"default": False}),
                "overwrite": ("BOOLEAN", {"default": False}),
            }
        }

    def convert(
        self,
        model_source: str,
        output_folder: str,
        output_filename: str,
        local_files_only: bool,
        overwrite: bool,
    ):
        model_source = (model_source or "").strip()
        output_filename = (output_filename or "").strip()
        if not model_source:
            raise RuntimeError("Model source cannot be empty.")
        if not output_filename:
            raise RuntimeError("Output filename cannot be empty.")
        if not output_filename.lower().endswith(".safetensors"):
            output_filename = f"{output_filename}.safetensors"

        output_roots = folder_paths.get_folder_paths(output_folder)
        if not output_roots:
            raise RuntimeError(f"Could not find a ComfyUI model directory for {output_folder}.")

        output_path = os.path.join(output_roots[0], output_filename)
        convert_prxpixel_source_to_fp8_mixed_file(
            model_source=model_source,
            output_path=output_path,
            local_files_only=local_files_only,
            overwrite=overwrite,
        )
        return (output_path,)


class LuminaPRXPixel:
    CATEGORY = "Lumina"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (PRX_PIXEL_MODEL_TYPE,),
                "clip": (PRX_PIXEL_CLIP_TYPE,),
                "prompt": (
                    "STRING",
                    {
                        "default": "A front-facing portrait of a lion in the golden savanna at sunset.",
                        "multiline": True,
                    },
                ),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": -1, "max": 0xFFFFFFFFFFFFFFFF}),
                "num_inference_steps": ("INT", {"default": 28, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16}),
                "cache_mode": (["keep_loaded", "clear_after_run"], {"default": "keep_loaded"}),
            }
        }

    def generate(
        self,
        model,
        clip,
        prompt: str,
        negative_prompt: str,
        seed: int,
        num_inference_steps: int,
        guidance_scale: float,
        width: int,
        height: int,
        cache_mode: str,
    ):
        prompt = (prompt or "").strip()
        if not prompt:
            raise RuntimeError("Prompt cannot be empty.")
        if model is None:
            raise RuntimeError("Model input is required.")
        if clip is None:
            raise RuntimeError("Clip input is required.")

        model_device = getattr(model, "preferred_device", None) or torch.device("cpu")
        clip_device = getattr(clip, "preferred_device", None) or torch.device("cpu")

        transformer_patcher = _ensure_model_patcher(model, model_device)
        text_encoder_patcher = _ensure_clip_patcher(clip, clip_device)

        model.transformer.eval()
        clip.text_encoder.eval()

        actual_seed = int(seed)
        if actual_seed < 0:
            actual_seed = torch.seed()
        generator_device = str(model_device) if model_device.type == "cuda" else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(actual_seed & 0xFFFFFFFFFFFFFFFF)

        progress = comfy.utils.ProgressBar(int(num_inference_steps))
        state = {"done": 0}

        def progress_callback(done: int, total: int):
            if getattr(mm, "interrupt_processing", False):
                raise RuntimeError("Generation interrupted by ComfyUI.")
            delta = done - state["done"]
            if delta > 0:
                progress.update(delta)
                state["done"] = done

        try:
            mm.soft_empty_cache()
        except Exception:
            pass

        def load_transformer_for_sampling():
            if model_device.type != "cpu":
                mm.load_models_gpu([transformer_patcher])

        def swap_clip_for_transformer():
            if clip_device.type != "cpu":
                try:
                    mm.unload_model_and_clones(text_encoder_patcher, unload_additional_models=False)
                except Exception:
                    try:
                        clip.text_encoder.to(_offload_device_for(clip_device, is_text_encoder=True))
                    except Exception:
                        pass
                try:
                    mm.soft_empty_cache()
                except Exception:
                    pass
            load_transformer_for_sampling()

        try:
            if model_device.type != "cpu":
                try:
                    mm.unload_model_and_clones(transformer_patcher, unload_additional_models=False)
                except Exception:
                    pass

            if clip_device.type != "cpu":
                mm.load_models_gpu([text_encoder_patcher])
                after_prompt_encoding_callback = swap_clip_for_transformer
            else:
                after_prompt_encoding_callback = load_transformer_for_sampling

            image = PRXPixelInference(model, clip).generate(
                prompt=prompt,
                negative_prompt=negative_prompt or "",
                height=int(height),
                width=int(width),
                num_inference_steps=int(num_inference_steps),
                guidance_scale=float(guidance_scale),
                generator=generator,
                device=model_device,
                text_encoder_device=clip_device,
                after_prompt_encoding_callback=after_prompt_encoding_callback,
                progress_callback=progress_callback,
            )
        finally:
            if cache_mode == "clear_after_run":
                _evict_model_cache(getattr(model, "cache_key", None))
                _evict_clip_cache(getattr(clip, "cache_key", None))

        if state["done"] < int(num_inference_steps):
            progress.update(int(num_inference_steps) - state["done"])

        return (image,)


NODE_CLASS_MAPPINGS = {
    "lumina_prx_pixel": LuminaPRXPixel,
    "load_prx_pixel_model": LoadPRXPixelModel,
    "load_prx_pixel_model_hf_repo": LoadPRXPixelModelHFRepo,
    "load_prx_clip_model_only": LoadPRXClipModelOnly,
    "prx_pixel_fp8_converter": PRXPixelFP8Converter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "lumina_prx_pixel": "PRX Pixel Sampling",
    "load_prx_pixel_model": "Load PRX Pixel model",
    "load_prx_pixel_model_hf_repo": "Load PRX Pixel model (HF Repo)",
    "load_prx_clip_model_only": "Load prx clip model only",
    "prx_pixel_fp8_converter": "fp8_converter",
}
