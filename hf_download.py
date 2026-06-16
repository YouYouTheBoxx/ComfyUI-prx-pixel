import os
import re

from huggingface_hub import snapshot_download

try:
    import folder_paths
except Exception:  # pragma: no cover - standalone usage
    folder_paths = None


PRX_PIXEL_HF_ROOT_DIRNAME = "prx_pixel_hf"
PRX_PIXEL_HF_CACHE_DIRNAME = "_hub_cache"

TRANSFORMER_ALLOW_PATTERNS = [
    "model_index.json",
    "scheduler/*",
    "transformer/*",
    "README.md",
    "LICENSE",
    "NOTICE",
]

CLIP_ALLOW_PATTERNS = [
    "model_index.json",
    "text_encoder/*",
    "tokenizer/*",
    "README.md",
    "LICENSE",
    "NOTICE",
]


def _sanitize_repo_id(repo_id: str) -> str:
    repo_id = (repo_id or "").strip()
    if not repo_id:
        return "empty-repo-id"
    return re.sub(r'[^A-Za-z0-9._-]+', "--", repo_id)


def _fallback_models_dir() -> str:
    custom_nodes_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    comfy_root = os.path.dirname(custom_nodes_dir)
    candidate = os.path.join(comfy_root, "models")
    if os.path.isdir(candidate):
        return candidate
    return os.path.join(custom_nodes_dir, "_prx_pixel_models")


def get_comfy_models_dir() -> str:
    models_dir = getattr(folder_paths, "models_dir", None) if folder_paths is not None else None
    if models_dir:
        return models_dir
    return _fallback_models_dir()


def get_prxpixel_hf_root() -> str:
    return os.path.join(get_comfy_models_dir(), PRX_PIXEL_HF_ROOT_DIRNAME)


def resolve_hf_snapshot_to_models_dir(
    model_source: str,
    allow_patterns: list[str],
    local_files_only: bool,
    component: str,
) -> str:
    if os.path.isdir(model_source):
        return os.path.abspath(model_source)

    hf_root = get_prxpixel_hf_root()
    cache_dir = os.path.join(hf_root, PRX_PIXEL_HF_CACHE_DIRNAME)
    local_dir = os.path.join(hf_root, component, _sanitize_repo_id(model_source))
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(local_dir, exist_ok=True)

    return snapshot_download(
        repo_id=model_source,
        allow_patterns=allow_patterns,
        local_files_only=local_files_only,
        cache_dir=cache_dir,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )


def resolve_transformer_model_directory(model_source: str, local_files_only: bool) -> str:
    return resolve_hf_snapshot_to_models_dir(
        model_source=model_source,
        allow_patterns=TRANSFORMER_ALLOW_PATTERNS,
        local_files_only=local_files_only,
        component="transformer",
    )


def resolve_clip_model_directory(model_source: str, local_files_only: bool) -> str:
    return resolve_hf_snapshot_to_models_dir(
        model_source=model_source,
        allow_patterns=CLIP_ALLOW_PATTERNS,
        local_files_only=local_files_only,
        component="clip",
    )
