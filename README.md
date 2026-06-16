# lumina_prx_pixel

ComfyUI custom nodes for `Photoroom/prxpixel-t2i`.

These nodes use a local PRXPixel runtime instead of the unreleased diffusers PRXPixel pipeline.

## Nodes

- `Load PRX Pixel model (HF Repo)`
  Loads the PRX Pixel transformer from a Hugging Face repo id or local diffusers-style model directory.

- `Load prx clip model only`
  Loads only the tokenizer and Qwen text encoder from a Hugging Face repo id or local diffusers-style model directory.

- `fp8_converter`
  Merges the sharded PRX Pixel transformer into one single-file mixed fp8 safetensors checkpoint for `diffusion_models` or `unet`.

- `Load PRX Pixel model`
  Loads one of those converted single-file PRX Pixel transformer checkpoints from `diffusion_models` or `unet`.

- `PRX Pixel Sampling`
  Runs generation from a PRX Pixel transformer input plus a PRX clip input.

## What changed

- The main generation node no longer accepts `model_source`.
- The transformer and clip are loaded separately.
- Transformer single-file checkpoints can be generated locally and then loaded through ComfyUI model folders.
- Text encoder and transformer loading are tied into ComfyUI model management so unload/reload works with Comfy's VRAM policy.
- Hugging Face downloads are staged under `ComfyUI/models/prx_pixel_hf` so cache and temp files stay inside the Comfy model tree.

## Typical workflows

### Regular HF workflow

1. Use `Load PRX Pixel model (HF Repo)` with `Photoroom/prxpixel-t2i`.
2. Use `Load prx clip model only` with the same source.
3. Feed both into `PRX Pixel Sampling`.

### Local mixed-fp8 workflow

1. Run `fp8_converter` once against `Photoroom/prxpixel-t2i`.
2. It writes a single `.safetensors` transformer file into `diffusion_models` or `unet`.
3. Use `Load PRX Pixel model` to load that local transformer.
4. Use `Load prx clip model only` for the text encoder.
5. Feed both into `PRX Pixel Sampling`.

The clip-only loader only downloads:

- `model_index.json`
- `tokenizer/*`
- `text_encoder/*`

## Notes about the converter

- The converter streams shard-by-shard so it does not need the whole transformer in RAM at once.
- The produced file stores a conservative mixed-fp8 recipe:
  large attention and MLP projection weights are stored in fp8, while more sensitive tensors stay in higher precision.
- The runtime loads those single-file checkpoints back into the local PRX implementation directly.

## Dependencies

This node avoids the unreleased `PRXPixelPipeline` dependency. It needs:

- `transformers>=4.57`
- `accelerate`
- `huggingface_hub`
- `safetensors`
- a stable `diffusers` install that already includes `FlowMatchEulerDiscreteScheduler`

You can install that with `install_requirements.bat`
or with the Python inside `ComfyUI\.venv`.
