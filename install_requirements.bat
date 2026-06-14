@echo off
setlocal

set "PYTHON=..\..\.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo Could not find ComfyUI's venv Python at "%PYTHON%".
    exit /b 1
)

"%PYTHON%" -m pip install --upgrade "transformers>=4.57" accelerate huggingface_hub safetensors
if errorlevel 1 exit /b 1

echo lumina_prx_pixel dependencies installed.
