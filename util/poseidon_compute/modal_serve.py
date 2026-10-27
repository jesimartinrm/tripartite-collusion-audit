"""vLLM model server on Modal — an OpenAI-compatible endpoint for experiments Nebius can't serve."""
import os
import subprocess
import modal

MODEL = os.environ.get("MODAL_SERVE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
GPU = os.environ.get("MODAL_SERVE_GPU", "A10G")
VLLM_PORT = 8000

app = modal.App("hackathon-serve")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm==0.6.6.post1",
        "transformers>=4.45.2,<4.49.0",
        "tokenizers>=0.19.1,<0.22.0",
        "openai>=1.52.0,<1.65.0",
        "pydantic>=2.9.0,<2.11.0",
        "fastapi>=0.107.0,<0.115.0",
        "uvicorn[standard]>=0.29.0,<0.35.0",
        "outlines==0.1.11",
        "xgrammar==0.1.11",
        "huggingface_hub[hf_transfer]==0.26.2",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

@app.function(
    image=image,
    gpu=GPU,
    timeout=24 * 3600,
    scaledown_window=300,
)
@modal.web_server(port=VLLM_PORT, startup_timeout=900)
def serve():
    """Launch vLLM's OpenAI-compatible server; Modal proxies it to a public URL."""
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--served-model-name", MODEL,
        "--max-model-len", "8192",
        "--enforce-eager",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "hermes",
    ]
    api_key = os.environ.get("MODAL_SERVE_API_KEY")
    if api_key and api_key != "unused":
        cmd += ["--api-key", api_key]
    subprocess.Popen(" ".join(cmd), shell=True)
