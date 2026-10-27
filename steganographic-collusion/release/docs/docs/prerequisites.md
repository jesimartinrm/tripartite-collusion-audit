# System Prerequisites

## Operating System
- Linux (Ubuntu 22.04+) or macOS 13+ verified.
- Windows users should leverage WSL2 with Ubuntu 22.04 for consistent paths.

## Python Environment
- Python 3.11.
- Recommended to install via `pyenv` or `conda` for reproducible builds.
- Install the full dependency set with `pip install -r requirements.txt` (numerical stack, YAML, the OpenAI-compatible client for hosted inference, and the local `distilgpt2` backend). `docs/requirements-extras.txt` is only the minimal numerical/test subset (`pytest`, `numpy`, `pandas`, `matplotlib`) and is not sufficient for the configs or hosted runs.

## Hardware
- CPU-only execution is sufficient: sweeps were profiled on 24 vCPU / 64 GB RAM machines.
- GPU acceleration is optional and currently unused; no CUDA-specific kernels are invoked.
- Disk footprint for full regeneration: ~5 GB (transcripts + metadata).

## External Dependencies
- No external services are contacted; all datasets are generated on the fly.
- Network access is optional and only required for publishing artifacts.

## Quick Verification
```bash
python --version  # should report 3.11.x
scripts/ci_check.sh
```

The CI script compiles Python modules and runs a minimal `pytest` suite to ensure the environment matches the expected configuration.
