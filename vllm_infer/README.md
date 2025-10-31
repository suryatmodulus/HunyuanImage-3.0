# Hunyuan Image3 API Service

Fast text-to-image generation with vLLM backend.

## Quick Start

### 1. Launch Server
```bash
sh vllm_infer/run_vllm_server.sh /path/to/model
```
Server starts at `http://localhost:8000`

### 2. Generate Images
```bash
python openai_client.py --prompt "your image description"
```

## Prerequisites

### 1. Using Docker (Recommended)

We provide a Dockerfile for easy setup: `docker/hyimage3_vllm.Dockerfile`.

```bash
# Build the Docker image
docker build -t hunyuan_image3_vllm -f docker/hyimage3_vllm.Dockerfile
# Run the Docker container
docker run --gpus all -it -p 8000:8000 hunyuan_image3_vllm \
    --mount type=bind,source=/path/to/model,target=/model \
    sh HunyuanImage-3.0/vllm_infer/run_vllm_server.sh /model
```

### 2. Manually Install Dependencies
```bash
# Install hunyuan_image_3 as a package
git clone https://github.com/Tencent-Hunyuan/HunyuanImage-3.0
cd HunyuanImage-3.0/
pip install -e .

# Install other dependencies
pip install apache-tvm-ffi==0.1.0b15
pip install diffusers transformers accelerate

# Install vLLM from specific commit
git clone --branch feature/hunyuan_image_3.0 https://github.com/kippergong/vllm.git
cd vllm
VLLM_USE_PRECOMPILED=1 pip install --editable .
cd ..

# Launch the vLLM server
sh vllm_infer/run_vllm_server.sh /path/to/model
```

### Verified Environment
The following environment has been tested and verified to work:
- **PyTorch**: 2.7.1, 2.8.0
- **CUDA**: 12.8
- **vLLM**: https://github.com/kippergong/vllm/tree/feature/hunyuan_image_3.0
- **hunyuan_image_3**: Latest version

## Examples

**Basic:**
```bash
python openai_client.py --prompt "panda eating bamboo in forest"
```

**Custom size:**
```bash
python openai_client.py --width 1280 --height 768 --prompt "sunset beach"
```

**Auto shape:**
```bash
python openai_client.py --bot-task "auto" --prompt "sunset beach"
```

**Arguments:**
- `--prompt`: Text description (required)
- `--width/--height`: Image size (default: 1024x1024)
- `--seed`: Reproducible results
- `--bot-task`: Task type (image/auto)
