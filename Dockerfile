FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
# keep PyTorch index consistent with cu124
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124
# must use dotted SM versions
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# install torch that matches CUDA 12.4
RUN python3 -m pip install --upgrade pip \
 && python3 -m pip install --no-cache-dir \
    torch==2.4.1+cu124 torchvision==0.19.1+cu124 torchaudio==2.4.1+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

WORKDIR /workspace
COPY . /workspace

# 1. Install your package
RUN python3 -m pip install --no-cache-dir .

# 2. Cleanup Scipy 
RUN python3 -c "import importlib, pathlib, shutil; p = pathlib.Path(importlib.import_module('scipy').__file__).parent; [shutil.rmtree(p / r) for r in ['tests', 'stats/tests'] if (p / r).exists()]"

# 3. Install PCN-PyTorch extensions and JupyterLab
RUN git clone --depth 1 https://github.com/qinglew/PCN-PyTorch.git /tmp/PCN-PyTorch \
 && python3 -m pip install --no-cache-dir --no-build-isolation /tmp/PCN-PyTorch/extensions/chamfer_distance \
 && rm -rf /tmp/PCN-PyTorch \
 && python3 -m pip install --no-cache-dir jupyterlab

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]