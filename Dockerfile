FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
# keep PyTorch index consistent with cu121
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu121
# must use dotted SM versions
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# install torch that matches CUDA 12.1
RUN python3 -m pip install --upgrade pip \
 && python3 -m pip install --no-cache-dir \
    torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

WORKDIR /workspace
COPY . /workspace

# install your package, then the CUDA extensions (with no build isolation)
RUN python3 -m pip install --no-cache-dir . \
 && git clone --depth 1 https://github.com/qinglew/PCN-PyTorch.git /tmp/PCN-PyTorch \
 && python3 -m pip install --no-cache-dir --no-build-isolation /tmp/PCN-PyTorch/extensions/chamfer_distance \
 && rm -rf /tmp/PCN-PyTorch \
 && python3 -m pip install --no-cache-dir jupyterlab

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
