# syntax=docker/dockerfile:1

FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY . /workspace

RUN python -m pip install --upgrade pip \
    && python -m pip install . \
    && git clone --depth 1 https://github.com/qinglew/PCN-PyTorch.git /tmp/PCN-PyTorch \
    && python -m pip install /tmp/PCN-PyTorch/chamfer_distance \
    && python -m pip install /tmp/PCN-PyTorch/pointnet2_ops_lib \
    && rm -rf /tmp/PCN-PyTorch \
    && python -m pip install jupyterlab

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
