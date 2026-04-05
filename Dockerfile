FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_INSTALL_PATH=/usr/local/cuda-11.7 \
    PTXAS_CUDA_INSTALL_PATH=/usr/local/cuda-11.7

SHELL ["/bin/bash", "-lc"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    git \
    build-essential \
    xutils-dev \
    bison \
    flex \
    zlib1g-dev \
    libglu1-mesa-dev \
    libxi-dev \
    libxmu-dev \
    freeglut3-dev \
    python3 \
    python3-numpy \
    python3-matplotlib \
    rsync && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/DICE-test-collection
COPY . /opt/DICE-test-collection

RUN test -d dice_gpgpu-sim && \
    test -d gpgpu-sim_distribution && \
    test -d dice-test-gpu-rodinia && \
    test -d gpu-rodinia && \
    test -d integrated_test

COPY docker/docker-entrypoint.sh /usr/local/bin/dice-entrypoint
RUN chmod +x /usr/local/bin/dice-entrypoint

ENTRYPOINT ["/usr/local/bin/dice-entrypoint"]
CMD ["shell"]
