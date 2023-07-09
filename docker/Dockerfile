# Define base image.
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
# FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
# metainformation
LABEL org.opencontainers.image.version = "0.1.18"
LABEL org.opencontainers.image.source = "https://github.com/nerfstudio-project/nerfstudio"
LABEL org.opencontainers.image.licenses = "Apache License 2.0"
LABEL org.opencontainers.image.base.name="docker.io/library/nvidia/cuda:11.8.0-devel-ubuntu22.04"
# LABEL org.opencontainers.image.base.name="docker.io/library/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04"

# Variables used at build time.
## CUDA architectures, required by Colmap and tiny-cuda-nn.
## NOTE: All commonly used GPU architectures are included and supported here. To speedup the image build process remove all architectures but the one of your explicit GPU. Find details here: https://developer.nvidia.com/cuda-gpus (8.6 translates to 86 in the line below) or in the docs.
# ARG CUDA_ARCHITECTURES="90, 89, 86, 80, 75, 70, 61, 52, 37"
ARG CUDA_ARCHITECTURES="90;89;86;80;75;70;61;52;37"

# Set environment variables.
## Set non-interactive to prevent asking for user inputs blocking image creation.
ENV DEBIAN_FRONTEND=noninteractive
## Set timezone as it is required by some packages.
ENV TZ=Europe/Berlin
## CUDA Home, required to find CUDA in some packages.
ENV CUDA_HOME="/usr/local/cuda"


# Install required apt packages and clear cache afterwards.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    ffmpeg \
    git \
    libatlas-base-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-program-options-dev \
    libboost-system-dev \
    libboost-test-dev \
    libhdf5-dev \
    libcgal-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libgflags-dev \
    libglew-dev \
    libgoogle-glog-dev \
    libmetis-dev \
    libprotobuf-dev \
    libqt5opengl5-dev \
    libsqlite3-dev \
    libsuitesparse-dev \
    nano \
    protobuf-compiler \
    python-is-python3 \
    python3.10-dev \
    python3-pip \
    qtbase5-dev \
    sudo \
    vim-tiny \
    wget && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and install packages.==22.0.2
RUN python3.10 -m pip install --upgrade pip setuptools pathtools promise pybind11 lit
# Install pytorch and submodules
RUN python3.10 -m pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
# Install tynyCUDNN (we need to set the target architectures as environment variable first).
ENV TCNN_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}
RUN python3.10 -m pip install git+https://github.com/NVlabs/tiny-cuda-nn.git@v1.6#subdirectory=bindings/torch


RUN wget https://github.com/Kitware/CMake/releases/download/v3.23.2/cmake-3.23.2.tar.gz
# unzip
RUN tar zxvf cmake-3.23.2.tar.gz
# install a dependency package 
RUN apt update
RUN apt install build-essential -y
RUN apt-get install libssl-dev -y
# build CMake 
RUN cd cmake-3.23.2
RUN ./bootstrap
RUN make
RUN make install
RUN cd ..

# the command below might not be needed
RUN export LIBRARY_PATH="/usr/local/cuda/lib64:$LIBRARY_PATH"
RUN pip install nerfstudio
# RUN ns-install-cli