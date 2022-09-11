FROM nvcr.io/nvidia/tensorflow:20.06-tf1-py3

RUN apt update && apt install -y python3-dev \
                                 python3-pip \
                                 imagemagick \
                                 libgl1-mesa-dev
RUN pip3 install --upgrade pip setuptools requests
RUN pip3 install nvidia-pyindex
RUN pip3 install nvidia-tensorflow
RUN pip3 install matplotlib \
                 imageio \
                 imageio-ffmpeg \
                 configargparse \
                 pillow \
                 opencv-python