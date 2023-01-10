FROM ubuntu:19.10

USER root

ARG DEBIAN_FRONTEND=noninteractive

RUN cp /etc/apt/sources.list /etc/apt/sources.list.bak
RUN sed -i -re 's/([a-z]{2}.)?archive.ubuntu.com|security.ubuntu.com/old-releases.ubuntu.com/g' /etc/apt/sources.list

RUN apt-get update && apt-get -y upgrade && apt-get autoremove && apt-get dist-upgrade
RUN apt-get install -y --no-install-recommends \
        build-essential \
        nvidia-cuda-toolkit \
        xdg-utils \
        apt-utils \
        cpio \
        curl \
        vim \
        git \
        lsb-release \
        pciutils \
        libgflags-dev \
        libboost-dev \
        libboost-log-dev \
        cmake \
        libx11-dev \
        libssl-dev \
        libsm6 \
        libxext6 \
        libxrender-dev \
        locales \
        libjpeg8-dev \
        libopenblas-dev \
        gnupg2 \
        protobuf-compiler \
        python3-dev \
        wget \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libusb-1.0-0-dev \
        sudo \
        python3.7 \
        python3-pip \
        python3-setuptools

RUN python3 --version
RUN pip3 --version

RUN mkdir -p /app/
RUN mkdir -p /app/object_segmentation/
RUN mkdir -p /app/object_segmentation/dataset/
RUN mkdir -p /app/object_segmentation/saved-model/

WORKDIR /app/object_segmentation/

COPY config.py /app/object_segmentation/
COPY data_generator.py /app/object_segmentation/
COPY models.py /app/object_segmentation/
COPY requirements.txt /app/object_segmentation/
COPY tensorboard_callbacks.py /app/object_segmentation/
COPY TrainModel.py /app/object_segmentation/
COPY utils.py /app/object_segmentation/

COPY dataset /app/object_segmentation/dataset/

RUN pip3 install git+https://github.com/lucasb-eyer/pydensecrf.git
RUN pip3 install -r requirements.txt

# RUN git clone https://github.com/tensorflow/tensorflow.git

CMD ["python3", "TrainModel.py", "unet", "500", "saved-model"]
