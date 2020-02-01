FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu16.04

ENV OPENCV_VERSION="3.4.6"

# Basic toolchain
RUN apt-get update && apt-get install -y \
        apt-utils \
        build-essential \
        git \
        wget \
        unzip \
        yasm \
        pkg-config \
        libcurl4-openssl-dev \
        zlib1g-dev \
        htop \
        cmake \
        nano \
        python3-pip \
        python3-dev \
        python3-tk \
        libx264-dev \
    && cd /usr/local/bin \
    && ln -s /usr/bin/python3 python \
    && pip3 install --upgrade pip \
    && apt-get autoremove -y

# Getting OpenCV dependencies available with apt
RUN apt-get update && apt-get install -y \
        libeigen3-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libjasper-dev \
        libswscale-dev \
        libavcodec-dev \
        libavformat-dev && \
    apt-get autoremove -y

# Getting other dependencies
RUN apt-get update && apt-get install -y \
        cppcheck \
        graphviz \
        doxygen \
        p7zip-full \
        libdlib18 \
        libdlib-dev && \
    apt-get autoremove -y


# Install OpenCV + OpenCV contrib (takes forever)
RUN mkdir -p /tmp && \
    cd /tmp && \
    wget --no-check-certificate -O opencv.zip https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip && \
    wget --no-check-certificate -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip && \
    unzip opencv.zip && \
    unzip opencv_contrib.zip && \
    mkdir opencv-${OPENCV_VERSION}/build && \
    cd opencv-${OPENCV_VERSION}/build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D WITH_CUDA=ON \
        -D CUDA_FAST_MATH=1 \
        -D WITH_CUBLAS=1 \
        -D WITH_FFMPEG=ON \
        -D WITH_OPENCL=ON \
        -D WITH_V4L=ON \
        -D WITH_OPENGL=ON \
        -D OPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib-${OPENCV_VERSION}/modules \
        .. && \
    make -j$(nproc) && \
    make install && \
    echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf && \
    ldconfig && \
    cd /tmp && \
    rm -rf opencv-${OPENCV_VERSION} opencv.zip opencv_contrib-${OPENCV_VERSION} opencv_contrib.zip && \
    cd /

# Compile and install ffmpeg from source
RUN git clone https://github.com/FFmpeg/FFmpeg /root/ffmpeg && \
    cd /root/ffmpeg && \
    ./configure --enable-gpl --enable-libx264 --enable-nonfree --disable-shared --extra-cflags=-I/usr/local/include && \
    make -j8 && make install -j8

# clone deep-high-resolution-net
ARG POSE_ROOT=/pose_root
RUN git clone https://github.com/leoxiaobin/deep-high-resolution-net.pytorch.git $POSE_ROOT
WORKDIR $POSE_ROOT
RUN mkdir output && mkdir log

RUN pip3 install -r requirements.txt && \
    pip3 install torch==1.1.0 \
    torchvision==0.3.0 \
    opencv-python \
    pillow==6.2.1

# build deep-high-resolution-net lib
WORKDIR $POSE_ROOT/lib
RUN make

# install COCO API
ARG COCOAPI=/cocoapi
RUN git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
WORKDIR $COCOAPI/PythonAPI
# Install into global site-packages
RUN make install

# download fastrrnn pretrained model for person detection
RUN python -c "import torchvision; model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True); model.eval()"

COPY inference.py $POSE_ROOT/tools
COPY inference-config.yaml $POSE_ROOT/
