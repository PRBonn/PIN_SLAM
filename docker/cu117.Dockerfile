FROM nvidia/cuda:11.7.1-base-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive 
ENV NVENCODE_CFLAGS="-I/usr/local/cuda/include"
ENV CV_VERSION=4.2.0
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"

# Install python3.10 on Ubuntu 20.04 using deadsnakes PPA
RUN apt-get update && \
    apt-get install software-properties-common -y && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python3.10 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Get all dependencies
RUN apt-get update && apt-get install -y \
    git zip unzip libssl-dev libcairo2-dev lsb-release libgoogle-glog-dev libgflags-dev libatlas-base-dev libeigen3-dev software-properties-common \
    build-essential cmake pkg-config libapr1-dev autoconf automake libtool curl libc6 libboost-all-dev debconf libomp5 libstdc++6 \
    libqt5core5a libqt5xml5 libqt5gui5 libqt5widgets5 libqt5concurrent5 libqt5opengl5 libcap2 libusb-1.0-0 libatk-adaptor neovim \
    python3-tornado python3-dev python3-numpy python3-virtualenv libpcl-dev libgoogle-glog-dev libgflags-dev libatlas-base-dev \
    libsuitesparse-dev python3-pcl pcl-tools libgtk2.0-dev libavcodec-dev libavformat-dev libswscale-dev libtbb2 libtbb-dev libjpeg-dev \
    libpng-dev libtiff-dev libdc1394-22-dev xfce4-terminal \
    libclang-dev \
    libatk-bridge2.0 \
    libfontconfig1-dev \
    libfreetype6-dev \
    libglib2.0-dev \
    libgtk-3-dev \
    libssl-dev \
    libxcb-render0-dev \
    libxcb-shape0-dev \
    libxcb-xfixes0-dev \
    libxkbcommon-dev \
    patchelf \
    wget &&\
    rm -rf /var/lib/apt/lists/*

# OpenCV with CUDA support
WORKDIR /opencv
RUN git clone https://github.com/opencv/opencv.git -b $CV_VERSION &&\
    git clone https://github.com/opencv/opencv_contrib.git -b $CV_VERSION

# While using OpenCV 4.2.0 we have to apply some fixes to ensure that CUDA is fully supported, thanks @https://github.com/gismo07 for this fix
RUN mkdir opencvfix && cd opencvfix &&\
    git clone https://github.com/opencv/opencv.git -b 4.5.2 &&\
    cd opencv/cmake &&\
    cp -r FindCUDA /opencv/opencv/cmake/ &&\
    cp FindCUDA.cmake /opencv/opencv/cmake/ &&\
    cp FindCUDNN.cmake /opencv/opencv/cmake/ &&\
    cp OpenCVDetectCUDA.cmake /opencv/opencv/cmake/
 
WORKDIR /opencv/opencv/build

RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D BUILD_EXAMPLES=OFF \
-D INSTALL_PYTHON_EXAMPLES=OFF \
-D INSTALL_C_EXAMPLES=OFF \
-D PYTHON_EXECUTABLE=$(which python2) \
-D PYTHON3_EXECUTABLE=$(which python3) \
-D PYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
-D PYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
-D BUILD_opencv_python2=ON \
-D BUILD_opencv_python3=ON \
-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules/ \
-D WITH_GSTREAMER=ON \
-D WITH_CUDA=ON \
-D ENABLE_PRECOMPILED_HEADERS=OFF \
.. &&\
make -j$(nproc) &&\
make install &&\
ldconfig &&\
rm -rf /opencv

WORKDIR /
ENV OpenCV_DIR=/usr/share/OpenCV

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --upgrade setuptools

# PyTorch for CUDA 11.7
RUN python3 -m pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu117

# PIN SLAM
RUN mkdir -p /src
WORKDIR /src/
RUN git clone https://github.com/PRBonn/PIN_SLAM.git
WORKDIR /src/PIN_SLAM

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --upgrade properties
# Fix uninstallation-issue with Blinker 1.4
RUN python3 -m pip install --ignore-installed blinker
RUN python3 -m pip install -r ./requirements.txt

# Load the KITTI data 
RUN mkdir -p /src/PIN_SLAM/data
WORKDIR /src/PIN_SLAM/data

RUN wget -O kitti_example.tar.gz -c https://uni-bonn.sciebo.de/s/Ycl28f1Cppghvjm/download
RUN tar -xvf kitti_example.tar.gz
RUN rm kitti_example.tar.gz

# Fix unknown ownership
RUN git config --global --add safe.directory '*'

# Finish installation
WORKDIR /src/PIN_SLAM

RUN apt-get clean

ENV NVIDIA_VISIBLE_DEVICES="all" \
    OpenCV_DIR=/usr/share/OpenCV \
    NVIDIA_DRIVER_CAPABILITIES="video,compute,utility,graphics" \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib:/usr/lib:/usr/local/lib \
    QT_GRAPHICSSYSTEM="native"
