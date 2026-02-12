#(RTX 5060 Ti / CUDA 12)
FROM nvcr.io/nvidia/pytorch:24.03-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip uninstall -y opencv-python opencv-python-headless

RUN pip install --no-cache-dir \
    "opencv-python-headless==4.8.0.74" \
    prefetch_generator \
    yacs \
    tqdm \
    tensorboardX \
    imageio \
    scipy \
    onnxruntime-gpu \
    jupyter \
    matplotlib

WORKDIR /workspace

CMD ["/bin/bash"]