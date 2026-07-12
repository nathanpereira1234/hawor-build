FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    TORCH_CUDA_ARCH_LIST="7.5" \
    FORCE_CUDA=1 \
    MAX_JOBS=2 \
    PYOPENGL_PLATFORM=egl \
    MPLBACKEND=Agg \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

RUN apt-get update && apt-get install -y \
    git wget build-essential ninja-build \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    libegl1 libgles2 freeglut3-dev ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pre-requirements.txt requirements.txt ./

RUN sed -i '/^torch==/d; /^torchvision==/d' pre-requirements.txt \
 && sed -i '/^chumpy/d; /^torch-scatter/d' requirements.txt

RUN pip install --no-cache-dir gradio==4.44.0 huggingface_hub
RUN pip install --no-cache-dir -r pre-requirements.txt

RUN pip install --no-cache-dir torch-scatter==2.1.2 \
    -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

RUN pip install --no-cache-dir git+https://github.com/mattloper/chumpy \
 && python -c "\
import re,pathlib,chumpy; \
p=pathlib.Path(chumpy.__file__).parent; \
[f.write_text(re.sub(r'\bnp\.(bool|object|float|int|complex)\b', r'\1', f.read_text())) for f in p.rglob('*.py')]"

COPY . .
RUN chmod -R 777 /app

RUN rm -rf ./thirdparty/DROID-SLAM/thirdparty/eigen && \
    git clone --depth 1 --branch 3.4.0 \
      https://gitlab.com/libeigen/eigen.git \
      ./thirdparty/DROID-SLAM/thirdparty/eigen

RUN rm -rf ./thirdparty/DROID-SLAM/thirdparty/lietorch/eigen && \
    git clone --depth 1 --branch 3.4.0 \
      https://gitlab.com/libeigen/eigen.git \
      ./thirdparty/DROID-SLAM/thirdparty/lietorch/eigen

RUN pip install --no-cache-dir ./thirdparty/DROID-SLAM
RUN pip install --no-cache-dir ./thirdparty/DROID-SLAM/thirdparty/lietorch
RUN pip install --no-cache-dir git+https://github.com/facebookresearch/pytorch3d.git@stable

# LAST: nothing may override the hub pin. gradio 4.44 needs HfFolder (removed in hub 1.x).
# `spaces` is imported by scripts/ even though we don't use ZeroGPU — install it as a no-op.
RUN pip install --no-cache-dir spaces && \
    pip install --no-cache-dir --force-reinstall \
      "huggingface_hub==0.25.2" "matplotlib==3.8.4"

EXPOSE 7860
CMD ["python", "app.py"]
