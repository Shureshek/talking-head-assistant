FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-dev git ffmpeg libsm6 libxext6 \
    build-essential libssl-dev libffi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# === 1. Устанавливаем PyTorch 2.0.1 + cu118 (точно как требует MuseTalk) ===
RUN pip install --no-cache-dir --default-timeout=1000\
    torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

# === 2. Копируем и устанавливаем все зависимости MuseTalk ===
COPY requirements_musetalk.txt .
RUN pip install --no-cache-dir -r requirements_musetalk.txt

# === 3. MMLab экосистема (самое важное!) ===
RUN pip install --no-cache-dir -U openmim
RUN mim install mmengine
RUN mim install "mmcv==2.0.1"
RUN mim install "mmdet==3.1.0"
RUN mim install "mmpose==1.1.0"


# === 4. Копируем весь код MuseTalk + нашу обёртку ===
COPY musetalk_core/ /app/musetalk_core/

# Volumes
RUN mkdir -p /app/volumes/models /app/volumes/temp /app/volumes/result_video


WORKDIR /app/musetalk_core

EXPOSE 8001

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]