# Базовый образ с поддержкой CUDA и PyTorch
FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    libgl1-mesa-glx \
    libglib2.0-0

RUN pip install jupyterlab==4.4.1 opencv-python timm

# Установка Python-зависимостей проекта
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Открытие порта для JupyterLab
EXPOSE 8888

# Стартовая команда
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--NotebookApp.token=melanoma123", "--no-browser"]
