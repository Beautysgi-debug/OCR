FROM python:3.10-slim

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libsndfile1 \
    ffmpeg \
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY tests .

# 创建目录
RUN mkdir -p uploads logs

EXPOSE 8000

# 环境变量
ENV PYTHONUNBUFFERED=1
ENV OPENAI_API_KEY=""

CMD ["python", "main.py"]