FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model conversion and download scripts
COPY pytorch_model.py .
COPY convert_to_onnx.py .

# Download model weights
RUN wget -O pytorch_model_weights.pth \
    https://www.dropbox.com/s/b7641ryzmkceoc9/pytorch_model_weights.pth?dl=1

# Convert model to ONNX
RUN python convert_to_onnx.py --weights pytorch_model_weights.pth --output model.onnx

# Copy application code
COPY model.py .
COPY app.py .

# Cerebrium requires the app to be in the root directory
ENV PYTHONPATH=/app

# Clean up PyTorch files to reduce image size
RUN rm -f pytorch_model_weights.pth pytorch_model.py convert_to_onnx.py

# The entrypoint is handled by Cerebrium