# Dockerfile for Thesis: Explainability in Transformer-based PPM
# Builds scipy/scikit-learn from source for numpy 1.x compatibility

FROM python:3.11-slim

# Install system dependencies for building scipy/sklearn
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    libopenblas-dev \
    pkg-config \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Create constraints file for numpy version
RUN echo "numpy==1.26.4" > /tmp/constraints.txt

# Install numpy first (needed for building scipy/sklearn)
RUN pip install --no-cache-dir numpy==1.26.4

# Install processtransformer without deps to avoid pulling newer tensorflow
RUN pip install --no-cache-dir --no-deps \
    "processtransformer @ git+https://github.com/Zaharah/processtransformer.git@3f041e4eae9a2efcea0da02166e98efaa953801c"

# Build scipy and scikit-learn from source against numpy 1.x
RUN pip install --no-cache-dir --no-binary scipy -c /tmp/constraints.txt "scipy>=1.12.0,<2.0.0"
RUN pip install --no-cache-dir --no-binary scikit-learn -c /tmp/constraints.txt scikit-learn==1.6.1

# Install remaining dependencies (keep --no-binary for scipy/sklearn to prevent overwriting source builds)
RUN pip install --no-cache-dir --no-binary scipy,scikit-learn -r requirements.txt

# Copy project files
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2

# Default command
CMD ["python", "main.py"]