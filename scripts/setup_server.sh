#!/bin/bash
# ============================================================
# SERVER SETUP SCRIPT
# Ubuntu 24.04 with 2x NVIDIA T4-16GB GPUs
# ============================================================

set -e

echo "============================================================"
echo "🚀 FINGERPRINT BLOOD GROUP DETECTION - SERVER SETUP"
echo "============================================================"

# Update system
echo "📦 Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# Install essential packages
echo "📦 Installing essential packages..."
sudo apt-get install -y \
    git \
    git-lfs \
    curl \
    wget \
    htop \
    tmux \
    vim \
    build-essential \
    python3.10 \
    python3.10-venv \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0

# Install Node.js 20.x for frontend
echo "📦 Installing Node.js 20.x..."
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install NVIDIA CUDA drivers if not present
if ! command -v nvidia-smi &> /dev/null; then
    echo "📦 Installing NVIDIA drivers..."
    sudo apt-get install -y nvidia-driver-535 nvidia-cuda-toolkit
fi

# Verify GPU
echo "🔧 Verifying GPU setup..."
nvidia-smi

# Setup Git LFS
echo "📦 Setting up Git LFS..."
git lfs install

# Clone or update repository
REPO_DIR="Bhavanaaa"
if [ ! -d "$REPO_DIR" ]; then
    echo "📥 Cloning repository..."
    git clone https://github.com/Bhargavvz/Fingerprint.git $REPO_DIR
fi

cd $REPO_DIR

# Pull LFS files
echo "📥 Pulling Git LFS files..."
git lfs pull

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA 11.8
echo "🔥 Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install requirements
echo "📦 Installing project dependencies..."
pip install -r requirements.txt

# Verify PyTorch CUDA
echo "🔧 Verifying PyTorch CUDA..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}')"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p checkpoints outputs outputs/logs

# Install frontend dependencies
echo "📦 Installing frontend dependencies..."
cd frontend
npm install
cd ..

echo ""
echo "============================================================"
echo "✅ SERVER SETUP COMPLETE!"
echo "============================================================"
echo ""
echo "To start training, run:"
echo "  source venv/bin/activate"
echo "  ./scripts/run_training.sh"
echo ""
echo "To start the web application:"
echo "  ./scripts/run_app.sh"
echo ""
