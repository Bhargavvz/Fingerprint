#!/bin/bash
# Simple training script - single GPU, no complications

echo "============================================================"
echo "🔬 FINGERPRINT BLOOD GROUP DETECTION - TRAINING"
echo "============================================================"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found!"
    exit 1
fi

# Activate virtual environment
if [ -d "venv" ]; then
    echo -e "${GREEN}✓ Activating virtual environment...${NC}"
    source venv/bin/activate
else
    echo -e "${YELLOW}⚠ No venv found, using system Python${NC}"
fi

# Check CUDA
python3 -c "import torch; print(f'🔧 CUDA Available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'🔧 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Create directories
mkdir -p outputs checkpoints outputs/logs

# Run training
echo ""
echo "🚀 Starting training..."
echo "============================================================"

python3 scripts/train_advanced.py --config configs/training_config.yaml

echo ""
echo "============================================================"
echo "📊 Training complete! Check outputs/ for results."
echo "============================================================"
