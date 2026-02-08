#!/bin/bash
# ============================================================
# TRAINING LAUNCH SCRIPT
# Multi-GPU training with 2x NVIDIA T4
# ============================================================

set -e

echo "============================================================"
echo "🔬 STARTING ADVANCED TRAINING"
echo "============================================================"

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Set environment variables for multi-GPU
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1

# Create output directories
mkdir -p checkpoints outputs outputs/logs

# Get GPU info
echo "🔧 GPU Configuration:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Training parameters
CONFIG="configs/training_config.yaml"
GPUS=2
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "📋 Configuration: $CONFIG"
echo "🖥️  GPUs: $GPUS"
echo "⏰ Started: $TIMESTAMP"
echo ""

# Start training with logging
echo "🚀 Launching training..."
python scripts/train_advanced.py \
    --config $CONFIG \
    --gpus $GPUS \
    2>&1 | tee outputs/logs/training_$TIMESTAMP.log

# Check if training was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ TRAINING COMPLETED SUCCESSFULLY!"
    echo "============================================================"
    
    # Show results
    echo ""
    echo "📊 Training outputs saved to: outputs/"
    ls -la outputs/
    
    echo ""
    echo "💾 Model checkpoints saved to: checkpoints/"
    ls -la checkpoints/
    
    # Push to Git with LFS
    echo ""
    read -p "📤 Push trained model to Git? (y/n): " push_model
    if [ "$push_model" = "y" ]; then
        ./scripts/push_models.sh
    fi
else
    echo ""
    echo "❌ Training failed! Check logs at outputs/logs/training_$TIMESTAMP.log"
    exit 1
fi
