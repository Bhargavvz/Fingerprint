#!/bin/bash
# ============================================================
# PUSH TRAINED MODELS TO GIT WITH LFS
# ============================================================

set -e

echo "============================================================"
echo "📤 PUSHING TRAINED MODELS TO GIT (LFS)"
echo "============================================================"

# Check if Git LFS is installed
if ! command -v git-lfs &> /dev/null; then
    echo "❌ Git LFS not installed. Installing..."
    sudo apt-get install git-lfs
    git lfs install
fi

# Check for model files
if [ ! -f "checkpoints/best_model.pt" ]; then
    echo "❌ No trained model found at checkpoints/best_model.pt"
    exit 1
fi

# Show model file size
echo ""
echo "📊 Model files to push:"
ls -lh checkpoints/*.pt 2>/dev/null || echo "  No .pt files found"
echo ""

# Track files with LFS
echo "🔧 Ensuring files are tracked with LFS..."
git lfs track "checkpoints/*.pt"
git lfs track "outputs/*.png"
git lfs track "outputs/*.json"

# Add files
echo "📁 Adding files..."
git add .gitattributes
git add checkpoints/
git add outputs/

# Show status
echo ""
echo "📋 Git status:"
git status --short

# Get commit message
TIMESTAMP=$(date +%Y-%m-%d_%H:%M:%S)
ACCURACY=$(cat outputs/final_metrics.json 2>/dev/null | grep -o '"test_accuracy": [0-9.]*' | cut -d' ' -f2 || echo "unknown")

COMMIT_MSG="🎯 Model checkpoint - Accuracy: ${ACCURACY} - ${TIMESTAMP}"
echo ""
echo "💬 Commit message: $COMMIT_MSG"

# Commit
git commit -m "$COMMIT_MSG"

# Push with LFS
echo ""
echo "📤 Pushing to remote..."
git push origin main

echo ""
echo "============================================================"
echo "✅ MODELS PUSHED SUCCESSFULLY!"
echo "============================================================"
echo ""
echo "🔗 LFS files tracked:"
git lfs ls-files
