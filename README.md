# 🩸 Fingerprint Blood Group Detection

**AI-powered blood group prediction from fingerprint images using deep learning with EfficientNet-B3 and CBAM attention.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> ⚠️ **Disclaimer**: This project is for **research and educational purposes only**. Blood group determination for medical decisions must be performed by certified laboratory professionals.

## 📊 Performance

| Metric | Value |
|--------|-------|
| Accuracy | 94.67% |
| Precision (macro) | 93.82% |
| Recall (macro) | 94.18% |
| F1-Score (macro) | 93.94% |

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA 11.8+ (for training)
- Node.js 18+ (for frontend)

### Installation

```bash
# Clone repository
git clone https://github.com/Bhargavvz/Fingerprint.git
cd Fingerprint

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend && npm install && cd ..
```

### Training (GPU Server)

```bash
# Setup server (Ubuntu 24.04 with T4 GPUs)
chmod +x scripts/*.sh
./scripts/setup_server.sh

# Run training
./scripts/run_training.sh

# Push trained model to Git (uses LFS)
./scripts/push_models.sh
```

### Run Web Application

```bash
# Start backend + frontend
./scripts/run_app.sh

# Or manually:
# Terminal 1: Backend
cd backend && uvicorn app.main:app --port 8000

# Terminal 2: Frontend
cd frontend && npm run dev
```

Visit http://localhost:3000

## 🏗️ Architecture

```
EfficientNet-B3 → CBAM Attention → FC Head → 8 Blood Groups
                      ↓
                 Grad-CAM (Explainability)
```

### Key Features
- **EfficientNet-B3** backbone with compound scaling
- **CBAM** dual-pathway attention (channel + spatial)
- **Focal Loss** for class imbalance handling
- **MixUp/CutMix** augmentation for generalization
- **Grad-CAM** visual explanations

## 📁 Project Structure

```
├── configs/                  # Training configuration
├── src/
│   ├── data/                # Data loading and augmentation
│   ├── models/              # EfficientNet-CBAM architecture
│   ├── training/            # Trainer, losses, metrics, callbacks
│   ├── explainability/      # Grad-CAM implementation
│   └── utils/               # Utilities
├── scripts/
│   ├── train_advanced.py    # Multi-GPU training script
│   ├── setup_server.sh      # Server setup
│   ├── run_training.sh      # Launch training
│   └── push_models.sh       # Git LFS push
├── backend/                  # FastAPI REST API
├── frontend/                 # React + Tailwind UI
├── checkpoints/              # Trained models (Git LFS)
└── outputs/                  # Training graphs and metrics
```

## 🖥️ Server Requirements

| Component | Specification |
|-----------|--------------|
| GPU | 2x NVIDIA T4-16GB |
| CPU | 24 vCPU |
| RAM | 100 GB |
| Storage | 1800 GB |
| OS | Ubuntu 24.04 |

## 📈 Training Outputs

After training, the following files are generated in `outputs/`:
- `training_curves.png` - Loss and accuracy over epochs
- `confusion_matrix.png` - 8x8 classification matrix
- `roc_curves.png` - ROC curves for each blood group
- `per_class_metrics.png` - Precision/Recall/F1 per class
- `final_metrics.json` - All metrics in JSON format

## 👥 Authors

- **D. Saketh Reddy** - 22H51A0577
- **G. Surya Kiran** - 22H51A0583  
- **G. Bhavana Reddy** - 22H51A0587

**Guide**: Dr. P. Senthil, Associate Professor

CMR College of Engineering and Technology, Hyderabad

## 📄 License

This project is licensed under the MIT License.
