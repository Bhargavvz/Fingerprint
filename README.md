# Fingerprint-Based Blood Group Detection

A deep learning system for predicting blood groups from fingerprint images using hybrid EfficientNet-B3 with CBAM attention mechanism and Explainable AI.

## 🚀 Features

- **Hybrid Deep Learning Model**: EfficientNet-B3 + CBAM attention mechanism
- **Explainable AI**: Grad-CAM visualizations for model interpretability
- **Production-Ready API**: FastAPI backend with comprehensive endpoints
- **Modern Frontend**: React + Tailwind CSS with beautiful UI
- **Docker Deployment**: Complete containerization with docker-compose

## ⚠️ Disclaimer

> **This is an academic research project. It is NOT intended for medical diagnosis.**
> Blood group determination requires proper laboratory testing by qualified healthcare professionals.

## 📁 Project Structure

```
Bhavanaaa/
├── Dataset/                     # Fingerprint images (8 blood group classes)
├── src/                         # Core ML source code
│   ├── data/                    # Dataset and augmentation
│   ├── models/                  # Model architecture
│   ├── training/                # Training pipeline
│   ├── explainability/          # Grad-CAM and explanations
│   └── utils/                   # Utilities
├── scripts/                     # Training and evaluation scripts
├── backend/                     # FastAPI application
├── frontend/                    # React application
├── configs/                     # Configuration files
├── docs/                        # Documentation
├── checkpoints/                 # Saved models
└── outputs/                     # Training outputs and logs
```

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- CUDA 11.8+ (for GPU training)
- Node.js 18+ (for frontend)

### Setup
```bash
# Clone repository
cd Bhavanaaa

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# For development
pip install -r requirements-dev.txt
```

## 📊 Dataset

The dataset contains fingerprint images organized by blood group:
- **Classes**: A+, A-, B+, B-, AB+, AB-, O+, O-
- **Total Images**: ~6000
- **Format**: BMP images
- **Split**: 70% train, 15% validation, 15% test

## 🚂 Training

```bash
# Basic training
python scripts/train.py

# With custom config
python scripts/train.py --config configs/training_config.yaml

# Resume training
python scripts/train.py --resume checkpoints/last.pt
```

## 🔍 Evaluation

```bash
# Evaluate model
python scripts/evaluate.py --checkpoint checkpoints/best.pt

# Generate predictions
python scripts/predict.py --image path/to/fingerprint.bmp
```

## 🌐 API

```bash
# Start backend
cd backend
uvicorn app.main:app --reload --port 8000

# API documentation at http://localhost:8000/docs
```

## 💻 Frontend

```bash
# Start frontend
cd frontend
npm install
npm run dev

# Access at http://localhost:3000
```

## 🐳 Docker Deployment

```bash
# Build and run all services
docker-compose up --build

# Access frontend at http://localhost:3000
# Access API at http://localhost:8000
```

## 📈 Results

| Metric | Value |
|--------|-------|
| Accuracy | TBD |
| Precision | TBD |
| Recall | TBD |
| F1-Score | TBD |

## 📚 Documentation

- [IEEE Documentation](docs/IEEE_Documentation.md)
- [API Reference](docs/API_Documentation.md)
- [Architecture](docs/architecture_diagram.md)

## 📝 License

This project is for academic purposes only.

## 👥 Authors

Academic Major Project - Fingerprint-Based Blood Group Detection
