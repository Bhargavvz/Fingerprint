# AWS Deployment Guide for BloodType AI

## Prerequisites

- AWS EC2 instance (Ubuntu 22.04 recommended, t3.medium or larger for ML)
- Docker and Docker Compose installed
- Git installed
- Port 3000 open in Security Group

---

## Step 1: Connect to Your AWS Server

```bash
ssh -i your-key.pem ubuntu@your-ec2-public-ip
```

---

## Step 2: Install Docker (if not installed)

```bash
# Update packages
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Logout and login again for group changes
exit
```

---

## Step 3: Clone the Repository

```bash
# SSH back in
ssh -i your-key.pem ubuntu@your-ec2-public-ip

# Clone the repo
git clone https://github.com/Bhargavvz/Fingerprint.git
cd Fingerprint
```

---

## Step 4: Set Up Model Checkpoint

Make sure your trained model is in the `checkpoints/` directory:

```bash
# Check if model exists
ls -la checkpoints/

# If not, download or copy it
# Option 1: From Git LFS
git lfs pull

# Option 2: From S3 (if you have it there)
# aws s3 cp s3://your-bucket/best_model.pt checkpoints/
```

---

## Step 5: Build and Start the Application

```bash
# Build and start containers
docker-compose up -d --build

# Check container status
docker-compose ps

# View logs
docker-compose logs -f
```

---

## Step 6: Configure AWS Security Group

In the AWS Console:
1. Go to EC2 → Security Groups
2. Select your instance's security group
3. Add Inbound Rule:
   - Type: Custom TCP
   - Port: 3000
   - Source: 0.0.0.0/0 (or your IP for restricted access)

---

## Step 7: Access the Application

Open in your browser:
```
http://your-ec2-public-ip:3000
```

---

## Management Commands

```bash
# Stop containers
docker-compose down

# Restart containers
docker-compose restart

# View backend logs
docker-compose logs -f backend

# View frontend logs
docker-compose logs -f frontend

# Rebuild after code changes
git pull
docker-compose up -d --build

# Clean up unused images
docker system prune -f
```

---

## Troubleshooting

### Check if containers are running:
```bash
docker-compose ps
```

### Check container logs:
```bash
docker-compose logs backend
docker-compose logs frontend
```

### Test backend directly:
```bash
curl http://localhost:8000/health
```

### Check if port is open:
```bash
sudo netstat -tlnp | grep 3000
```

### Memory issues (for ML model):
If the backend crashes due to memory, increase instance size or add swap:
```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

---

## Architecture

```
┌─────────────────────────────────────────────┐
│                  Internet                   │
│                     │                       │
│                Port 3000                    │
│                     ▼                       │
│  ┌─────────────────────────────────────┐   │
│  │        Frontend (Nginx)             │   │
│  │   - Serves React build              │   │
│  │   - Proxies /api/* to backend       │   │
│  └──────────────┬──────────────────────┘   │
│                 │                           │
│           Docker Network                    │
│                 │                           │
│  ┌──────────────▼──────────────────────┐   │
│  │        Backend (FastAPI)            │   │
│  │   - /api/predict                    │   │
│  │   - /api/explain                    │   │
│  │   - PyTorch model loaded            │   │
│  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```
