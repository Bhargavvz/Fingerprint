#!/bin/bash
# ============================================================
# RUN FULL APPLICATION (Backend + Frontend)
# ============================================================

set -e

echo "============================================================"
echo "🚀 STARTING FULL APPLICATION"
echo "============================================================"

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Check if model exists
if [ ! -f "checkpoints/best_model.pt" ]; then
    echo "⚠️  Warning: No trained model found at checkpoints/best_model.pt"
    echo "   The API will start but predictions won't work until a model is trained."
fi

# Kill any existing processes on the ports
echo "🔧 Cleaning up existing processes..."
pkill -f "uvicorn" 2>/dev/null || true
pkill -f "npm run dev" 2>/dev/null || true

# Start backend in background
echo "🚀 Starting FastAPI backend on port 8000..."
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
cd ..

# Wait for backend to start
sleep 3

# Start frontend in background
echo "🚀 Starting React frontend on port 3000..."
cd frontend
npm run dev -- --host 0.0.0.0 &
FRONTEND_PID=$!
cd ..

echo ""
echo "============================================================"
echo "✅ APPLICATION STARTED!"
echo "============================================================"
echo ""
echo "🌐 Frontend: http://localhost:3000"
echo "🔗 Backend API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for interrupt
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT
wait
