#!/bin/bash

# Production deployment script for Car Classification API

set -e

echo "🚀 Deploying Car Classification API..."

# Check if model exists
if [ ! -f "checkpoints/best_model.pth" ]; then
    echo "❌ Error: Model checkpoint not found!"
    echo "Please ensure you have trained the model and best_model.pth exists in checkpoints/"
    exit 1
fi

# Check if label mapping exists, create if not
if [ ! -f "checkpoints/label_mapping.json" ]; then
    echo "⚠️  Creating default label mapping..."
    cat > checkpoints/label_mapping.json << EOF
{
    "make": {"toyota": 0},
    "model": {"rav4": 0},
    "steer_wheel": {"left": 0}
}
EOF
fi

# Install additional production dependencies
echo "📦 Installing production dependencies..."
pip install flask gunicorn

# Create production directory structure
mkdir -p logs
mkdir -p static
mkdir -p templates

# Set permissions
chmod +x production_app.py
chmod +x api_client.py

echo "✅ Deployment preparation complete!"

# Option 1: Run with Flask (development)
if [ "$1" = "dev" ]; then
    echo "🔧 Starting in development mode..."
    python production_app.py

# Option 2: Run with Gunicorn (production)
elif [ "$1" = "prod" ]; then
    echo "🏭 Starting in production mode with Gunicorn..."
    gunicorn --bind 0.0.0.0:5000 --workers 4 --timeout 120 production_app:app

# Option 3: Run with Docker
elif [ "$1" = "docker" ]; then
    echo "🐳 Building and running with Docker..."
    cd docker
    docker-compose up --build -d
    echo "✅ API running at http://localhost:5000"
    echo "📊 Check logs with: docker-compose logs -f"

# Default: Show options
else
    echo "🎯 Usage: ./deploy.sh [dev|prod|docker]"
    echo ""
    echo "Options:"
    echo "  dev    - Run with Flask (development)"
    echo "  prod   - Run with Gunicorn (production)"
    echo "  docker - Run with Docker (containerized)"
    echo ""
    echo "Examples:"
    echo "  ./deploy.sh dev     # Development server"
    echo "  ./deploy.sh prod    # Production server"
    echo "  ./deploy.sh docker  # Docker container"
fi
