#!/bin/bash

# =============================================================================
# LLAMA VISION CAPTIONING - AUTOMATED SETUP
# =============================================================================
# This script sets up everything needed for vision model captioning:
# - Creates conda environment
# - Installs CUDA toolkit if needed
# - Builds llama.cpp with CUDA support
# - Installs Python dependencies
# =============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Configuration
ENV_NAME="llama-vision"
PYTHON_VERSION="3.10"
WORKSPACE_DIR="$PWD/workspace"

log_info "🚀 Starting automated setup for Llama Vision Captioning..."

# Check if conda is installed
if ! command -v conda >/dev/null 2>&1; then
    log_error "Conda is not installed. Please install Miniconda/Anaconda first."
    log_info "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create conda environment
log_info "🐍 Creating conda environment: $ENV_NAME"
if conda info --envs | grep -q "^$ENV_NAME "; then
    log_warning "Environment $ENV_NAME already exists. Removing it first..."
    conda remove -n $ENV_NAME --all -y
fi

conda create -n $ENV_NAME python=$PYTHON_VERSION -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

log_success "Conda environment '$ENV_NAME' created and activated"

# Install system dependencies based on OS
log_info "📦 Installing system dependencies..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux - check if we have sudo
    if command -v sudo >/dev/null 2>&1; then
        sudo apt update || { log_warning "apt update failed, continuing..."; }
        sudo apt install -y build-essential cmake git wget curl || {
            log_warning "Some system packages failed to install, continuing..."
        }
    else
        log_warning "No sudo access. Please ensure build-essential, cmake, git are installed"
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    if ! command -v brew >/dev/null 2>&1; then
        log_warning "Homebrew not found. Please install it for optimal experience."
        log_info "Or ensure Xcode command line tools are installed: xcode-select --install"
    else
        brew install cmake git wget curl || {
            log_warning "Some brew packages failed to install, continuing..."
        }
    fi
fi

# Install Python packages
log_info "📦 Installing Python dependencies..."
pip install --upgrade pip

# Install PyTorch with CUDA support (if available)
if command -v nvidia-smi >/dev/null 2>&1; then
    log_info "🎮 NVIDIA GPU detected, installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    log_info "🖥️  Installing PyTorch (CPU version)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install other requirements
pip install -r requirements.txt

# Create workspace directory
mkdir -p $WORKSPACE_DIR
cd $WORKSPACE_DIR

# Clone llama.cpp if not exists
if [[ ! -d "llama.cpp" ]]; then
    log_info "📥 Cloning llama.cpp repository..."
    git clone https://github.com/ggml-org/llama.cpp.git
fi

cd llama.cpp

# Install Python requirements for conversion
log_info "🔧 Installing llama.cpp Python requirements..."
pip install -r requirements/requirements-convert-hf-to-gguf.txt

# Build llama.cpp
log_info "🔨 Building llama.cpp..."
mkdir -p build
cd build

# Configure build based on available hardware
CMAKE_ARGS=""
if command -v nvidia-smi >/dev/null 2>&1 && command -v nvcc >/dev/null 2>&1; then
    log_info "🎮 Building with CUDA support..."
    CMAKE_ARGS="-DGGML_CUDA=ON"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    log_info "🍎 Building with Metal support (macOS)..."
    CMAKE_ARGS="-DGGML_METAL=ON"
else
    log_info "🖥️  Building with CPU support..."
    CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release"
fi

cmake .. $CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Verify build
if [[ -f "bin/llama-server" ]] && [[ -f "bin/llama-cli" ]]; then
    log_success "✅ llama.cpp built successfully!"
    
    # Test GPU detection if available
    if command -v nvidia-smi >/dev/null 2>&1; then
        log_info "🧪 Testing GPU detection..."
        ./bin/llama-cli --list-devices 2>/dev/null || log_warning "GPU detection test failed"
    fi
else
    log_error "❌ llama.cpp build failed!"
    exit 1
fi

cd ../../..  # Back to main directory

log_success "🎉 Setup completed successfully!"
log_info ""
log_info "Next steps:"
log_info "1. Activate the environment: conda activate $ENV_NAME"
log_info "2. Run the captioning script: python run.py --hf-token YOUR_TOKEN"
log_info ""
log_info "Or run everything in one go:"
log_info "conda activate $ENV_NAME && python run.py --hf-token YOUR_TOKEN"
