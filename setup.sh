#!/bin/bash

# =============================================================================
# LLAMA VISION CAPTIONING - SERVER-FRIENDLY SETUP
# =============================================================================
# This script sets up ONLY the conda environment - no system packages
# No sudo required - perfect for servers with limited permissions
# 
# PREREQUISITES (ask your server admin to ensure these are available):
# - conda/miniconda installed
# - git command available
# - Basic build tools (gcc, make) - usually pre-installed on servers
# - CUDA toolkit (optional, for GPU acceleration)
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
WORKSPACE_DIR="$PWD/workspace"

echo "🚀 Llama Vision Captioning - Server Setup"
echo "========================================="
log_info "This setup requires NO sudo access - perfect for servers!"
echo ""

# Check prerequisites
log_info "📋 Checking prerequisites..."

# Check conda
if ! command -v conda >/dev/null 2>&1; then
    log_error "❌ Conda not found. Please ask server admin to install Miniconda"
    log_info "   Download: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi
log_success "✅ Conda found: $(conda --version)"

# Check git
if ! command -v git >/dev/null 2>&1; then
    log_error "❌ Git not found. Please ask server admin to install git"
    exit 1
fi
log_success "✅ Git found: $(git --version | head -1)"

# Check if environment.yml exists
if [[ ! -f "environment.yml" ]]; then
    log_error "❌ environment.yml not found in current directory"
    log_info "   Make sure you're in the llama-vision-captioning directory"
    exit 1
fi

# Show environment.yml contents for review
echo ""
log_info "📄 Environment configuration (please review):"
echo "=============================================="
cat environment.yml
echo "=============================================="
echo ""

# Ask for confirmation
read -p "$(echo -e ${YELLOW}Review the environment.yml above. Modify it if needed, then continue? [y/N]: ${NC})" -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    log_info "👍 Setup cancelled. Modify environment.yml as needed and run again."
    exit 0
fi

# Remove existing environment if it exists
if conda info --envs | grep -q "^$ENV_NAME "; then
    log_warning "⚠️  Environment '$ENV_NAME' exists. Removing it first..."
    conda remove -n $ENV_NAME --all -y
fi

# Create conda environment from environment.yml
log_info "🔨 Creating conda environment from environment.yml..."
conda env create -f environment.yml

# Activate environment
log_info "🔌 Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# Verify activation
if [[ "$CONDA_DEFAULT_ENV" == "$ENV_NAME" ]]; then
    log_success "✅ Environment '$ENV_NAME' activated successfully"
else
    log_error "❌ Failed to activate environment"
    exit 1
fi

# Show Python and key package versions
log_info "📦 Checking installed packages..."
python --version
echo "Key packages:"
python -c "import torch; print(f'  PyTorch: {torch.__version__}')" 2>/dev/null || echo "  PyTorch: Not installed"
python -c "import transformers; print(f'  Transformers: {transformers.__version__}')" 2>/dev/null || echo "  Transformers: Not installed"
python -c "import numpy; print(f'  NumPy: {numpy.__version__}')" 2>/dev/null || echo "  NumPy: Not installed"
python -c "import PIL; print(f'  Pillow: {PIL.__version__}')" 2>/dev/null || echo "  Pillow: Not installed"

# GPU check (optional)
echo ""
log_info "🎮 GPU Detection:"
if command -v nvidia-smi >/dev/null 2>&1; then
    log_success "✅ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  GPU info not available"
    
    # Test PyTorch CUDA
    python -c "import torch; print(f'  PyTorch CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || echo "  PyTorch CUDA test failed"
else
    log_warning "⚠️  No nvidia-smi found. Will use CPU (slower but works)"
fi

# Create workspace directory
log_info "📁 Creating workspace directory..."
mkdir -p $WORKSPACE_DIR

# Clone llama.cpp if not exists
if [[ ! -d "$WORKSPACE_DIR/llama.cpp" ]]; then
    log_info "📥 Cloning llama.cpp repository..."
    cd $WORKSPACE_DIR
    git clone https://github.com/ggml-org/llama.cpp.git
    cd llama.cpp
    
    # Install Python requirements for conversion scripts
    log_info "📦 Installing llama.cpp conversion requirements..."
    pip install -r requirements/requirements-convert-hf-to-gguf.txt
    
    cd ../..
else
    log_info "✅ llama.cpp already exists, updating..."
    cd $WORKSPACE_DIR/llama.cpp
    git pull
    cd ../..
fi

echo ""
log_success "🎉 Environment setup completed successfully!"
echo ""
log_info "📋 Next Steps:"
log_info "1. Build llama.cpp:"
log_info "   cd workspace/llama.cpp"
log_info "   mkdir -p build && cd build"
log_info "   cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release  # For GPU"
log_info "   # OR cmake .. -DCMAKE_BUILD_TYPE=Release             # For CPU only"
log_info "   make -j$(nproc 2>/dev/null || echo 4)"
log_info ""
log_info "2. Run the captioning:"
log_info "   conda activate $ENV_NAME"
log_info "   python run.py --hf-token YOUR_HUGGINGFACE_TOKEN"
echo ""
log_info "💡 Pro tip: The admin can modify environment.yml to add/remove packages as needed"
