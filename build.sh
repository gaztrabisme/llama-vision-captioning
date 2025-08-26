# Server-Friendly Build Script for llama.cpp
# Use this after running setup.sh to build llama.cpp with proper settings

#!/bin/bash

set -e

# Colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }

echo "🔨 Building llama.cpp for Server"
echo "================================"

# Check if in correct environment
if [[ "$CONDA_DEFAULT_ENV" != "llama-vision" ]]; then
    log_warning "⚠️  Please activate the environment first:"
    echo "   conda activate llama-vision"
    exit 1
fi

# Check if workspace exists
if [[ ! -d "workspace/llama.cpp" ]]; then
    log_warning "⚠️  llama.cpp not found. Run setup.sh first."
    exit 1
fi

cd workspace/llama.cpp

# Clean previous build
if [[ -d "build" ]]; then
    log_info "🧹 Cleaning previous build..."
    rm -rf build
fi

mkdir -p build
cd build

# Detect GPU support
CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release"

if command -v nvidia-smi >/dev/null 2>&1 && command -v nvcc >/dev/null 2>&1; then
    log_info "🎮 Building with CUDA support..."
    CMAKE_ARGS="$CMAKE_ARGS -DGGML_CUDA=ON"
else
    log_info "🖥️  Building with CPU support only..."
fi

# Configure
log_info "⚙️  Configuring build..."
cmake .. $CMAKE_ARGS

# Build with all available cores
CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
log_info "🔨 Building with $CORES parallel jobs..."
make -j$CORES

# Verify build
if [[ -f "bin/llama-server" ]] && [[ -f "bin/llama-cli" ]]; then
    log_success "✅ llama.cpp built successfully!"
    ls -la bin/llama-*
    
    # Test basic functionality
    log_info "🧪 Testing basic functionality..."
    ./bin/llama-cli --help >/dev/null 2>&1 && log_success "✅ llama-cli works"
    ./bin/llama-server --help >/dev/null 2>&1 && log_success "✅ llama-server works"
    
    # GPU test if available
    if command -v nvidia-smi >/dev/null 2>&1; then
        log_info "🎮 Testing GPU detection..."
        ./bin/llama-cli --list-devices 2>/dev/null || log_warning "⚠️  GPU detection test failed"
    fi
else
    echo "❌ Build failed! Check the output above for errors."
    exit 1
fi

cd ../../..

log_success "🎉 Build completed! You can now run:"
echo "   python run.py --hf-token YOUR_HUGGINGFACE_TOKEN"
