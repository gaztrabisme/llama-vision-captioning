# System Requirements for Server Admins

This document lists the **system-level requirements** that need to be available on the server before running the setup. The `setup.sh` script will **NOT** install these (no sudo required).

## 🔧 Essential System Requirements

### Required (Must Have)
```bash
# Package managers / Version control
conda (or miniconda)    # Python environment management
git                     # Version control

# Build essentials (usually pre-installed on servers)  
gcc                     # C compiler
g++                     # C++ compiler
make                    # Build automation
cmake                   # Build system generator
```

### Optional (For Better Performance)
```bash
# NVIDIA GPU Support (for faster processing)
nvidia-driver          # NVIDIA graphics driver
nvidia-cuda-toolkit     # CUDA development kit
nvidia-smi             # GPU monitoring tool

# Alternative: CPU-only setup works but is slower
```

## 📦 Installation Examples

### Ubuntu/Debian Servers
```bash
# Essential packages
sudo apt update
sudo apt install -y git cmake make build-essential

# CUDA support (optional)
sudo apt install -y nvidia-driver-470 nvidia-cuda-toolkit

# Miniconda (if not installed)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

### CentOS/RHEL Servers  
```bash
# Essential packages
sudo yum groupinstall -y "Development Tools"
sudo yum install -y git cmake make gcc gcc-c++

# CUDA support (optional)
sudo yum install -y nvidia-driver cuda-toolkit

# Miniconda (if not installed)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

### Already Installed on Most Servers
Many servers already have these packages. Check with:
```bash
# Check if already available
which git cmake make gcc g++ conda
conda --version
git --version
cmake --version
make --version
gcc --version

# GPU check (optional)
nvidia-smi
nvcc --version
```

## 🎯 Minimal Setup for Servers

If you want the **absolute minimum** for servers:

### Must Have
- `conda` or `miniconda` 
- `git`
- `cmake`, `make`, `gcc` (for building llama.cpp)

### Can Skip
- NVIDIA drivers (will use CPU, slower but works)
- Advanced build tools
- System Python packages (conda handles everything)

## 🚫 What Our Setup.sh Does NOT Do

✅ **Safe for servers** - Our setup script:
- Only creates conda environments
- Only installs Python packages
- No system-level modifications  
- No sudo commands
- No package manager calls (apt/yum/etc)

❌ **Server admins don't need to worry about:**
- System package installations
- Sudo permission requirements  
- Conflicting with existing software
- Breaking system Python

## 🔧 Environment Customization

The `environment.yml` file can be modified by server admins to:

```yaml
# Remove GPU support (CPU only)
dependencies:
  - pytorch-cpu
  - torchvision-cpu  
  - torchaudio-cpu
  # Remove: pytorch-cuda=12.1

# Remove optional packages
# - opencv          # Advanced image processing
# - jupyter         # Development tools
# - ipython         # Interactive Python

# Change Python version
- python=3.9        # Instead of 3.10

# Pin specific versions for stability
- numpy=1.24.3      # Instead of >=1.24.0
- pillow=10.0.1     # Instead of >=10.0.0
```

## 🎮 GPU vs CPU Performance

### With GPU (Recommended)
- **Speed**: 2-5 seconds per image
- **Requirements**: NVIDIA GPU + 8GB+ VRAM
- **Setup**: Install CUDA toolkit

### CPU Only (Fallback)
- **Speed**: 10-30 seconds per image  
- **Requirements**: Just CPU + 8GB+ RAM
- **Setup**: Modify environment.yml to remove CUDA packages

## ✅ Verification Commands

After system setup, verify with:
```bash
# Check all requirements
conda --version
git --version  
cmake --version
make --version
gcc --version

# Optional GPU check
nvidia-smi
nvcc --version

# Then run our setup
./setup.sh
```

---

**💡 Note for Server Admins**: The `environment.yml` file contains all Python dependencies with specific versions. Review and modify it as needed for your server environment before running `./setup.sh`.
