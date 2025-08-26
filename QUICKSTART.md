# 🚀 SERVER QUICK START GUIDE

## For Server Admins (Review First) 

**📋 Check what's needed** (usually already installed):
```bash
conda --version    # ✅ Miniconda/Anaconda  
git --version      # ✅ Version control
cmake --version    # ✅ Build system
make --version     # ✅ Build automation
gcc --version      # ✅ Compiler
```

**📄 Review packages** (customize as needed):
```bash
nano environment.yml  # Edit packages before setup
```

## For Users (3 Commands) 

```bash
# 1. Setup (no sudo, server-safe)
./setup.sh

# 2. Build llama.cpp  
conda activate llama-vision
./build.sh

# 3. Run captioning
python run.py --hf-token YOUR_HUGGINGFACE_TOKEN
```

## 🎯 What Changed for Servers?

### ✅ Server-Friendly Now
- **No sudo required** - Only conda environment
- **Reviewable packages** - See everything in `environment.yml`  
- **Minimal dependencies** - Just conda + basic build tools
- **No system changes** - Self-contained setup

### ✅ What You Get
- **Vietnamese AI captions** with structured JSON
- **Checkpoint recovery** - Resume interrupted work  
- **Parallel processing** - Fast batch processing
- **GPU auto-detection** - Uses CUDA if available

### ✅ Perfect for Production
- **Clean environments** - No conflicts with system packages
- **Resource control** - Configurable workers and memory
- **Error handling** - Robust retry logic
- **Progress tracking** - Detailed logging

## 🔧 Server Customization

**Edit `environment.yml` for your needs:**

```yaml
# Remove GPU support (CPU only)
- pytorch-cpu
- torchvision-cpu  
- torchaudio-cpu

# Remove optional packages
# - opencv     # Advanced image processing
# - jupyter    # Development tools

# Pin versions for stability  
- numpy=1.24.3   # Exact version
- pillow=10.0.1  # No surprises
```

## 💡 Pro Server Tips

**Performance tuning:**
```bash
# More workers for powerful servers
python run.py --hf-token TOKEN --max-workers 12

# Conservative for shared servers  
python run.py --hf-token TOKEN --max-workers 2
```

**Error recovery:**
```bash
# Fix only failed images
python run.py --hf-token TOKEN --fix-errors
```

**Skip downloads (if models exist):**
```bash
python run.py --hf-token TOKEN --skip-model --skip-dataset
```

---

**🎉 Ready for Production**: Your server admin can now safely deploy this with full control over packages and no system modifications!
