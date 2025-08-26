# Llama Vision Captioning - Server Edition

**Production-ready Vietnamese image captioning using llama.cpp with vision capabilities**

🚀 **Server-friendly setup**: No sudo required, conda environment management, minimal dependencies

## 🏃‍♂️ Quick Start for Servers

### For Server Admins (One-time Setup)

1. **Check system requirements** (see [SYSTEM_REQUIREMENTS.md](SYSTEM_REQUIREMENTS.md)):
   ```bash
   # Ensure these are available:
   conda --version    # Miniconda/Anaconda
   git --version      # Version control
   cmake --version    # Build system
   make --version     # Build tool
   gcc --version      # Compiler
   ```

2. **Review and customize environment**:
   ```bash
   # Edit environment.yml to match your server needs
   nano environment.yml  # Add/remove packages as needed
   ```

### For Users (Every Time)

3. **Setup environment** (server-safe, no sudo):
   ```bash
   ./setup.sh
   ```

4. **Build llama.cpp**:
   ```bash
   conda activate llama-vision
   ./build.sh  # Detects GPU automatically
   ```

5. **Run captioning**:
   ```bash
   python run.py --hf-token YOUR_HUGGINGFACE_TOKEN
   ```

## 📁 Server-Friendly Structure

```
llama-vision-captioning/
├── 📄 environment.yml       # Conda environment (REVIEWABLE)
├── 🔧 setup.sh             # No sudo, conda-only setup
├── 🔨 build.sh             # Separate llama.cpp build
├── 🎯 run.py               # Main processing script
├── 📋 SYSTEM_REQUIREMENTS.md # For server admins
├── 📖 README.md            # This file
└── workspace/              # Created during setup
    ├── llama.cpp/          # Built from source
    ├── models/             # Downloaded models
    └── dataset/            # Processing data
```

## 🎯 Key Server Features

### ✅ Server Admin Friendly
- **No sudo required** - Only conda environment management
- **Reviewable dependencies** - All packages listed in `environment.yml`
- **Minimal system requirements** - Usually already available on servers
- **No system modifications** - Self-contained in conda environment
- **GPU auto-detection** - Works with/without NVIDIA GPUs

### ✅ Production Ready
- **Checkpoint recovery** - Resume interrupted processing
- **Error handling** - Robust retry logic with exponential backoff
- **Parallel processing** - Configurable worker threads
- **Resource monitoring** - Memory and GPU usage optimization
- **Logging** - Comprehensive error tracking and progress

### ✅ Vietnamese AI Prompts
- **Structured JSON output** - Camera, objects, spatial, activity analysis
- **Accurate object counting** - People, vehicles, animals with descriptions
- **Spatial awareness** - Left/right/center/foreground/background
- **Natural captions** - Human-readable Vietnamese descriptions
- **News graphics filtering** - Excludes logos, tickers, overlays

## ⚙️ Environment Customization

Server admins can customize `environment.yml`:

### GPU vs CPU Setup
```yaml
# For GPU servers (recommended)
dependencies:
  - pytorch
  - pytorch-cuda=12.1

# For CPU-only servers (slower but works)  
dependencies:
  - pytorch-cpu
  - torchvision-cpu
  - torchaudio-cpu
```

### Package Filtering
```yaml
# Minimal setup (remove if not needed):
# - opencv         # Advanced image processing
# - jupyter        # Development tools  
# - ipython        # Interactive Python

# Pin versions for stability:
- numpy=1.24.3     # Instead of >=1.24.3
- pillow=10.0.1    # Exact versions
```

## 🔧 Advanced Usage

### Parallel Processing
```bash
python run.py --hf-token TOKEN --max-workers 8
```

### Error Recovery
```bash
python run.py --hf-token TOKEN --fix-errors
```

### Skip Setup Steps
```bash
python run.py --hf-token TOKEN --skip-model --skip-dataset
```

### Custom Configuration
```bash
python run.py --config config.yaml --hf-token TOKEN
```

## 📊 Performance Expectations

### Server Hardware Recommendations

| Setup | Processing Speed | VRAM Usage | RAM Usage |
|-------|------------------|------------|-----------|
| **High-end Server** (RTX 4090/A100) | 1-3 sec/image | 12GB | 8GB |
| **Mid-range Server** (RTX 3080/4080) | 3-5 sec/image | 8GB | 6GB |
| **CPU Only** (No GPU) | 15-30 sec/image | 0GB | 12GB |

### Scaling Guidelines
- **Small batch** (<1000 images): `--max-workers 4`
- **Medium batch** (1000-10000): `--max-workers 8`  
- **Large batch** (10000+): `--max-workers 12`

## 🇻🇳 Vietnamese Prompt Output

### Example Structured Output
```json
{
  "camera": {
    "angle": "từ trên cao",
    "shot_type": "toàn cảnh", 
    "movement": "tĩnh"
  },
  "setting": {
    "location": "sân bóng đá",
    "environment": "ngoài trời",
    "venue_type": "thể thao",
    "time_of_day": "ban ngày"
  },
  "objects": {
    "people": {
      "count": 22,
      "description": "Các cầu thủ mặc áo trắng và xanh dương"
    },
    "sports_equipment": {
      "count": 1, 
      "description": "Quả bóng đá màu trắng"
    }
  },
  "spatial": {
    "center": "Khu vực phạt đền",
    "left_side": "Khán đài phía tây", 
    "right_side": "Khán đài phía đông"
  },
  "caption": "Cảnh quay từ trên cao một sân bóng đá, hai đội mặc áo trắng và xanh dương, đang thực hiện quả phạt đền, có 4 cầu thủ Uzbekistan trong khung hình."
}
```

### Filtering Rules
- ✅ **Includes**: Objects, activities, spatial relationships, special camera angles
- ❌ **Excludes**: News logos, tickers, clocks, graphics overlays
- 🎯 **Focus**: Accurate counting, specific object names, natural Vietnamese

## 🚨 Troubleshooting for Servers

### Environment Issues
```bash
# Check environment
conda activate llama-vision
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Rebuild if needed
conda env remove -n llama-vision
./setup.sh
```

### Build Issues
```bash
# Clean rebuild
rm -rf workspace/llama.cpp/build
./build.sh
```

### Memory Issues
```bash
# Reduce workers for limited RAM
python run.py --hf-token TOKEN --max-workers 1

# Check GPU memory
nvidia-smi
```

### Permission Issues
```bash
# All operations should work without sudo
# If you get permission errors, check:
ls -la workspace/  # Should be owned by your user
conda info --envs # Environment should be in your home
```

## 📞 Server Admin Support

### System Requirements
- See [SYSTEM_REQUIREMENTS.md](SYSTEM_REQUIREMENTS.md) for detailed setup
- Most requirements usually pre-installed on development servers
- No root/sudo access needed for our scripts

### Package Review Process
1. **Review** `environment.yml` before setup
2. **Modify** packages as needed for your server
3. **Test** with `./setup.sh` in safe environment
4. **Deploy** to production servers

### Security Considerations
- Scripts only create conda environments (user-space)
- No system package installations
- No network services exposed by default
- All data processed locally

---

**🎯 Made for Production Servers**: Minimal dependencies, no sudo required, fully reviewable package list, checkpoint recovery, comprehensive error handling.

**🇻🇳 AI-Powered**: Advanced Vietnamese language prompts with structured metadata extraction and natural language captions.
