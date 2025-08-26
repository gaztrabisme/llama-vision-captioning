# Llama Vision Captioning

**One-command solution for automated image captioning using llama.cpp with vision capabilities**

This repository provides a complete, automated setup for running vision model captioning with Vietnamese prompts. Just input your Hugging Face token and everything else is handled automatically.

## 🚀 Quick Start

### 1. Setup Environment (One Time)

```bash
# Clone or download this repository
cd llama-vision-captioning

# Make setup script executable and run it
chmod +x setup.sh
./setup.sh
```

This will:
- Create a conda environment `llama-vision`
- Install all dependencies
- Build llama.cpp with CUDA support (if available)
- Install Python requirements

### 2. Run Captioning (Every Time)

```bash
# Activate the environment
conda activate llama-vision

# Run the complete captioning pipeline
python run.py --hf-token YOUR_HUGGINGFACE_TOKEN
```

That's it! The script will:
1. Download and convert the Qwen2.5-VL model to GGUF format
2. Download the dataset from Google Drive
3. Start llama-server with vision support
4. Process all images with Vietnamese prompts
5. Save results with checkpoint recovery

## 📋 Requirements

### System Requirements
- **GPU**: NVIDIA GPU with CUDA support (recommended) or CPU
- **RAM**: 16GB+ recommended (8GB minimum)
- **Storage**: 20GB+ free space for models and dataset
- **OS**: Linux, macOS, or Windows with WSL

### Software Requirements
- **Conda/Miniconda**: For Python environment management
- **Git**: For downloading repositories
- **Build tools**: GCC/Clang, CMake, Make
- **CUDA toolkit**: For GPU acceleration (optional but recommended)

## 🔧 Advanced Usage

### Process with More Workers
```bash
python run.py --hf-token YOUR_TOKEN --max-workers 8
```

### Fix Only Error Files
```bash
python run.py --hf-token YOUR_TOKEN --fix-errors
```

### Skip Setup Steps (if already done)
```bash
python run.py --hf-token YOUR_TOKEN --skip-model --skip-dataset
```

## 📁 Directory Structure

After running, your directory will look like:
```
llama-vision-captioning/
├── setup.sh                 # Environment setup script
├── run.py                   # Main processing script
├── requirements.txt         # Python dependencies
├── workspace/
│   ├── llama.cpp/           # llama.cpp source and binaries
│   ├── models/
│   │   ├── qwen2.5-vl-aio/  # Original HF model
│   │   └── gguf/            # Converted GGUF models
│   └── dataset/
│       └── ground_truth/    # Images and captions
│           ├── captions/    # Generated caption files
│           └── checkpoint.pkl # Progress checkpoint
├── captioning.log           # Processing logs
└── README.md
```

## 🎯 Features

### ✅ Complete Automation
- One command setup and execution
- Automatic model download and conversion
- Automatic dataset download from Google Drive
- Automatic server startup and management

### ✅ Robust Processing
- Checkpoint recovery for resuming interrupted work
- Error detection and retry logic
- Parallel processing with configurable workers
- Vietnamese prompt with structured JSON output

### ✅ Production Ready
- Comprehensive error handling
- Progress tracking with detailed logging
- Memory-efficient processing
- Compatible with existing checkpoint files

## 🇻🇳 Vietnamese Prompt Details

The system uses a comprehensive Vietnamese prompt that generates structured JSON with:

- **Camera details**: Angle, shot type, movement
- **Setting information**: Location, environment, venue type, time of day
- **Object detection**: People, vehicles, animals, and other objects with counts
- **Spatial analysis**: Left/right/center/top/bottom/foreground/background
- **Activity recognition**: Primary and secondary actions
- **Text elements**: Signs, displays, overlays (news graphics excluded from caption)
- **Natural caption**: Human-readable Vietnamese description

### Example Output
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
    }
  },
  "caption": "Cảnh quay từ trên cao một sân bóng đá, hai đội mặc áo trắng và xanh dương, đang thực hiện quả phạt đền, có 4 cầu thủ Uzbekistan trong khung hình."
}
```

## 🔧 Troubleshooting

### Setup Issues

**Error: Conda not found**
```bash
# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

**Error: CUDA not found**
- Install NVIDIA CUDA Toolkit from https://developer.nvidia.com/cuda-downloads
- Or run without GPU acceleration (will be slower)

**Error: Build failed**
```bash
# Install build tools (Ubuntu/Debian)
sudo apt install build-essential cmake git

# Or macOS
xcode-select --install
brew install cmake
```

### Runtime Issues

**Error: Server failed to start**
- Check `captioning.log` for detailed error messages
- Verify model files exist in `workspace/models/gguf/`
- Try reducing parallel workers: `--max-workers 1`

**Error: Out of memory**
- Reduce batch size by using fewer workers: `--max-workers 2`
- Use CPU instead of GPU for inference
- Close other applications to free memory

**Error: Model download failed**
- Verify your Hugging Face token has access to the private model
- Check internet connection
- Try running again (downloads resume automatically)

### Getting Help

1. Check the log file: `captioning.log`
2. Verify system requirements are met
3. Try running with `--max-workers 1` to isolate issues
4. Ensure you have a valid Hugging Face token with access to `GazTrab/Qwen2.5-VL-AIO`

## 📊 Performance

### Expected Processing Speed
- **With GPU**: 2-5 seconds per image
- **CPU only**: 10-30 seconds per image
- **Parallel workers**: Scales with available resources

### Resource Usage
- **VRAM**: 8-12GB for Q8 model
- **RAM**: 4-8GB for processing
- **Storage**: ~15GB for model + dataset

## 🏗️ Technical Details

### Model Information
- **Base Model**: Qwen2.5-VL-AIO (Private HuggingFace model)
- **Format**: GGUF Q8_0 quantization
- **Vision Support**: Multi-modal projector (mmproj) for image processing
- **Context**: 4096 tokens maximum

### Server Configuration
- **Backend**: llama.cpp server with OpenAI-compatible API
- **Concurrency**: 10 parallel slots
- **GPU**: Automatic detection and utilization
- **Batching**: Continuous batching enabled for better throughput

### Dataset
- **Source**: Google Drive (automatically downloaded)
- **Structure**: `ground_truth/` with images and `captions/` output
- **Checkpoint**: Automatic progress saving in `checkpoint.pkl`
- **Formats**: PNG, JPG, JPEG, GIF, BMP, WebP supported

## 📄 License

This project is provided as-is for research and educational purposes. Please ensure you comply with:
- llama.cpp license terms
- Qwen model license terms  
- Hugging Face terms of service
- Any applicable dataset licenses

---

**Made with ❤️ by AI Assistant**

*For issues or improvements, please check the troubleshooting section or review the logs.*
