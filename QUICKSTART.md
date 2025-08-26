# 🚀 QUICK START GUIDE

## What You Get
A complete, automated solution for Vietnamese image captioning using llama.cpp with vision capabilities.

## 📁 Repository Contents
```
llama-vision-captioning/
├── README.md          # Comprehensive documentation
├── setup.sh          # One-time environment setup
├── run.py            # Main processing script
├── check_system.py   # System requirements checker
├── requirements.txt  # Python dependencies
├── config.yaml       # Advanced configuration
└── make_executable.sh # Make scripts executable
```

## ⚡ Super Quick Start (3 Commands)

```bash
# 1. Setup everything (one time only)
./setup.sh

# 2. Activate environment & run
conda activate llama-vision
python run.py --hf-token YOUR_HUGGINGFACE_TOKEN

# That's it! ✨
```

## 🔍 Before You Start

Check if your system is ready:
```bash
python check_system.py
```

## 🎯 What Happens When You Run

1. **Downloads model** - `GazTrab/Qwen2.5-VL-AIO` from HuggingFace
2. **Converts to GGUF** - Q8 quantization + vision projector
3. **Downloads dataset** - From Google Drive automatically
4. **Starts server** - llama.cpp with 10 concurrent slots
5. **Processes images** - With Vietnamese structured prompts
6. **Saves results** - JSON format with checkpoint recovery

## 📊 Expected Output

Each image produces structured JSON like:
```json
{
  "camera": {"angle": "từ trên cao", "shot_type": "toàn cảnh"},
  "objects": {"people": {"count": 5, "description": "..."}},
  "caption": "Cảnh quay từ trên cao một sân bóng đá..."
}
```

## 💡 Pro Tips

- **First run**: Takes longer (downloads ~15GB)
- **Resumable**: Interrupted? Just run again
- **Parallel**: Use `--max-workers 8` for speed
- **Fix errors**: Use `--fix-errors` flag
- **GPU recommended**: 8GB+ VRAM ideal

## 🆘 Need Help?

1. Read the full `README.md`
2. Check `captioning.log` for errors
3. Run with `--max-workers 1` to debug
4. Ensure HF token has model access

---
**🎉 Ready to caption thousands of images with AI? Let's go!**
