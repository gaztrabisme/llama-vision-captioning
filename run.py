#!/usr/bin/env python3
"""
Llama Vision Captioning - One Command Solution
==============================================

This script handles everything automatically:
1. Downloads and converts the vision model to GGUF
2. Downloads the dataset from Google Drive
3. Starts llama-server with vision support
4. Runs image captioning with Vietnamese prompts
5. Handles checkpoint recovery for resuming work

Usage:
    python run.py --hf-token YOUR_HUGGINGFACE_TOKEN
    python run.py --hf-token YOUR_TOKEN --max-workers 8 --fix-errors

Author: AI Assistant
"""

import os
import sys
import json
import pickle
import time
import base64
import threading
import logging
import argparse
import signal
import random
import subprocess
import shutil
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import OpenAI client for llama.cpp server communication
from openai import OpenAI
import gdown

# =============================================================================
# CONFIGURATION
# =============================================================================

# Vietnamese prompt from user specification
VIETNAMESE_PROMPT = """**Nhiệm vụ**: Phân tích hình ảnh và cung cấp metadata có cấu trúc cùng mô tả tự nhiên chi tiết bằng tiếng Việt.

### Hướng dẫn chính:
1. **Chỉ bao gồm đối tượng có thật** - không tạo trường cho đối tượng không tồn tại
2. **Đặt tên đối tượng cụ thể** - dùng tên có thể tìm kiếm (boats, bicycles, flowers thay vì decorations)
3. **Đếm chính xác** - số nguyên dương (1,2,3...), ước lượng nếu quá nhiều
4. **Dùng "None"** cho trường không xác định được
5. **Loại trừ đồ họa tin tức** khỏi caption (logo, ticker, đồng hồ)
6. **Caption chỉ góc đặc biệt** ("từ trên cao", "cận cảnh") - bỏ qua góc thông thường

### Format JSON:
{
  "camera": {
    "angle": "", // "từ trên cao", "mặt đất", "góc thấp", "ngang mắt", vv
    "shot_type": "", // "cận cảnh", "trung bình", "toàn cảnh", vv
    "movement": "" // "tĩnh", "quay ngang", "zoom", "theo dõi", vv
  },
  "setting": {
    "location": "", // "sân bóng đá", "lớp học", "đường phố", vv
    "environment": "", // "trong nhà", "ngoài trời", "bán kín"
    "venue_type": "", // "thể thao", "giáo dục", "thương mại", vv
    "time_of_day": "" // "ban ngày", "buổi tối", "hoàng hôn", vv
  },
  "objects": {
    // CHỈ bao gồm đối tượng thực sự tồn tại trong cảnh
    // Sử dụng tên cụ thể, có thể tìm kiếm: people, boats, bicycles, flowers, books, signs, buildings, etc.
    "people": {
      "count": 0, // Số lượng người (số nguyên dương)
      "description": "" // Mô tả chi tiết: giới tính, độ tuổi, màu quần áo, phụ kiện, vv
    },
    "vehicles": {
      "count": 0, // Số lượng phương tiện (số nguyên dương)
      "description": "" // Mô tả: loại xe, màu sắc, kích thước, trạng thái, vv
    },
    "animals": {
      "count": 0, // Số lượng động vật (số nguyên dương)
      "description": "" // Mô tả: loại động vật, màu sắc, kích thước, hành động, vv
    }
    // Ba nhóm trên: people, vehicles, animals chỉ là ví dụ mẫu, không bắt buộc phải có
    // Tự do thêm nhóm khác với tên cụ thể: boats, bicycles, flowers, signs, buildings, furniture, sports_equipment, food_items, etc.
    // Mỗi nhóm cần có count (số nguyên dương) và description
  },
  "spatial": {
    "left_side": "",
    "right_side": "",
    "center": "",
    "top": "",
    "bottom": "",
    "foreground": "",
    "background": ""
  },
  "activity": {
    "primary_action": "",
    "secondary_actions": [],
    "movement_patterns": ""
  },
  "text_elements": {
    "time_display": "",
    "channel_logo": "", // CHỈ phân tích, KHÔNG vào caption
    "news_ticker": "", // CHỈ phân tích, KHÔNG vào caption
    "graphics_overlay": "", // CHỈ phân tích, KHÔNG vào caption
    "scene_text": {
      // Tự do tạo nhóm: street_signs, billboards, shop_names, etc.
    }
  },
  "caption": "" // Tổng hợp metadata thành mô tả tự nhiên, LOẠI TRỪ đồ họa tin tức
}

### Caption phải:
- **Loại trừ**: Logo kênh, ticker, đồng hồ, đồ họa overlay, góc thông thường
- **Bao gồm**: Góc đặc biệt, bối cảnh, đối tượng + số lượng + chi tiết, hành động, vị trí

### Ví dụ mẫu cho trường "caption":
**Ví dụ 1**: "Cảnh quay từ trên cao một sân bóng đá, hai đội mặc áo trắng và xanh dương, đang thực hiện quả phạt đền, có 4 cầu thủ Uzbekistan trong khung hình."
**Ví dụ 2**: "Cảnh quay cận cảnh một người đàn ông tóc trắng đang cầm cửa kính viền đen, trước mặt có lá cờ nhiều màu đỏ vàng xanh, xung quanh có nhiều micro được đưa lên trong đó có một chiếc micro màu xanh lá."
**Ví dụ 3**: "Cảnh quay một trạm xăng với một người đang đổ xăng cho khách hàng, sau đó có một người bỏ chạy khi một chiếc xe khác lao thẳng vào vị trí đang đổ xăng."
**Ví dụ 4**: "Cảnh trong một lớp học, bảng trang trí trên tường có thể nhìn rõ 5 bông hoa to theo thứ tự màu xanh dương, cam, vàng, xanh lá, đỏ, bên dưới những bông hoa này ghi lớp 1A với một số."
**Ví dụ 5**: "Cảnh một người đang trèo lên cây hái quả sầu riêng, có 1 người mặc áo xanh và 1 người mặc áo đen đứng dưới hỗ trợ."
"""

# Configuration
WORKSPACE_DIR = "workspace"
MODEL_NAME = "GazTrab/Qwen2.5-VL-AIO"
DATASET_GDRIVE_ID = "1PZZ98Fp6OLBjZpnsr-gvxUKMm8m7Pkq7"
SERVER_PORT = 8080
MAX_TOKENS = 4096

# Error indicators
ERROR_MESSAGES = [
    "An internal error has occurred",
    "Error in process_and_save for",
    "500 An internal error has occurred",
    "500 INTERNAL",
    "503 Service Unavailable",
    "Server is overloaded",
    "Error processing",
    "error",
    "Error code: 429",
    "Error code: 500",
    "Error code: 503",
    "Rate limit exceeded",
    "Too many requests",
    "Max retries",
    "Connection error",
    "Timeout",
    "Failed to connect",
    "RESOURCE_EXHAUSTED",
    "Model not loaded",
    "Server not ready"
]

# Global state
shutdown_requested = False
server_process = None

# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('captioning.log'),
        logging.StreamHandler()
    ]
)

def signal_handler(signum, frame):
    """Handle CTRL+C and other termination signals gracefully"""
    global shutdown_requested, server_process
    shutdown_requested = True
    logging.info("\n🛑 Shutdown requested. Cleaning up...")
    
    if server_process:
        logging.info("🔌 Stopping server...")
        server_process.terminate()
        try:
            server_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_process.kill()
    
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def log_step(step, message):
    """Log a step with consistent formatting"""
    logging.info(f"[STEP {step}] {message}")

def run_command(cmd, cwd=None, env=None):
    """Run a shell command with error handling"""
    try:
        result = subprocess.run(
            cmd, shell=True, check=True, capture_output=True, text=True,
            cwd=cwd, env=env
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed: {cmd}")
        logging.error(f"Error: {e.stderr}")
        raise

def wait_for_server(url, max_attempts=30):
    """Wait for server to be ready"""
    import requests
    
    for attempt in range(max_attempts):
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                return True
        except:
            pass
        
        time.sleep(2)
        if attempt % 5 == 0:
            logging.info(f"⏳ Waiting for server... (attempt {attempt + 1}/{max_attempts})")
    
    return False

def exponential_backoff_with_jitter(attempt, base_delay=1, max_delay=30):
    """Calculate delay for exponential backoff with jitter"""
    delay = min(base_delay * (2 ** attempt), max_delay)
    jitter = delay * 0.25
    delay += random.uniform(-jitter, jitter)
    return max(delay, 0.1)

def has_error_content(content):
    """Check if content contains error indicators"""
    if not content or not isinstance(content, str):
        return True
    content_lower = content.lower()
    return any(error_msg.lower() in content_lower for error_msg in ERROR_MESSAGES)

# =============================================================================
# MODEL SETUP
# =============================================================================

def setup_model(hf_token):
    """Download and convert model to GGUF format"""
    log_step(1, "Setting up model...")
    
    model_dir = Path(WORKSPACE_DIR) / "models"
    gguf_dir = model_dir / "gguf"
    local_model_dir = model_dir / "qwen2.5-vl-aio"
    
    # Create directories
    model_dir.mkdir(parents=True, exist_ok=True)
    gguf_dir.mkdir(exist_ok=True)
    
    # Check if model already exists
    main_gguf = gguf_dir / "qwen2.5-vl-aio-q8.gguf"
    mmproj_gguf = gguf_dir / "mmproj-qwen2.5-vl-aio.gguf"
    
    if main_gguf.exists() and mmproj_gguf.exists():
        logging.info("✅ Model already exists, skipping download and conversion")
        return str(main_gguf), str(mmproj_gguf)
    
    # Set HF token
    os.environ['HUGGINGFACE_HUB_TOKEN'] = hf_token
    
    # Download model if not exists
    if not local_model_dir.exists():
        logging.info("📥 Downloading model from Hugging Face...")
        cmd = f"huggingface-cli download {MODEL_NAME} --local-dir {local_model_dir} --token {hf_token}"
        run_command(cmd, cwd=model_dir)
    
    # Convert to GGUF
    llama_dir = Path(WORKSPACE_DIR) / "llama.cpp"
    
    if not main_gguf.exists():
        logging.info("🔄 Converting main model to GGUF Q8...")
        cmd = f"python convert_hf_to_gguf.py {local_model_dir} --outfile {main_gguf} --outtype q8_0"
        run_command(cmd, cwd=llama_dir)
    
    if not mmproj_gguf.exists():
        logging.info("🔄 Converting multimodal projector...")
        cmd = f"python convert_hf_to_gguf.py {local_model_dir} --mmproj --outfile {mmproj_gguf}"
        run_command(cmd, cwd=llama_dir)
    
    logging.info("✅ Model setup completed")
    return str(main_gguf), str(mmproj_gguf)

# =============================================================================
# DATASET SETUP
# =============================================================================

def setup_dataset():
    """Download dataset from Google Drive"""
    log_step(2, "Setting up dataset...")
    
    dataset_dir = Path(WORKSPACE_DIR) / "dataset"
    dataset_zip = dataset_dir / "dataset.zip"
    ground_truth_dir = dataset_dir / "ground_truth"
    
    # Create directory
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if dataset already exists
    if ground_truth_dir.exists() and (ground_truth_dir / "captions").exists():
        logging.info("✅ Dataset already exists, skipping download")
        return str(ground_truth_dir)
    
    # Download dataset
    logging.info(f"📥 Downloading dataset from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={DATASET_GDRIVE_ID}", 
                   str(dataset_zip), quiet=False)
    
    # Extract dataset
    logging.info("📦 Extracting dataset...")
    import zipfile
    with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)
    
    # Clean up zip file
    dataset_zip.unlink()
    
    # Verify structure
    if not ground_truth_dir.exists():
        raise FileNotFoundError("Dataset extraction failed - ground_truth directory not found")
    
    # Create captions directory if it doesn't exist
    captions_dir = ground_truth_dir / "captions"
    captions_dir.mkdir(exist_ok=True)
    
    logging.info("✅ Dataset setup completed")
    return str(ground_truth_dir)

# =============================================================================
# SERVER MANAGEMENT
# =============================================================================

def start_server(model_path, mmproj_path):
    """Start llama-server with vision support"""
    log_step(3, "Starting llama-server...")
    
    global server_process
    
    # Check if server is already running
    import requests
    try:
        response = requests.get(f"http://localhost:{SERVER_PORT}/health", timeout=2)
        if response.status_code == 200:
            logging.info("✅ Server already running")
            return f"http://localhost:{SERVER_PORT}/v1"
    except:
        pass
    
    # Start server
    llama_dir = Path(WORKSPACE_DIR) / "llama.cpp"
    server_bin = llama_dir / "build" / "bin" / "llama-server"
    
    if not server_bin.exists():
        raise FileNotFoundError("llama-server binary not found. Run setup.sh first.")
    
    cmd = [
        str(server_bin),
        "-m", model_path,
        "--mmproj", mmproj_path,
        "-ngl", "999",  # Use all GPU layers
        "-c", "8192",   # Context size
        "-np", "10",    # Parallel requests
        "--host", "0.0.0.0",
        "--port", str(SERVER_PORT),
        "--cont-batching",
        "--verbose"
    ]
    
    logging.info(f"🚀 Starting server: {' '.join(cmd)}")
    
    # Set CUDA environment if available
    env = os.environ.copy()
    if shutil.which("nvcc"):
        env["CUDA_VISIBLE_DEVICES"] = "0"
    
    server_process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
    )
    
    # Wait for server to start
    endpoint_url = f"http://localhost:{SERVER_PORT}/v1"
    if wait_for_server(f"http://localhost:{SERVER_PORT}"):
        logging.info(f"✅ Server started successfully at {endpoint_url}")
        return endpoint_url
    else:
        # Check if process died
        if server_process.poll() is not None:
            stdout, stderr = server_process.communicate()
            logging.error(f"Server failed to start. stderr: {stderr.decode()}")
        raise RuntimeError("Server failed to start within timeout")

# =============================================================================
# IMAGE PROCESSING
# =============================================================================

def process_image(image_path, client, max_retries=3):
    """Process a single image with retry logic"""
    for attempt in range(max_retries + 1):
        try:
            # Read and encode image
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            
            # Validate image
            try:
                with Image.open(image_path) as img:
                    img.verify()
            except Exception as e:
                return json.dumps({"error": f"Invalid image file {image_path}: {str(e)}"})
            
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            
            # Determine MIME type
            ext = Path(image_path).suffix.lower()
            mime_type = {
                ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png", ".gif": "image/gif",
                ".webp": "image/webp", ".bmp": "image/bmp"
            }.get(ext, "image/jpeg")
            
            # Make API request
            response = client.chat.completions.create(
                model="qwen2.5-vl-aio",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": VIETNAMESE_PROMPT
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=MAX_TOKENS,
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            
            # Validate JSON response
            try:
                json.loads(content)
                return content
            except json.JSONDecodeError:
                if attempt < max_retries:
                    logging.warning(f"⚠️ Invalid JSON response for {image_path}, retrying...")
                    continue
                else:
                    return json.dumps({
                        "error": "Invalid JSON response",
                        "raw_content": content[:500] + "..." if len(content) > 500 else content
                    })
            
        except Exception as e:
            error_str = str(e)
            
            # Check if retryable error
            is_retryable = any(code in error_str.lower() for code in 
                             ['429', '500', '502', '503', 'timeout', 'connection'])
            
            if is_retryable and attempt < max_retries:
                delay = exponential_backoff_with_jitter(attempt)
                logging.warning(f"🔄 Retryable error for {image_path} (attempt {attempt + 1}): {error_str}")
                time.sleep(delay)
                continue
            else:
                logging.error(f"❌ Failed to process {image_path}: {error_str}")
                return json.dumps({"error": error_str})
    
    return json.dumps({"error": "Max retries exceeded"})

# =============================================================================
# MAIN PROCESSING
# =============================================================================

def process_images(ground_truth_dir, endpoint_url, max_workers=4, fix_errors=False):
    """Process all images with parallel workers"""
    log_step(4, "Processing images...")
    
    input_dir = Path(ground_truth_dir)
    output_dir = input_dir / "captions"
    checkpoint_file = input_dir / "checkpoint.pkl"
    
    output_dir.mkdir(exist_ok=True)
    
    # Load checkpoint
    processed_files = set()
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'rb') as f:
                processed_files = pickle.load(f)
            logging.info(f"📂 Loaded checkpoint: {len(processed_files)} processed files")
        except Exception as e:
            logging.warning(f"⚠️ Failed to load checkpoint: {e}")
    
    # Find all image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_dir.rglob(f"*{ext}"))
        image_files.extend(input_dir.rglob(f"*{ext.upper()}"))
    
    # Filter files to process
    files_to_process = []
    for image_path in image_files:
        rel_path = image_path.relative_to(input_dir)
        output_path = output_dir / (rel_path.stem + ".txt")
        
        # Check if already processed and not fixing errors
        if rel_path in processed_files and not fix_errors:
            continue
        
        # If fixing errors, check if output has errors
        if fix_errors and output_path.exists():
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                if not has_error_content(content):
                    continue  # Skip files without errors
            except:
                pass  # Process if can't read file
        
        files_to_process.append((image_path, output_path, rel_path))
    
    if not files_to_process:
        logging.info("✅ All images already processed!")
        return
    
    logging.info(f"📊 Found {len(files_to_process)} images to process")
    
    # Create OpenAI client
    client = OpenAI(base_url=endpoint_url, api_key="not-needed")
    
    # Process with parallel workers
    def process_single(args):
        image_path, output_path, rel_path = args
        
        if shutdown_requested:
            return False
        
        try:
            result = process_image(image_path, client)
            
            # Save result
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result)
            
            # Update checkpoint
            processed_files.add(rel_path)
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(processed_files, f)
            
            # Log status
            status = "❌" if has_error_content(result) else "✅"
            logging.info(f"{status} {rel_path}")
            return True
            
        except Exception as e:
            logging.error(f"❌ Failed to process {rel_path}: {e}")
            return False
    
    # Run with progress bar
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(files_to_process), desc="Processing images") as pbar:
            futures = [executor.submit(process_single, args) for args in files_to_process]
            
            for future in as_completed(futures):
                if shutdown_requested:
                    break
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"❌ Worker error: {e}")
                finally:
                    pbar.update(1)
    
    logging.info("✅ Image processing completed")

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Llama Vision Captioning - One Command Solution")
    parser.add_argument("--hf-token", required=True, help="Hugging Face token for model access")
    parser.add_argument("--max-workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--fix-errors", action="store_true", help="Only process files with errors")
    parser.add_argument("--skip-model", action="store_true", help="Skip model setup (if already done)")
    parser.add_argument("--skip-dataset", action="store_true", help="Skip dataset setup (if already done)")
    
    args = parser.parse_args()
    
    try:
        logging.info("🚀 Starting Llama Vision Captioning...")
        
        # Create workspace
        Path(WORKSPACE_DIR).mkdir(exist_ok=True)
        os.chdir(Path.cwd())  # Ensure we're in the right directory
        
        # Step 1: Setup model
        if not args.skip_model:
            model_path, mmproj_path = setup_model(args.hf_token)
        else:
            model_dir = Path(WORKSPACE_DIR) / "models" / "gguf"
            model_path = str(model_dir / "qwen2.5-vl-aio-q8.gguf")
            mmproj_path = str(model_dir / "mmproj-qwen2.5-vl-aio.gguf")
            logging.info("⏭️ Skipping model setup")
        
        # Step 2: Setup dataset
        if not args.skip_dataset:
            ground_truth_dir = setup_dataset()
        else:
            ground_truth_dir = str(Path(WORKSPACE_DIR) / "dataset" / "ground_truth")
            logging.info("⏭️ Skipping dataset setup")
        
        # Step 3: Start server
        endpoint_url = start_server(model_path, mmproj_path)
        
        # Step 4: Process images
        process_images(ground_truth_dir, endpoint_url, args.max_workers, args.fix_errors)
        
        logging.info("🎉 All tasks completed successfully!")
        
    except KeyboardInterrupt:
        logging.info("👋 Interrupted by user")
    except Exception as e:
        logging.error(f"💥 Fatal error: {e}")
        sys.exit(1)
    finally:
        # Cleanup
        if server_process:
            logging.info("🔌 Stopping server...")
            server_process.terminate()

if __name__ == "__main__":
    main()
