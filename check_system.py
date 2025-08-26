#!/usr/bin/env python3
"""
System Check for Llama Vision Captioning
========================================

This script checks if your system meets the requirements
for running the llama vision captioning pipeline.
"""

import sys
import subprocess
import shutil
import platform
from pathlib import Path

def check_command(cmd):
    """Check if a command is available"""
    return shutil.which(cmd) is not None

def get_gpu_info():
    """Get GPU information"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip().split('\n')
    except:
        pass
    return None

def get_memory_info():
    """Get system memory information"""
    try:
        if platform.system() == "Linux":
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemTotal:' in line:
                        mem_kb = int(line.split()[1])
                        mem_gb = mem_kb / (1024 * 1024)
                        return f"{mem_gb:.1f} GB"
        elif platform.system() == "Darwin":  # macOS
            result = subprocess.run(['sysctl', 'hw.memsize'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                mem_bytes = int(result.stdout.split()[1])
                mem_gb = mem_bytes / (1024 * 1024 * 1024)
                return f"{mem_gb:.1f} GB"
    except:
        pass
    return "Unknown"

def check_disk_space():
    """Check available disk space"""
    try:
        stat = shutil.disk_usage('.')
        free_gb = stat.free / (1024 * 1024 * 1024)
        return f"{free_gb:.1f} GB"
    except:
        return "Unknown"

def main():
    print("🔍 Llama Vision Captioning - System Check")
    print("=" * 50)
    
    # System info
    print(f"🖥️  OS: {platform.system()} {platform.release()}")
    print(f"🐍 Python: {sys.version.split()[0]}")
    print(f"💾 RAM: {get_memory_info()}")
    print(f"💿 Disk Space: {check_disk_space()}")
    print()
    
    # Check required commands
    print("📋 Required Commands:")
    commands = {
        'conda': 'Conda package manager',
        'git': 'Git version control', 
        'cmake': 'CMake build system',
        'make': 'Make build tool'
    }
    
    all_good = True
    for cmd, desc in commands.items():
        status = "✅" if check_command(cmd) else "❌"
        print(f"  {status} {cmd:<10} - {desc}")
        if not check_command(cmd):
            all_good = False
    print()
    
    # Check optional commands
    print("🔧 Optional Commands (for better performance):")
    optional_commands = {
        'nvcc': 'NVIDIA CUDA compiler',
        'nvidia-smi': 'NVIDIA GPU monitoring',
        'brew': 'Homebrew package manager (macOS)'
    }
    
    for cmd, desc in optional_commands.items():
        status = "✅" if check_command(cmd) else "⚠️ "
        print(f"  {status} {cmd:<12} - {desc}")
    print()
    
    # GPU Information
    print("🎮 GPU Information:")
    gpu_info = get_gpu_info()
    if gpu_info:
        for gpu in gpu_info:
            name, memory = gpu.split(', ')
            print(f"  ✅ {name} ({memory} MB VRAM)")
    else:
        print("  ⚠️  No NVIDIA GPU detected (will use CPU)")
    print()
    
    # Python packages check
    print("📦 Python Environment:")
    try:
        import conda
        env_name = subprocess.run(['conda', 'info', '--json'], 
                                capture_output=True, text=True)
        if env_name.returncode == 0:
            import json
            info = json.loads(env_name.stdout)
            active_env = Path(info['active_prefix']).name
            print(f"  ✅ Active environment: {active_env}")
        else:
            print("  ⚠️  Could not detect conda environment")
    except:
        print("  ⚠️  Conda not available in Python")
    print()
    
    # Recommendations
    print("🎯 Recommendations:")
    if not all_good:
        print("  ❌ Missing required commands. Please install:")
        for cmd, desc in commands.items():
            if not check_command(cmd):
                if platform.system() == "Linux":
                    if cmd in ['cmake', 'make']:
                        print(f"     sudo apt install {cmd}")
                    elif cmd == 'git':
                        print(f"     sudo apt install git")
                elif platform.system() == "Darwin":
                    if cmd == 'conda':
                        print(f"     Download from: https://docs.conda.io/en/latest/miniconda.html")
                    else:
                        print(f"     brew install {cmd} (or xcode-select --install)")
        print()
    
    if not gpu_info:
        print("  ⚠️  Consider installing NVIDIA GPU support for faster processing")
        print("     - Install NVIDIA drivers")
        print("     - Install CUDA toolkit")
        print("     - Processing will be slower on CPU but still works")
        print()
    
    memory_gb = get_memory_info().replace(" GB", "")
    try:
        if float(memory_gb) < 8:
            print("  ⚠️  Less than 8GB RAM detected. Consider:")
            print("     - Reducing --max-workers to 1 or 2")
            print("     - Closing other applications during processing")
            print()
    except:
        pass
    
    # Final status
    if all_good:
        print("🎉 System looks good! You can run:")
        print("   ./setup.sh")
        print("   conda activate llama-vision")
        print("   python run.py --hf-token YOUR_TOKEN")
    else:
        print("⚠️  Please install missing requirements first, then run setup.sh")
    
    print()
    print("For more help, see README.md")

if __name__ == "__main__":
    main()
