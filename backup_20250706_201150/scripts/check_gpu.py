#!/usr/bin/env python3
import torch
import sys

def check_gpu():
    """Check GPU availability and setup"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"✅ GPU Available: {gpu_name}")
        print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
        print(f"🔧 CUDA Version: {torch.version.cuda}")
        return True
    else:
        print("⚠️  No GPU available - using CPU (slower)")
        return False

if __name__ == "__main__":
    gpu_available = check_gpu()
    sys.exit(0 if gpu_available else 1)
