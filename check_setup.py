import sys
import torch
from importlib.metadata import version
import platform
import psutil

import shared_resources

print(f"Python version: {sys.version}")
print(f"Torch version: {torch.__version__}")
print(f"System: {platform.system()}; Machine: {platform.machine()}")

try:
    import olmocr
    print(f"✅ olmocr: Found {version('olmocr')}")
except ImportError:
    print("❌ olmocr: Not Found")

try:
    import mlx.core as mx
    import mlx_vlm
    print(f"✅ MLX available (Device: {mx.default_device()}) Unified memory: {psutil.virtual_memory().total / (1024**3)}GB")
    print("✅ mlx-vlm: Found")
except ImportError as e:
    print(f"❌ MLX: Not available")

if torch.backends.mps.is_available():
    print("✅ MPS: Available")
else:
    print("❌ MPS: Not available")    

if torch.cuda.is_available():
    cuda = torch.cuda.get_device_properties(0) # TODO handle multiple GPUs
    print(f"✅ CUDA: Available: \"{cuda.name}\" ({cuda.major}.{cuda.minor}) {round(cuda.total_memory/1024**3, 2)}GB")
else:
    print("❌ CUDA: Not available")

if torch.cpu.is_available():
    print(f"✅ CPU: Available: {platform.processor()}; {round(psutil.virtual_memory().total/1024**3, 2)}GB memory")
else:
    print("❌ CPU: Not available")

