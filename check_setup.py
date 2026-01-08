import sys
import torch
from importlib.metadata import version
import platform
import psutil

print(f"Python version: {sys.version}")
print(f"Torch version: {torch.__version__}")
print(f"System: {platform.system()}; {round(psutil.virtual_memory().total/1024**3, 2)}GB memory")

try:
    import olmocr
    print(f"✅ olmocr: Found {version('olmocr')}")
except ImportError:
    print("❌ olmocr: Not Found")

try:
    import mlx.core as mx
    import mlx_vlm
    print(f"✅ MLX Available (Device: {mx.default_device()})")
    print("✅ mlx-vlm: Found")
except ImportError as e:
    print(f"❌ MLX: Not Available")

if torch.backends.mps.is_available():
    print("✅ MPS: Available")
else:
    print("❌ MPS: Not Available")    

if torch.cuda.is_available():
    cuda = torch.cuda.get_device_properties(0) # TODO handle multiple GPUs
    print(f"✅ CUDA: Available: {cuda['name']} ({cuda['major']}.{cuda['minor']}, {cuda['total_memory']})")
else:
    print("❌ CUDA: Not available")

if torch.cpu.is_available():
    print(f"✅ CPU: Available: {platform.processor()}")
else:
    print("❌ CPU: Not available")

