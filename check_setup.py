import sys
import torch

print(f"Python version: {sys.version}")

try:
    import olmocr
    print("✅ olmocr: Found")
except ImportError:
    print("❌ olmocr: Not Found")

try:
    import mlx.core as mx
    import mlx_vlm
    print(f"✅ MLX: Found (Device: {mx.default_device()})")
    print("✅ mlx-vlm: Found")
except ImportError as e:
    print(f"❌ MLX Error: {e}")

if torch.backends.mps.is_available():
    print("✅ Metal Performance Shaders (MPS): Available")