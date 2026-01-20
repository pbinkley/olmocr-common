import platform
from termcolor import colored, cprint
import torch

print_announcement = lambda x: cprint(x, "blue", "on_light_grey")

def get_device():
    print_announcement(f"{platform.system()} / {platform.machine()}")
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        if torch.backends.mps.is_available(): 
            import run_inference_mps
            return "mps"
        else: 
            import run_inference_mlx
            return "mlx"
    if torch.cuda.is_available():
        import run_inference_torch
        return "cuda"
    else:
        import run_inference_torch
        return "cpu"
