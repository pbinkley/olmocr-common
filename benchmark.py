import time
import os
import asyncio
import base64
import io
from PIL import Image
from olmocr.pipeline import build_page_query
from mlx_vlm import load, generate
from mlx_vlm.utils import load_config
from mlx_vlm.prompt_utils import apply_chat_template
import run_mlx_inference

# Model ID
MODEL_ID = "allenai/olmOCR-7B-0225-preview"

async def benchmark_tahoe(pdf_path):
    print(f"🚀 Benchmarking on macOS Tahoe (2026)...")

    query = await build_page_query(
        pdf_path, 
        page=1, 
        target_longest_image_dim=1024
    )
    
    model, processor = load(MODEL_ID)

    result, start_gen = await run_mlx_inference.run_mlx_inference(pdf_path, query, MODEL_ID, True, model, processor)




    # Extract the text from the object
    output_text = result.text if hasattr(result, "text") else str(result)

    # 1. Try to get the actual token count from various possible attributes
    if hasattr(result, "num_tokens"):
        actual_tokens = result.num_tokens
    elif hasattr(result, "token_count"):
        actual_tokens = result.token_count
    else:
        # Manual fallback: Use the processor's tokenizer to get the exact count
        actual_tokens = len(processor.tokenizer.encode(output_text))

    # 2. Try to get the generation time

    if hasattr(result, "generation_time"):
         gen_time = result.generation_time
    else:
        # Fallback to the time we measured with time.time()
        gen_time = time.time() - start_gen


#    import pdb; pdb.set_trace()

#    gen_time = result.generation_time
#    gen_time = result.generation_tps

    print(f"\n--- TAHOE PERFORMANCE ---")
    print(f"Generated Tokens: {actual_tokens}")
    print(f"Generation Time: {gen_time:.2f}s")
    print(f"True Speed: {actual_tokens / gen_time:.2f} tokens/sec")
    print(f"---------------------------\n")

if __name__ == "__main__":
    # Ensure sample.pdf exists in your folder
    if os.path.exists("docs/sample.pdf"):
        asyncio.run(benchmark_tahoe("docs/sample.pdf"))
    else:
        print("❌ Please put a 'sample.pdf' in this folder.")