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

# Model ID
MODEL_ID = "allenai/olmOCR-7B-0225-preview"

async def benchmark_tahoe(pdf_path):
    print(f"🚀 Benchmarking on macOS Tahoe (2026)...")

    # 1. Load Model & Processor
    print(f"📦 Loading {MODEL_ID}...")
    start_load = time.time()
    model, processor = load(MODEL_ID)
    config = load_config(MODEL_ID)
    print(f"✅ Loaded in {time.time() - start_load:.2f}s")

    # 2. Build Query (olmocr 0.5.1 style)
    print(f"📄 Rendering PDF page...")
    query = await build_page_query(pdf_path, page=1, target_longest_image_dim=1024)

    # 3. Extract Prompt and Image from the Message Object
    # query format: {"model": "...", "messages": [{"role": "user", "content": [...]}]}
    user_message = query["messages"][0]["content"]
    
    prompt_text = ""
    image_base64 = ""
    
    for item in user_message:
        if item["type"] == "text":
            prompt_text = item["text"]
        elif item["type"] == "image_url":
            # Extract base64 from 'data:image/png;base64,iVBORw...'
            image_base64 = item["image_url"]["url"].split(",")[1]

    # 4. Convert Base64 back to a PIL Image for MLX-VLM
    image_data = base64.b64decode(image_base64)
    pil_image = Image.open(io.BytesIO(image_data))

    # 5. Format for MLX-VLM Chat Template
    # This ensures special tokens like <|image_placeholder|> are placed correctly
    formatted_prompt = apply_chat_template(
        processor, 
        config, 
        prompt_text, 
        num_images=1
    )

    # 6. Run Inference
    print(f"🧠 Generating OCR (Page 1)...")
    start_gen = time.time()

    # Returns a GenerationResult object
    result = generate(
        model, 
        processor, 
        formatted_prompt, 
        [pil_image], 
        max_tokens=2048, 
        verbose=True
    )

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