from olmocr.pipeline import build_page_query
import base64
from PIL import Image
import io
import time

async def run_mlx_inference(pdf_path, query, MODEL_ID, benchmarking, model, processor):
    """Native Apple Silicon inference using mlx-vlm."""
    from mlx_vlm import load, generate
    from mlx_vlm.utils import load_image
    from mlx_vlm.utils import load_config
    from mlx_vlm.prompt_utils import apply_chat_template

    if benchmarking:
        print(f"📦 Loading {MODEL_ID}...")
        start_load = time.time()

    # mlx-vlm expects a specific prompt format and a PIL image or path
    config = load_config(MODEL_ID)

    if benchmarking:
        print(f"✅ Loaded in {time.time() - start_load:.2f}s")
        print(f"📄 Rendering PDF page...")

    # olmOCR's build_page_query returns a list of messages. 
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
    
    # The image is stored in the 'image' attribute of the query object
    image_data = base64.b64decode(image_base64)
    pil_image = Image.open(io.BytesIO(image_data))

    # This ensures special tokens like <|image_placeholder|> are placed correctly
    formatted_prompt = apply_chat_template(
        processor, 
        config, 
        prompt_text, 
        num_images=1
    )
    
    if benchmarking:
        print(f"🧠 Generating OCR (Page 1)...")
        start_gen = time.time()
    else:
        start_gen = None

    result = generate(
        model, 
        processor, 
        formatted_prompt, 
        [pil_image], 
        max_tokens=2048, 
        verbose=False
    )

    return result, start_gen

