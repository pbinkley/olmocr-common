import os
import torch
import platform
import asyncio
import base64
import io
import json
import glob
from PIL import Image
from tqdm import tqdm
from olmocr.pipeline import build_page_query
#import olmocr

# Model ID
MODEL_ID = "allenai/olmOCR-7B-0225-preview"

def get_runtime():
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "mlx"
    return "cuda" if torch.cuda.is_available() else "cpu"

async def run_mlx_inference(pdf_path, query):
    """Native Apple Silicon inference using mlx-vlm."""
    from mlx_vlm import load, generate
    from mlx_vlm.utils import load_image
    from mlx_vlm.utils import load_config
    from mlx_vlm.prompt_utils import apply_chat_template

    # mlx-vlm expects a specific prompt format and a PIL image or path
    model, processor = load(MODEL_ID)
    config = load_config(MODEL_ID)

    query = await build_page_query(
        pdf_path, 
        page=1, 
        target_longest_image_dim=1024
    )
    
    # olmOCR's build_page_query returns a list of messages. 
    # We extract the text prompt from the message content.
    prompt_text = ""

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
    
    result = generate(
        model, 
        processor, 
        formatted_prompt, 
        [pil_image], 
        max_tokens=2048, 
        verbose=False
    )

    return result

async def run_torch_inference(query, device):
    """Standard PyTorch inference for CUDA or CPU."""
    from transformers import AutoProcessor, AutoModelForVision2Seq
    
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID, torch_dtype=dtype, device_map=device
    )

    # Use the processor's chat template for consistency
    formatted_prompt = processor.apply_chat_template(query, add_generation_prompt=True)
    image = query[0].get("image") or query.image

    inputs = processor(text=formatted_prompt, images=image, return_tensors="pt").to(device, dtype=dtype)
    
    output = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
    return processor.batch_decode(output, skip_special_tokens=True)[0]

async def process_pdf(pdf_path, output_folder):
    runtime = get_runtime()
    file_name = os.path.basename(pdf_path)
    
    # build_page_query handles PDF conversion to image + anchor text
    # It returns a PromptObject which behaves like a list/dict
    query = await build_page_query(pdf_path, page=1, target_longest_image_dim=1024)

    if runtime == "mlx":
        result = await run_mlx_inference(pdf_path, query)
    else:
        result = await run_torch_inference(query, runtime)

    output_text = result.text if hasattr(result, "text") else str(result)

    # Save output
    output_path = os.path.join(output_folder, f"{file_name}.json")
    with open(output_path, "w") as f:
        json.dump({"file": file_name, "text": output_text}, f, indent=4)

async def main():
    input_dir = "./docs"
    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)
    
    files = glob.glob(os.path.join(input_dir, "*.pdf"))
    for f in tqdm(files):
        await process_pdf(f, output_dir)

if __name__ == "__main__":
    asyncio.run(main())