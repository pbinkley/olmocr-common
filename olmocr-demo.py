import os
import torch
import asyncio
import base64
import io
import json
import glob
from PIL import Image
from tqdm import tqdm
from olmocr.pipeline import build_page_query
from termcolor import colored, cprint

#import olmocr

from shared_resources import *

# Model ID
MODEL_ID = "allenai/olmOCR-7B-0225-preview"




async def process_pdf(pdf_path, output_folder):
    device = get_device()
    print_announcement(f"Device: {device}")
    file_name = os.path.basename(pdf_path)
    print_announcement(f"File: {file_name}")
    
    # build_page_query handles PDF conversion to image + anchor text
    # It returns a PromptObject which behaves like a list/dict
    query = await build_page_query(pdf_path, page=1, target_longest_image_dim=1024)

    print_announcement("Have query")

    if device == "mlx":
        result, gen_time, processor = await run_inference.run_inference(device, pdf_path, query, MODEL_ID, False)
    else:
        result = await run_torch_inference(query, device)

    print_announcement("Have result")

    output_text = result.text if hasattr(result, "text") else str(result)
    print_announcement(output_text)
    
    # Save output
    output_path = os.path.join(output_folder, f"{file_name}.json")
    with open(output_path, "w") as f:
        json.dump({"file": file_name, "text": output_text}, f, indent=4)

async def main():
    input_dir = "./docs"
    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)
    
    files = glob.glob(os.path.join(input_dir, "*.pdf"))
    print_announcement(f"Process {len(files)} file{"s" if len(files) > 1 else ""}")
    file_counter = 1
    for f in tqdm(files):
        print_announcement(f"File {file_counter}: {f}")
        file_counter += 1
        await process_pdf(f, output_dir)

if __name__ == "__main__":
    asyncio.run(main())