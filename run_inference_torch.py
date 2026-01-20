async def run_inference(platform, query, MODEL_ID, benchmarking, pdf_path):
# async def run_torch_inference(query, device):
    """Standard PyTorch inference for CUDA or CPU."""
    from transformers import AutoProcessor, AutoModelForVision2Seq
    
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID, dtype=dtype, device_map=device
    )

    # Use the processor's chat template for consistency
    prompt = """Attached is one page of a document that you must process. Just return the plain text representation of this document as if you were reading it naturally. Convert equations to LateX and tables to HTML.
If there are any figures or charts, label them with the following markdown syntax ![Alt text describing the contents of the figure](page_startx_starty_width_height.png)
Return your output as markdown, with a front matter section on top specifying values for the primary_language, is_rotation_valid, rotation_correction, is_table, and is_diagram parameters."""
    formatted_prompt = processor.apply_chat_template(
        query, prompt, add_generation_prompt=True
    )

    #import pdb; pdb.set_trace()
    image = query['messages'][0].get("image")# or query.image

    inputs = processor(text=formatted_prompt, images=image, return_tensors="pt").to(device, dtype=dtype)
    
    output = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
    return processor.batch_decode(output, skip_special_tokens=True)[0]
