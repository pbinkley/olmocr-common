async def run_mlx_inference(pdf_path, query, MODEL_ID):
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

