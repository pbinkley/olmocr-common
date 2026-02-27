# based on https://docs.pytorch.org/docs/stable/notes/mps.html

async def run_inference(platform, query, MODEL_ID, benchmarking, pdf_path):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).eval()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    device = torch.device("mps")
    model.to(device)

    # Render page 1 to an image
    image_base64 = render_pdf_to_base64png(pdf_path, 1, target_longest_image_dim=1288)

    # Build the full prompt
    messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": build_no_anchoring_v4_yaml_prompt()},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
        ],
    }
    ]

    # Apply the chat template and processor
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    main_image = Image.open(BytesIO(base64.b64decode(image_base64)))

    inputs = processor(
        text=[text],
        images=[main_image],
        padding=True,
        return_tensors="pt",
        )
    inputs = {key: value.to(device) for (key, value) in inputs.items()}


    # Generate the output
    output = model.generate(
        **inputs,
        max_new_tokens = args.max_tokens,
        temperature = args.temperature,
        num_return_sequences=1,
        do_sample=True,
        )

    # Decode the output
    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = output[:, prompt_length:]
    text_output = processor.tokenizer.batch_decode(
        new_tokens, skip_special_tokens=True
        )

    print(text_output)
