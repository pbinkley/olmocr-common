This is a very preliminary version of a set of scripts which are intended to support using Olmocr on various platforms. 

So far the scripts only support Mac, using [MLX](https://ml-explore.github.io/mlx/build/html/index.html) to maximize performance. The scripts have been vibe coded with Gemini. The MLX code base is changing rapidly and the difficulty of syncing up the versions of various components and the API were tricky to solve, and Gemini helped.

To install, check out this repo, create a virtual environment, and install with [uv](https://docs.astral.sh/uv/).

```
python -m venv venv
source venv/bin/activate
uv pip install -r requirements.txt
```

Then run ```python check_setup.py``` to confirm that it works. The output should look like this:

```
Python version: 3.12.7 | packaged by Anaconda, Inc. | (main, Oct  4 2024, 08:22:19) [Clang 14.0.6 ]
✅ olmocr: Found
✅ MLX: Found (Device: Device(gpu, 0))
✅ mlx-vlm: Found
✅ Metal Performance Shaders (MPS): Available
```

You can then run a benchmark. Create a directory ```docs``` and put an image pdf in it, named ```sample.pdf```. Then run ```python benchmark.py```. The output should look like this:

```
🚀 Benchmarking on macOS Tahoe (2026)...
📦 Loading allenai/olmOCR-7B-0225-preview...
Fetching 13 files: 100%|██████████████████████████████████████████████████████████████████████████████████████| 13/13 [00:00<00:00, 132666.55it/s]
The image processor of type `Qwen2VLImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`. Note that this behavior will be extended to all models in a future release.
Fetching 13 files: 100%|██████████████████████████████████████████████████████████████████████████████████████| 13/13 [00:00<00:00, 259647.39it/s]
✅ Loaded in 1.83s
📄 Rendering PDF page...
🧠 Generating OCR (Page 1)...
==========
Files: [<PIL.PngImagePlugin.PngImageFile image mode=RGB size=791x1024 at 0x12F822570>] 

Prompt: <|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Attached is one page of a document that you must process. Just return the plain text representation of this document as if you were reading it naturally. Convert equations to LateX and tables to HTML.
If there are any figures or charts, label them with the following markdown syntax ![Alt text describing the contents of the figure](page_startx_starty_width_height.png)
Return your output as markdown, with a front matter section on top specifying values for the primary_language, is_rotation_valid, rotation_correction, is_table, and is_diagram parameters.<|vision_start|><|image_pad|><|vision_end|><|im_end|>
<|im_start|>assistant

{"primary_language":"en","is_rotation_valid":true,"rotation_correction":0,"is_table":false,"is_diagram":false,"front_matter":null,"body":"December 19, 1932.\n\nProfessor Robert C. Binkley,\n95 Widener Library,\nCambridge, Massachusetts.\n\nDear Binkley:\n\nThe program as printed for the Conference of Historical Societies is wrong. I am not going to read a paper. Some months ago Dr. Flick asked me for suggestions for the program, and then proceeded to put me down for a paper without consulting me. When I discovered this and protested, he agreed to revise the program, and on November 18 he sent me a version of it which called for a paper by you on \"The Reproduction and Marketing of Historical Material,\" to be followed by a discussion led by me of the same subject. He wrote me that he had sent a copy of the corrected program to you. Apparently, however, he was too late to get the correction into the printed program.\n\nAt any rate, the fact is that I am not going to read a paper. I had no intention of doing so any way, and the fact that I have just been laid up for a week with the influenza makes it quite out of the question. You are quite free, therefore, to cover the whole field in any way you wish, and I will speak informally on such points as occur to me while listening to your paper.\n\nI quite agree with you that it would be a good opportunity to lay before the historical societies the possibility of an organized market. That is what I had in mind when I suggested the subject.\n\nSincerely yours,\n\nSolon J. Buck."}
==========
Prompt: 1166 tokens, 463.742 tokens-per-sec
Generation: 369 tokens, 31.863 tokens-per-sec
Peak memory: 19.111 GB

--- TAHOE PERFORMANCE ---
Generated Tokens: 368
Generation Time: 14.13s
True Speed: 26.05 tokens/sec
---------------------------
```

The text will of course depend on your pdf, but the tokens/sec should be a proper reflection of your hardware. The sample output above was created on a Macbook Pro with an M4 Max chip and 48GB combined memory, running Tahoe 26.1.

## Future Plans

- detect and support other platforms (Macs with different hardware, various CUDA versions, etc.)
- abstract all the hardware-dependent parts so that Olmocr scripts can be written once to run everywhere
- extend benchmarking to support e.g. comparing performance and output when using full-sized and quantized models, etc. etc.
