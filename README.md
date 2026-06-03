# Auto-Creator
Generate Instagram-ready image posts from a prompt, with no UI.

## What it does

- Reads your prompt from the command line
- Uses the `HF_TOKEN` from the project `.env`
- Generates a photorealistic Instagram-style image
- Upscales the saved image to a 2048 x 2048 canvas
- Creates a ready-to-copy Instagram post bundle with:
  - caption
  - hashtags
  - alt text
  - first comment

## Project Structure

- `main.py` - CLI entrypoint and orchestration
- `config.py` - adjustable models and generation parameters
- `.env.example` - token template for local setup
- `.gitignore` - repo hygiene and local-only files
- `requirements.txt` - Python dependencies

## Run

From the project root:

```powershell
.venv\Scripts\python.exe Auto-Creator\main.py --prompt "A premium skincare product on a marble vanity with soft morning light"
```

If you skip `--prompt`, the script will ask you interactively.

## Setup

1. Create a `.env` file at the project root and add `HF_TOKEN=...`
2. Install dependencies:

```powershell
.venv\Scripts\pip.exe install -r Auto-Creator\requirements.txt
```

## Output

- Generated image: `images\*.png`
- Instagram bundle: `images\*_instagram_post.md`

## Tune

- `config.py` holds the model IDs and generation parameters
- Change `DEFAULT_IMAGE_MODEL` to switch image generation backends
- Change `DEFAULT_CAPTION_MODEL` to tune caption style
- Adjust `IMAGE_SIZE`, `CAPTION_MAX_TOKENS`, `CAPTION_TEMPERATURE`, and `MAX_HASHTAGS` to fine-tune output behavior

## Dependencies

- `huggingface_hub`
- `python-dotenv`
- `Pillow`
