from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent

# Runtime defaults
DEFAULT_IMAGE_DIR = ROOT_DIR / "images"
DEFAULT_IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"
DEFAULT_CAPTION_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Generation controls
IMAGE_SIZE = (2048, 2048)
IMAGE_MAX_PROMPT_CHARS = 48
CAPTION_MAX_TOKENS = 350
CAPTION_TEMPERATURE = 0.7
CAPTION_TOP_P = 0.9

# Filesystem defaults
CAPTION_FILE_SUFFIX = "_instagram_post.md"
IMAGE_FILE_EXTENSION = ".png"

# Output behavior
MAX_HASHTAGS = 15
