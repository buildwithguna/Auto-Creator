from __future__ import annotations

import argparse
import json
import os
import re
import sys
from io import BytesIO
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from PIL import Image

from config import (
    CAPTION_FILE_SUFFIX,
    CAPTION_MAX_TOKENS,
    CAPTION_TEMPERATURE,
    CAPTION_TOP_P,
    DEFAULT_CAPTION_MODEL,
    DEFAULT_IMAGE_DIR,
    DEFAULT_IMAGE_MODEL,
    IMAGE_FILE_EXTENSION,
    IMAGE_MAX_PROMPT_CHARS,
    IMAGE_SIZE,
    MAX_HASHTAGS,
    ROOT_DIR,
)


@dataclass(frozen=True)
class GeneratedPost:
    prompt: str
    image_prompt: str
    image_path: Path
    caption_path: Path
    caption: str
    hashtags: str
    alt_text: str
    first_comment: str


def load_environment() -> None:
    candidates = [
        ROOT_DIR / ".env",
        Path(__file__).resolve().parent / ".env",
        Path.cwd() / ".env",
    ]
    for candidate in candidates:
        if candidate.exists():
            load_dotenv(candidate, override=False)


def require_token() -> str:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise RuntimeError(
            "Missing HF token. Add HF_TOKEN to the .env file in the project root."
        )
    return token


def slugify(text: str, limit: int = 48) -> str:
    limit = min(limit, IMAGE_MAX_PROMPT_CHARS)
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return slug[:limit].strip("-") or "post"


def build_image_prompt(user_prompt: str) -> str:
    return (
        "Create a photorealistic, premium Instagram-ready image for this brief: "
        f"{user_prompt.strip()}. "
        "Style requirements: ultra-realistic, editorial quality, natural lighting, "
        "sharp focus, rich but believable colors, cinematic composition, clean background, "
        "modern social media aesthetic, high detail, no text, no logos, no watermark, "
        "no distorted anatomy, no low-res artifacts. "
        "The final image should feel like a polished creator brand photo that can be posted "
        "directly to Instagram."
    )


def upscale_to_2k(image: Image.Image) -> Image.Image:
    target = IMAGE_SIZE
    if image.size == target:
        return image
    return image.resize(target, Image.Resampling.LANCZOS)


def save_image(image: Image.Image, output_dir: Path, prompt: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{stamp}_{slugify(prompt)}{IMAGE_FILE_EXTENSION}"
    image_path = output_dir / filename
    image.save(image_path, format="PNG", optimize=True)
    return image_path


def parse_json_block(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            return json.loads(match.group(0))
        raise


def fallback_caption(prompt: str) -> dict[str, str]:
    caption = (
        f"Turned this idea into a clean, high-impact visual: {prompt.strip()}. "
        "Built for scroll-stopping engagement, brand polish, and a premium Instagram feel."
    )
    hashtags = " ".join(
        [
            "#instagramcontent",
            "#contentcreator",
            "#visualstorytelling",
            "#aesthetic",
            "#photorealistic",
            "#socialmediacontent",
            "#brandcontent",
            "#creatorstudio",
            "#instagood",
            "#explorepage",
        ]
    )
    return {
        "caption": caption,
        "hashtags": hashtags,
        "alt_text": prompt.strip(),
        "first_comment": "What do you think of this version?",
    }


def normalize_hashtags(raw_hashtags: str, prompt: str) -> str:
    tokens = re.split(r"[\s,]+", raw_hashtags.strip())
    cleaned: list[str] = []
    for token in tokens:
        token = token.strip().lstrip("#")
        token = re.sub(r"[^A-Za-z0-9_]", "", token)
        if token:
            cleaned.append(f"#{token}")

    if not cleaned:
        return fallback_caption(prompt)["hashtags"]

    seen: set[str] = set()
    unique: list[str] = []
    for tag in cleaned:
        lower = tag.lower()
        if lower not in seen:
            seen.add(lower)
            unique.append(tag)

    return " ".join(unique[:MAX_HASHTAGS])


def clean_caption_text(raw_caption: str, prompt: str) -> str:
    caption = re.sub(r"(?<!\S)#[A-Za-z0-9_]+", "", raw_caption)
    caption = re.sub(r"[ \t]+", " ", caption)
    caption = re.sub(r"\n{3,}", "\n\n", caption)
    caption = re.sub(r" *\n *", "\n", caption).strip()
    return caption or fallback_caption(prompt)["caption"]


def generate_caption_bundle(client: InferenceClient, prompt: str, image_prompt: str) -> dict[str, str]:
    messages = [
        {
            "role": "system",
            "content": (
                "You write premium Instagram post packages. "
                "Return only valid JSON with exactly these keys: caption, hashtags, alt_text, first_comment. "
                "Use no markdown. Use no emojis. Keep the caption polished, natural, and conversion-friendly. "
                "Make hashtags a single space-separated string with 8 to 15 relevant tags. "
                "Make alt_text concise and descriptive. "
                "Make first_comment a short engagement prompt."
            ),
        },
        {
            "role": "user",
            "content": (
                f"User prompt: {prompt.strip()}\n"
                f"Image prompt used: {image_prompt.strip()}"
            ),
        },
    ]

    try:
        raw = client.chat_completion(
            messages=messages,
            model=DEFAULT_CAPTION_MODEL,
            max_tokens=CAPTION_MAX_TOKENS,
            temperature=CAPTION_TEMPERATURE,
            top_p=CAPTION_TOP_P,
            response_format={"type": "json_object"},
        )
        data = parse_json_block(raw.choices[0].message.content)
        return {
            "caption": clean_caption_text(str(data.get("caption", "")).strip(), prompt),
            "hashtags": normalize_hashtags(str(data.get("hashtags", "")).strip(), prompt),
            "alt_text": str(data.get("alt_text", "")).strip() or fallback_caption(prompt)["alt_text"],
            "first_comment": str(data.get("first_comment", "")).strip()
            or fallback_caption(prompt)["first_comment"],
        }
    except Exception:
        return fallback_caption(prompt)


def write_caption_file(output_dir: Path, prompt: str, image_path: Path, bundle: dict[str, str]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{stamp}_{slugify(prompt)}{CAPTION_FILE_SUFFIX}"
    caption_path = output_dir / filename

    content = (
        f"# Instagram Post Bundle\n\n"
        f"Source prompt:\n{prompt.strip()}\n\n"
        f"Image file:\n{image_path}\n\n"
        f"Caption:\n{bundle['caption'].strip()}\n\n"
        f"Hashtags:\n{bundle['hashtags'].strip()}\n\n"
        f"Alt text:\n{bundle['alt_text'].strip()}\n\n"
        f"First comment:\n{bundle['first_comment'].strip()}\n"
    )
    caption_path.write_text(content, encoding="utf-8")
    return caption_path


def generate_instagram_post(prompt: str, output_dir: Path) -> GeneratedPost:
    load_environment()
    token = require_token()
    client = InferenceClient(token=token)

    image_prompt = build_image_prompt(prompt)
    generated_image = client.text_to_image(
        prompt=image_prompt,
        model=DEFAULT_IMAGE_MODEL,
    )

    if not isinstance(generated_image, Image.Image):
        if isinstance(generated_image, (bytes, bytearray)):
            generated_image = Image.open(BytesIO(generated_image))
        else:
            generated_image = Image.open(generated_image)

    image = upscale_to_2k(generated_image)
    image_path = save_image(image, output_dir, prompt)

    bundle = generate_caption_bundle(client, prompt, image_prompt)
    caption_path = write_caption_file(output_dir, prompt, image_path, bundle)

    return GeneratedPost(
        prompt=prompt,
        image_prompt=image_prompt,
        image_path=image_path,
        caption_path=caption_path,
        caption=bundle["caption"],
        hashtags=bundle["hashtags"],
        alt_text=bundle["alt_text"],
        first_comment=bundle["first_comment"],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate an Instagram-ready image and post package from a user prompt."
    )
    parser.add_argument(
        "--prompt",
        help="Describe the image you want to generate.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_IMAGE_DIR),
        help="Directory where the image and caption bundle will be saved.",
    )
    return parser


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = build_parser()
    args = parser.parse_args()

    prompt = args.prompt or input("Describe the Instagram post you want to generate: ").strip()
    if not prompt:
        raise SystemExit("A prompt is required.")

    result = generate_instagram_post(prompt, Path(args.output_dir))

    print(f"Image saved: {result.image_path}")
    print(f"Caption bundle saved: {result.caption_path}")
    print("\nCaption:\n" + result.caption)
    print("\nHashtags:\n" + result.hashtags)
    print("\nAlt text:\n" + result.alt_text)
    print("\nFirst comment:\n" + result.first_comment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
