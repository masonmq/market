#!/usr/bin/env python3
"""
Gemini-backed agent that predicts a discrete popularity level (1–5) from a post
image plus metadata, using the same rubric prompt as the OpenAI variant.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

import google.generativeai as genai
from PIL import Image

from post_metadata import post_metadata

# Prefer GEMINI_API_KEY; GOOGLE_API_KEY is the common name in Google AI Studio docs.
#GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
GEMINI_API_KEY = "AIzaSyA4CPL_tuilGaL-rKN94T-FyBM-TKTfE-Q"
SYSTEM_PROMPT_TEMPLATE = """You are a senior social media analyst with expertise in {agent_expert_domain}, and you have years of experience in audience behavior, feed algorithms, and cross-platform engagement patterns.

You evaluate the post strictly based on the provided image and metadata.

The metadata includes:

### Category

* `Category`: the primary category of the post (11 classes).
* `Subcategory`: one of 77 second-level categories.
* `Concept`: one of 668 different descriptions.

### Text

* `Title`: the title of the post defined by the user.
* `Mediatype`: the type of attached media, including `photo` and `video`.
* `Alltags`: user-defined tags.

### Temporal-spatial

* `Postdate`: the publication timestamp of the post. It can be converted to datetime using the following Python code:

```python
import time
timestamp = 1457068974
timeArray = time.localtime(timestamp)
datetime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
```

* `Latitude`: latitude in the range [-90, 90]. Values beyond 6 decimal places are truncated.
* `Longitude`: longitude in the range [-180, 180]. Values beyond 6 decimal places are truncated.
* `Geoaccuracy`: accuracy level of the location information.

  * World level ≈ 1
  * Country ≈ 3
  * Region ≈ 6
  * City ≈ 11
  * Street ≈ 16
    The valid range is 1–16, with a default value of 16 if unspecified.

### User Profile

* `Photo_firstdate`: the upload date of the user's first photo.
* `Photo_count`: the total number of photos posted by the user.
* `Ispro`: whether the user is a Pro member.
* `Photo_firstdatetaken`: the capture date of the user's first photo.
* `Timezone_offset`: the user's timezone offset.
* `User description`: descriptive information about the user.
* `Location description`: descriptive information about the user's location.

### Additional Information

* `Pathalias`: the user-defined path alias.
* `Ispublic`: indicates whether the post has public read permissions.
* `Mediastatus`: indicates whether the media is ready to be accessed by others.

Your task is to predict the **popularity_level** of a post relative to typical user-generated photo content on a large social media platform, using the provided image and metadata.

Do NOT claim or assume real view counts or external engagement data.

### Scoring Rubric (1–5)

* `1` = Very low expected engagement
* `2` = Below average engagement
* `3` = Average content
* `4` = Above average engagement
* `5` = High viral potential

### Output format

Return ONLY valid JSON (no markdown, no explanation):

```json
{{
  "popularity_level": <integer from 1 to 5>
}}
```
"""


def _resolve_image_path(image_path: str) -> Path:
    p = Path(image_path)
    if not p.is_file():
        candidate = Path(__file__).resolve().parent / image_path
        if candidate.is_file():
            p = candidate
        else:
            raise FileNotFoundError(
                f"Image not found: {image_path!r} (also tried {str(candidate)!r})"
            )
    return p


def _coerce_popularity_level(obj: Any) -> int:
    if obj is None:
        raise TypeError("Missing popularity_level in model output.")

    if isinstance(obj, bool):
        raise TypeError(f"popularity_level must be an integer 1–5, got bool: {obj!r}")

    if isinstance(obj, int):
        if 1 <= obj <= 5:
            return obj
        raise ValueError(f"popularity_level out of range: {obj!r}")

    if isinstance(obj, float):
        if obj.is_integer() and 1 <= int(obj) <= 5:
            return int(obj)
        raise ValueError(f"popularity_level must be an integer 1–5, got {obj!r}")

    if isinstance(obj, str):
        s = obj.strip()
        try:
            v = int(s)
        except ValueError as e:
            raise ValueError(f"popularity_level not parseable as int: {obj!r}") from e
        if 1 <= v <= 5:
            return v
        raise ValueError(f"popularity_level out of range after parse: {obj!r}")

    raise TypeError(
        f"popularity_level must be int-like (1–5), got {type(obj).__name__}: {obj!r}"
    )


def _call_gemini(
    system_prompt: str,
    metadata_text: str,
    image_path: Path,
    *,
    api_key: str,
    model_name: str,
) -> dict[str, Any]:
    if not api_key:
        raise RuntimeError(
            "Missing Gemini API key. Set GEMINI_API_KEY or GOOGLE_API_KEY, "
            "or pass --api-key when running this script."
        )

    genai.configure(api_key=api_key)

    generation_config: dict[str, Any] = {
        "temperature": 0.2,
        "response_mime_type": "application/json",
    }

    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_prompt,
        generation_config=generation_config,
    )

    last_err: Exception | None = None
    for attempt in range(1, 4):
        try:
            img = Image.open(image_path)
            try:
                response = model.generate_content([metadata_text, img])
            finally:
                img.close()

            try:
                text = response.text
            except ValueError as e:
                fb = getattr(response, "prompt_feedback", None)
                raise ValueError(
                    f"Gemini returned no usable text (blocked or empty). prompt_feedback={fb!r}"
                ) from e

            if not text or not text.strip():
                raise ValueError("Gemini response had empty content.")

            return json.loads(text.strip())
        except Exception as e:
            last_err = e
            if attempt >= 3:
                break
            time.sleep(min(20.0, (2 ** (attempt - 1)) + random.random()))

    assert last_err is not None
    raise last_err


def _resolve_metadata_text(
    image_path: str,
    *,
    metadata: str | None,
    metadata_file: str | None,
) -> str:
    if metadata_file:
        p = Path(metadata_file)
        if not p.is_file():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file!r}")
        return p.read_text(encoding="utf-8")
    if metadata is not None:
        return metadata

    post_data = json.loads(post_metadata(image_path))
    if "metadata" not in post_data or not isinstance(post_data["metadata"], str):
        raise KeyError("post_metadata output missing string field 'metadata'")
    return post_data["metadata"]


def popularity_level_prediction_with_agent(
    image_path: str,
    expert_domain: str,
    *,
    metadata: str | None = None,
    metadata_file: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """
    Call Gemini with the popularity-level rubric prompt, image, and metadata.

    Metadata defaults to the dataset bundle from ``post_metadata(image_path)``.
    Override with ``metadata=...`` or ``metadata_file=...``.

    Returns:
        {"popularity_level": int}  # int in 1..5
    """
    key = api_key if api_key is not None else GEMINI_API_KEY
    model_name = model if model is not None else os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    metadata_text = _resolve_metadata_text(
        image_path, metadata=metadata, metadata_file=metadata_file
    )
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(agent_expert_domain=expert_domain)
    resolved = _resolve_image_path(image_path)

    raw = _call_gemini(
        system_prompt,
        metadata_text,
        resolved,
        api_key=key,
        model_name=model_name,
    )
    level = _coerce_popularity_level(raw.get("popularity_level"))
    return {"popularity_level": level}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Predict popularity_level (1–5) from image + metadata via Gemini."
    )
    parser.add_argument("image_path", nargs="?", default="train/3175@N73/30916.jpg")
    parser.add_argument("--expert-domain", default="Entertainment")
    parser.add_argument(
        "--metadata",
        default=None,
        help="Inline metadata text (overrides dataset lookup unless --metadata-file).",
    )
    parser.add_argument(
        "--metadata-file",
        default=None,
        help="Path to a UTF-8 text file used as metadata (highest precedence).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Gemini API key (overrides GEMINI_API_KEY / GOOGLE_API_KEY).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Gemini model id (default: env GEMINI_MODEL or gemini-2.5-pro).",
    )
    args = parser.parse_args()

    api_key = args.api_key if args.api_key else GEMINI_API_KEY

    out = popularity_level_prediction_with_agent(
        args.image_path,
        args.expert_domain,
        metadata=args.metadata,
        metadata_file=args.metadata_file,
        api_key=api_key,
        model=args.model,
    )
    print(json.dumps(out, ensure_ascii=False))
