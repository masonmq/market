#!/usr/bin/env python3
"""
OpenAI-backed agent that predicts a discrete popularity level (1–5) from a post
image plus metadata, using the popularity-level rubric prompt.
"""

"""
2026/5/21 Modification Note – OpenAI Model Compatibility Debugging

The original implementation sent the same multimodal request to all models,
including metadata text and an image input through image_url. This caused
compatibility errors for models that do not support image input or certain
request parameters.

Observed issues:
- turbo failed because image_url is not supported.
- o1 failed when temperature was included, but ran successfully after
  temperature was disabled while keeping image_url.
- o1-mini returned model_not_found, suggesting that the current API key/project
  does not have access to it, or that it is unavailable in the current setup.
  It is also not directly applicable to the current image-based task because
  it is text-only.
- Some model outputs may vary slightly, such as gpt-4o predicting level 3 in
  one run and level 4 in another, likely due to borderline classification
  variability.

Code adjustment:
- Separated image-input support from temperature support.
- Kept image_url only for models that support image input.
- Disabled temperature for models that do not support it.

The prediction logic and JSON output parsing were not changed. This modification
only adjusts the API request format so that each model is handled according to
its supported input types and parameters.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import random
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from post_metadata import post_metadata

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

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


SUPPORTS_IMAGE_INPUT = {"o1", "gpt-4o", "gpt-4o-mini"}

NO_TEMPERATURE = {"o1", "o1-mini", "o3-mini"}


def _image_path_to_data_url(image_path: str) -> str:
    p = Path(image_path)
    if not p.is_file():
        candidate = Path(__file__).resolve().parent / image_path
        if candidate.is_file():
            p = candidate
        else:
            raise FileNotFoundError(
                f"Image not found: {image_path!r} (also tried {str(candidate)!r})"
            )

    mime, _ = mimetypes.guess_type(str(p))
    if mime is None:
        mime = "application/octet-stream"

    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _coerce_popularity_level(obj: Any) -> int:
    if obj is None:
        raise TypeError("Missing popularity_level in model output.")

    if isinstance(obj, bool):
        raise TypeError(
            f"popularity_level must be an integer 1–5, got bool: {obj!r}")

    if isinstance(obj, int):
        if 1 <= obj <= 5:
            return obj
        raise ValueError(f"popularity_level out of range: {obj!r}")

    if isinstance(obj, float):
        if obj.is_integer() and 1 <= int(obj) <= 5:
            return int(obj)
        raise ValueError(
            f"popularity_level must be an integer 1–5, got {obj!r}")

    if isinstance(obj, str):
        s = obj.strip()
        try:
            v = int(s)
        except ValueError as e:
            raise ValueError(
                f"popularity_level not parseable as int: {obj!r}") from e
        if 1 <= v <= 5:
            return v
        raise ValueError(f"popularity_level out of range after parse: {obj!r}")

    raise TypeError(
        f"popularity_level must be int-like (1–5), got {type(obj).__name__}: {obj!r}"
    )


def _call_openai(messages: list[dict[str, Any]], api_key: str, model: str) -> dict[str, Any]:
    if not api_key:
        raise RuntimeError(
            "Missing OpenAI API key. Set the OPENAI_API_KEY environment variable "
            "or pass --api-key when running this script."
        )

    client = OpenAI(api_key=api_key)

    last_err: Exception | None = None
    for attempt in range(1, 4):
        try:
            params = {
                "model": model,
                "messages": messages,
                "response_format": {"type": "json_object"},
            }

            # Only add temperature for models that support it
            if model not in NO_TEMPERATURE:
                params["temperature"] = 0

            resp = client.chat.completions.create(**params)

            content = resp.choices[0].message.content
            if content is None:
                raise ValueError("OpenAI response had empty content.")
            return json.loads(content)
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
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_file!r}")
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
) -> dict[str, Any]:
    """
    Call OpenAI with the popularity-level rubric prompt, image, and metadata.

    Metadata defaults to the dataset bundle from ``post_metadata(image_path)``.
    Override with ``metadata=...`` or ``metadata_file=...``.

    Returns:
        {"popularity_level": int}  # int in 1..5
    """
    key = api_key if api_key is not None else OPENAI_API_KEY
    model = os.getenv("OPENAI_MODEL", "gpt-4o")

    metadata_text = _resolve_metadata_text(
        image_path, metadata=metadata, metadata_file=metadata_file
    )
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        agent_expert_domain=expert_domain)

    if model in SUPPORTS_IMAGE_INPUT:
        data_url = _image_path_to_data_url(image_path)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": metadata_text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ]
    else:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": metadata_text,
            },
        ]

    raw = _call_openai(messages, key, model)
    level = _coerce_popularity_level(raw.get("popularity_level"))
    return {"popularity_level": level}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Predict popularity_level (1–5) from image + metadata via OpenAI."
    )
    parser.add_argument("image_path", nargs="?",
                        default="train/1@N18/1075.jpg")
    parser.add_argument("--expert-domain", default="Travel&Active&Sports")
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
        help="OpenAI API key (not stored; overrides OPENAI_API_KEY).",
    )
    args = parser.parse_args()

    api_key = args.api_key if args.api_key else OPENAI_API_KEY

    out = popularity_level_prediction_with_agent(
        args.image_path,
        args.expert_domain,
        metadata=args.metadata,
        metadata_file=args.metadata_file,
        api_key=api_key,
    )
    print(json.dumps(out, ensure_ascii=False))
