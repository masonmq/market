#!/usr/bin/env python3
"""
OpenAI-backed agent that predicts popularity level probabilities (1–5)
given a post's image + bundled metadata.
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

OPENAI_API_KEY = ""

SYSTEM_PROMPT_TEMPLATE = """You are a senior social media analyst with expertise in **{agent_expert_domain}**, and you have years of experience in audience behavior, feed algorithms, and cross-platform engagement patterns.

You evaluate the post strictly based on the provided image and metadata. 
The metadata includes:
===Category===: 
Category: the first category of the post.(11 classes)
Subcategory: there are 77 classes in 2nd level category.
Concept: there are 668 different description
===text===:
Title: the tile of the post defined by the user.
Mediatype: the type of the attached media file, including 'photo' and 'video'.
Alltags: the customized tags from users.
===Temporal-spatial===:
Postdate: the publish timestamp of the post. It can be converted to Datetime by following python code:
```python
import time
timestamp = 1457068974
timeArray = time.localtime(timestamp)
datetime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
```
Latitude: the latitude whose valid range is -90 to 90. Anything more than 6 decimal places will be truncated.
Longitude: the longitude whose valid range is -180 to 180. Anything more than 6 decimal places will be truncated.
Geoaccuracy: recorded accuracy level of the location information. World level is 1, Country is ~3, Region ~6, City ~11, Street ~16. The current range is 1-16. Defaults to 16 if not specified.
===User profile===:
Photo_firstdate: the date of the first photo uploaded by the user.
Photo_count: the number of posted photo by the user.
Ispro: is the user belong to pro member.
Photo_firstdatetaken: the date of the first photo taken by the user.
Timezone_offset: the time zone of the user.
User description: the feature used to describe the user data.
Location description: the feature used to describe the user location.
===Additional information===:
Pathalias: the path alias provided by the user.
Ispublic: indicates that the post is authenticated with 'read' permissions.
Mediastatus: indicates that the attached media is ready to access by others.

Your task is to predict the **popularity_level_probabilities** of a post relative to typical user-generated photo content on a large social platform, using only the provided image and metadata.

Do NOT claim or assume real view counts or external engagement data.

### Scoring rubric (1–5):

* 1 = Very low expected engagement (weak visual hook, cluttered or generic, poor audience fit)
* 2 = Below average engagement (limited shareability or niche appeal)
* 3 = Average content (balanced strengths and weaknesses)
* 4 = Above average engagement (strong hook, clear theme, good audience fit)
* 5 = High viral or standout potential (highly memorable or highly shareable content)

### Output requirements:

* You must assign a probability distribution over levels 1–5.
* Each value must be:

  * non-negative
  * a decimal (e.g., 0.15)
  * sum exactly to 1.0

### Output format:

Return ONLY valid JSON (no markdown, no explanation):

```json
{{
  "popularity_level_probabilities": {{
    "1": float,
    "2": float,
    "3": float,
    "4": float,
    "5": float
  }}
}}
```"""


def _image_path_to_data_url(image_path: str) -> str:
    p = Path(image_path)
    if not p.is_file():
        # Common case: caller passes a relative dataset path like "train/...".
        # Resolve relative to this script's directory (the SMPD folder).
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


def _coerce_probs(obj: Any) -> dict[str, float]:
    if not isinstance(obj, dict):
        raise TypeError(
            f"Expected popularity_level_probabilities to be an object, got {type(obj).__name__}"
        )

    keys = ["1", "2", "3", "4", "5"]
    probs: dict[str, float] = {}
    for k in keys:
        if k not in obj:
            raise KeyError(f"Missing probability key {k!r}")
        v = obj[k]
        try:
            fv = float(v)
        except (TypeError, ValueError) as e:
            raise TypeError(f"Probability for key {k!r} must be numeric, got {v!r}") from e
        if fv < 0:
            raise ValueError(f"Probability for key {k!r} must be non-negative, got {fv}")
        probs[k] = fv

    total = sum(probs.values())
    if total <= 0:
        raise ValueError("Probabilities sum to 0; cannot normalize.")

    # Normalize and adjust the last bucket to ensure exact sum==1.0 after rounding.
    norm = {k: probs[k] / total for k in keys}
    rounded = {k: round(norm[k], 6) for k in keys}
    rounded["5"] = round(1.0 - sum(rounded[k] for k in keys[:-1]), 6)
    if rounded["5"] < 0:
        # Fall back to unrounded normalization if rounding created a negative remainder.
        rounded = norm
        rounded["5"] = 1.0 - sum(rounded[k] for k in keys[:-1])

    # Final sanity: ensure numeric and sum exactly 1.0 (within float representation).
    final_total = sum(float(rounded[k]) for k in keys)
    if abs(final_total - 1.0) > 1e-6:
        # Last-resort exact fix (keeps distribution valid).
        rounded["5"] = float(rounded["5"]) + (1.0 - final_total)

    return {k: float(rounded[k]) for k in keys}


def _call_openai(messages: list[dict[str, Any]]) -> dict[str, Any]:
    api_key = OPENAI_API_KEY
    if not api_key:
        raise RuntimeError(
            "Missing OpenAI API key. Set llm_agent.OPENAI_API_KEY in Python "
            "or pass --api-key when running this script."
        )

    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    last_err: Exception | None = None
    for attempt in range(1, 4):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.2,
            )

            content = resp.choices[0].message.content
            if content is None:
                raise ValueError("OpenAI response had empty content.")
            return json.loads(content)
        except Exception as e:  # network/model/JSON failures
            last_err = e
            if attempt >= 3:
                break
            # Exponential backoff with jitter.
            time.sleep(min(20.0, (2 ** (attempt - 1)) + random.random()))

    assert last_err is not None
    raise last_err


def llm_agent(
    image_path: str,
    expert_domain: str,
    *,
    market_prices: dict[str, float] | None = None,
) -> dict:
    """
    Call OpenAI with:
      - system: SYSTEM_PROMPT (domain filled)
      - user: metadata text + image (vision input)

    Returns:
      {
        "popularity_level_probabilities": {
          "1": float, "2": float, "3": float, "4": float, "5": float
        }
      }
    """
    post_data = json.loads(post_metadata(image_path))
    if "metadata" not in post_data or not isinstance(post_data["metadata"], str):
        raise KeyError("post_metadata output missing string field 'metadata'")

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(agent_expert_domain=expert_domain)
    if market_prices is not None:
        # Keep key order stable for readability.
        ordered = {k: float(market_prices[k]) for k in ["1", "2", "3", "4", "5"] if k in market_prices}
        system_prompt += (
            "\n\nThe current market estimate of popularity_level_probabilities is "
            + json.dumps(ordered, ensure_ascii=False)
            + "."
        )
    data_url = _image_path_to_data_url(image_path)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": post_data["metadata"]},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]

    raw = _call_openai(messages)
    probs_obj = raw.get("popularity_level_probabilities")
    probs = _coerce_probs(probs_obj)
    return {"popularity_level_probabilities": probs}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run OpenAI LLM agent on one post.")
    parser.add_argument("image_path", nargs="?", default="train/1@N18/1075.jpg")
    parser.add_argument("--expert-domain", default="Electronics")
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenAI API key (passed directly to the client; not stored).",
    )
    args = parser.parse_args()

    if args.api_key:
        OPENAI_API_KEY = args.api_key

    out = llm_agent(args.image_path, args.expert_domain)
    print(json.dumps(out, ensure_ascii=False))

