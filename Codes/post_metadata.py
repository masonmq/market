#!/usr/bin/env python3
"""
Build a JSON string: metadata (lookup_train_post-style printout without log-views
block), label (popularity_log_views), and main category string.
"""

from __future__ import annotations

import json
import time
from io import StringIO
from pathlib import Path
from typing import Any


def _default_data_dir() -> Path:
    return Path(__file__).resolve().parent / "train_allmetadata_json"


def normalize_image_path(s: str) -> str:
    s = s.strip().strip('"').strip("'")
    s = s.replace("\\", "/")
    return s.lstrip("/")


def find_row_index(img_list_path: Path, target: str) -> int:
    target = normalize_image_path(target)
    with img_list_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if normalize_image_path(line) == target:
                return i
    raise LookupError(f"No row with image path: {target!r}")


def read_label_line(label_path: Path, row: int) -> str:
    with label_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == row:
                return line.strip()
    raise IndexError(f"Label file has no line at index {row}")


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def nth_json_array_item(data: list[Any], row: int) -> dict[str, Any]:
    if row < 0 or row >= len(data):
        raise IndexError(f"JSON array length {len(data)} has no index {row}")
    item = data[row]
    if not isinstance(item, dict):
        raise TypeError(f"Expected object at index {row}, got {type(item).__name__}")
    return item


def nth_user_data_row(columnar: dict[str, Any], row: int) -> dict[str, Any]:
    key = str(row)
    out: dict[str, Any] = {}
    for col, series in columnar.items():
        if not isinstance(series, dict):
            out[col] = series
            continue
        if key not in series:
            raise KeyError(f"user_data column {col!r} missing key {key!r}")
        out[col] = series[key]
    return out


def format_postdate(ts: Any) -> str | None:
    if ts is None or ts == "":
        return None
    try:
        t = int(str(ts))
    except (TypeError, ValueError):
        return str(ts)
    time_array = time.localtime(t)
    return time.strftime("%Y-%m-%d %H:%M:%S", time_array)


def _load_post_bundle(image_path: str, data_dir: Path | None = None) -> dict[str, Any]:
    if data_dir is None:
        data_dir = _default_data_dir()

    img_list = data_dir / "train_img_filepath.txt"
    label_txt = data_dir / "train_label.txt"
    category_json = data_dir / "train_category.json"
    text_json = data_dir / "train_text.json"
    temporal_json = data_dir / "train_temporalspatial_information.json"
    user_json = data_dir / "train_user_data.json"
    extra_json = data_dir / "train_additional_information.json"

    for p in (
        img_list,
        label_txt,
        category_json,
        text_json,
        temporal_json,
        user_json,
        extra_json,
    ):
        if not p.is_file():
            raise FileNotFoundError(f"Missing required file: {p}")

    row = find_row_index(img_list, image_path)
    norm_path = normalize_image_path(image_path)
    label = read_label_line(label_txt, row)

    category = load_json(category_json)
    text = load_json(text_json)
    temporal = load_json(temporal_json)
    user_columnar = load_json(user_json)
    extra = load_json(extra_json)

    cat_obj = nth_json_array_item(category, row)
    text_obj = nth_json_array_item(text, row)
    temporal_obj = nth_json_array_item(temporal, row)
    user_obj = nth_user_data_row(user_columnar, row)
    extra_obj = nth_json_array_item(extra, row)

    temporal_display = dict(temporal_obj)
    pd = temporal_display.get("Postdate")
    formatted = format_postdate(pd)
    if formatted is not None:
        temporal_display["Postdate_datetime"] = formatted

    return {
        "image_path": norm_path,
        "row_index": row,
        "popularity_log_views": float(label),
        "category": cat_obj,
        "text": text_obj,
        "temporal_spatial": temporal_display,
        "user_profile": user_obj,
        "additional_information": extra_obj,
    }


def _print_record(buf: StringIO, title: str, obj: dict[str, Any]) -> None:
    print(f"\n=== {title} ===", file=buf)
    print(json.dumps(obj, indent=2, ensure_ascii=False), file=buf)


def post_metadata(image_path: str) -> str:
    """
    Return a JSON string:
      {"metadata": "<lookup_train_post-style stdout minus log-views section>",
       "label": <popularity_log_views float>,
       "category": "<main Category string>"}

    image_path: e.g. "train/1@N18/1075.jpg" (same as train_img_filepath.txt rows).
    """
    bundle = _load_post_bundle(image_path)

    norm_path = bundle["image_path"]
    row = bundle["row_index"]
    label = bundle["popularity_log_views"]
    cat_obj = bundle["category"]
    text_obj = bundle["text"]
    temporal_display = bundle["temporal_spatial"]
    user_obj = bundle["user_profile"]
    extra_obj = bundle["additional_information"]

    category_str = cat_obj.get("Category")
    if not isinstance(category_str, str):
        raise TypeError(
            f'Expected string "Category" in category row, got {type(category_str).__name__}'
        )

    buf = StringIO()
    print("=== Image path (row index) ===", file=buf)
    print(json.dumps({"image_path": norm_path, "row_index": row}, indent=2, ensure_ascii=False), file=buf)
    _print_record(buf, "Category", cat_obj)
    _print_record(buf, "Text", text_obj)
    _print_record(buf, "Temporal-spatial", temporal_display)
    _print_record(buf, "User profile", user_obj)
    _print_record(buf, "Additional information", extra_obj)

    metadata_text = buf.getvalue()
    payload: dict[str, Any] = {
        "metadata": metadata_text,
        "label": label,
        "category": category_str,
    }
    return json.dumps(payload, ensure_ascii=False)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Print JSON: metadata, label, category.")
    p.add_argument("image_path", nargs="?", default="train/1@N18/1075.jpg")
    args = p.parse_args()
    print(post_metadata(args.image_path))
