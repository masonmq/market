#!/usr/bin/env python3
"""
Look up one ICIP training post 

"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

# image path input examples: 
# ICIP/train_imgs/train_imgs/49108855442.jpg
# 49108855442.jpg
# 49108855442

DEFAULT_IMAGE = "49108855442"

CSV_FILENAMES = {
    "headers": "headers_TRAIN.csv",
    "image_info": "img_info_TRAIN.csv",
    "popularity": "popularity_TRAIN.csv",
    "users": "users_TRAIN.csv",
}

INT_FIELDS = {
    "Size",
    "NumSets",
    "NumGroups",
    "Contacts",
    "PhotoCount",
    "GroupsCount",
}

FLOAT_FIELDS = {
    "AvgGroupsMemb",
    "AvgGroupPhotos",
    "Latitude",
    "Longitude",
    "MeanViews",
    "GroupsAvgMembers",
    "GroupsAvgPictures",
}

BOOL_FIELDS = {"Ispro", "HasStats"}

def _default_data_dir() -> Path:
    return Path(__file__).resolve().parent / "ICIP"


def required_paths(data_dir: Path) -> dict[str, Path]:
    return {name: data_dir / filename for name, filename in CSV_FILENAMES.items()}


def verify_required_files(paths: dict[str, Path]) -> None:
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        joined = "\n  ".join(missing)
        raise FileNotFoundError(f"Missing required file(s):\n  {joined}")


def clean_target(value: str) -> str:
    return value.strip().strip('"').strip("'")


def extract_flickr_id(value: str) -> str:

    value = clean_target(value)
    if not value:
        raise ValueError("The lookup value is empty.")

    if value.isdigit():
        return value

    parsed = urlparse(value)
    candidate = Path(parsed.path if parsed.scheme else value.replace("\\", "/")).name
    number_groups = re.findall(r"\d+", candidate)

    if not number_groups:
        number_groups = re.findall(r"\d+", value)

    if not number_groups:
        raise ValueError(
            f"Could not extract a numeric FlickrId from {value!r}. "
            "Pass the FlickrId directly or use --row."
        )

    return max(number_groups, key=len)


def iter_csv(path: Path) -> Iterable[dict[str, str]]:
    """Yield CSV rows and attach their zero-based source-row positions."""
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header row: {path}")

        for source_row, row in enumerate(reader):
            cleaned = {key: value for key, value in row.items() if key not in (None, "")}
            cleaned["__source_row__"] = str(source_row)
            if row.get("") not in (None, ""):
                cleaned["__original_index__"] = row[""]
            yield cleaned


def find_matches(path: Path, key: str, value: str) -> list[dict[str, str]]:
    return [row for row in iter_csv(path) if row.get(key) == value]


def row_at(path: Path, row_index: int) -> dict[str, str]:
    if row_index < 0:
        raise IndexError("Row index must be non-negative.")

    for index, row in enumerate(iter_csv(path)):
        if index == row_index:
            return row

    raise IndexError(f"{path.name} has no data row at zero-based index {row_index}.")


def to_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(float(value))


def to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)

# covert unix timestamp to datetime string
def unix_timestamp(value: str | None) -> dict[str, Any] | None:
    if value is None or value == "":
        return None

    timestamp = int(float(value))
    utc_text = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )
    return {"unix": timestamp, "utc": utc_text}


def convert_row(row: dict[str, str]) -> dict[str, Any]:
    """Convert known numeric, boolean, tag, and timestamp fields."""
    converted: dict[str, Any] = {}

    for key, value in row.items():
        if key.startswith("__"):
            continue
        if key in {"FlickrId", "UserId", "URL", "Username"}:
            converted[key] = value
        elif key in {"DatePosted", "DateTaken", "DateCrawl"}:
            converted[key] = unix_timestamp(value)
        elif key in BOOL_FIELDS:
            converted[key] = None if value == "" else value == "1"
        elif key in INT_FIELDS:
            converted[key] = to_int(value)
        elif key in FLOAT_FIELDS or re.fullmatch(r"Day\d{2}", key):
            converted[key] = to_float(value)
        elif key == "Tags":
            if value == "":
                converted[key] = []
            else:
                try:
                    parsed = json.loads(value)
                    converted[key] = parsed if isinstance(parsed, list) else value
                except json.JSONDecodeError:
                    converted[key] = value
        else:
            converted[key] = value if value != "" else None

    return converted


def select_latest_header(matches: list[dict[str, str]]) -> dict[str, str]:
    """Choose the header record with the latest crawl timestamp."""
    return max(matches, key=lambda row: to_int(row.get("DateCrawl")) or -1)


def select_latest_user(matches: list[dict[str, str]]) -> dict[str, str]:
    """
    Choose the last matching user snapshot in users_TRAIN.csv.
    """
    return matches[-1]


def candidate_image_names(flickr_id: str, url: str | None) -> list[str]:
    candidates = [f"train_imgs/{flickr_id}.jpg"]
    if url:
        url_name = Path(urlparse(url).path).name
        if url_name:
            url_candidate = f"train_imgs/{url_name}"
            if url_candidate not in candidates:
                candidates.append(url_candidate)
    return candidates


def load_post_bundle(
    lookup: str | None = None,
    data_dir: Path | None = None,
    row_index: int | None = None,
) -> dict[str, Any]:
    """Load all available metadata for one ICIP training post."""
    if data_dir is None:
        data_dir = _default_data_dir()

    paths = required_paths(data_dir)
    verify_required_files(paths)


    if lookup is None:
        lookup = DEFAULT_IMAGE
        flickr_id = extract_flickr_id(lookup)
        lookup_input = lookup

    header_matches = find_matches(paths["headers"], "FlickrId", flickr_id)
    image_info_matches = find_matches(paths["image_info"], "FlickrId", flickr_id)
    popularity_matches = find_matches(paths["popularity"], "FlickrId", flickr_id)

    if not image_info_matches:
        raise LookupError(
            f"FlickrId {flickr_id!r} is not present in img_info_TRAIN.csv. "
            "Only the 20,337 training-image records can be looked up."
        )
    if not popularity_matches:
        raise LookupError(
            f"FlickrId {flickr_id!r} is not present in popularity_TRAIN.csv."
        )
    if not header_matches:
        raise LookupError(f"FlickrId {flickr_id!r} is not present in headers_TRAIN.csv.")

    selected_header_raw = select_latest_header(header_matches)
    image_info_raw = image_info_matches[0]
    popularity_raw = popularity_matches[0]

    user_id = selected_header_raw.get("UserId", "")
    user_matches = find_matches(paths["users"], "UserId", user_id)
    selected_user_raw = select_latest_user(user_matches) if user_matches else None

    selected_header = convert_row(selected_header_raw)
    image_info = convert_row(image_info_raw)
    popularity = convert_row(popularity_raw)
    selected_user = convert_row(selected_user_raw) if selected_user_raw else None

    trajectory = {
        key: popularity[key]
        for key in sorted(popularity)
        if re.fullmatch(r"Day\d{2}", key)
    }

    return {
        "lookup": {
            "image_id": lookup_input,
            "flickr_id": flickr_id,
            "canonical_img_info_row": row_index
            if row_index is not None
            else to_int(image_info_raw.get("__source_row__")),
            "candidate_local_image_paths": candidate_image_names(
                flickr_id, selected_header_raw.get("URL")
            ),
        },
        "image_header": selected_header,
        "image_information": image_info, 
        "popularity_trajectory": trajectory,
        "user_profile": selected_user,
        # "source_rows": {
        #     "headers_TRAIN.csv": to_int(selected_header_raw.get("__source_row__")),
        #     "img_info_TRAIN.csv": to_int(image_info_raw.get("__source_row__")),
        #     "popularity_TRAIN.csv": to_int(popularity_raw.get("__source_row__")),
        #     "users_TRAIN.csv": (
        #         to_int(selected_user_raw.get("__source_row__"))
        #         if selected_user_raw is not None
        #         else None
        #     ),
        # },
    }


def print_record(title: str, value: Any) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(value, indent=2, ensure_ascii=False))


def print_bundle(bundle: dict[str, Any]) -> None:
    print_record("Lookup image", bundle["lookup"])
    print_record("Image header", bundle["image_header"])
    print_record("Image information", bundle["image_information"])
    print_record("Popularity trajectory", bundle["popularity_trajectory"])
    print_record("User profile", bundle["user_profile"])


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Look up an ICIP training post by FlickrId, image path/filename, "
            "Flickr URL, or img_info row index."
        )
    )
    parser.add_argument(
        "lookup",
        nargs="?",
        default=None,
        help=(
            "FlickrId, image path/filename, or Flickr URL. "
            f"Default: {DEFAULT_IMAGE }"
        ),
    )
    parser.add_argument(
        "--row",
        type=int,
        default=None,
        help="Zero-based data-row index in img_info_TRAIN.csv.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_default_data_dir(),
        help="Directory containing the four *_TRAIN.csv files.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the entire bundle as one JSON object.",
    )
    args = parser.parse_args()

    if args.row is not None and args.lookup is not None:
        parser.error("Use either a lookup value or --row, not both.")

    try:
        bundle = load_post_bundle(
            lookup=args.lookup,
            data_dir=args.data_dir,
            row_index=args.row,
        )
    except (FileNotFoundError, LookupError, ValueError, IndexError) as error:
        print(error, file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(bundle, indent=2, ensure_ascii=False))
    else:
        print_bundle(bundle)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
