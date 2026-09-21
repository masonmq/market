#!/usr/bin/env python3
# Plot popularity-score distributions for selected training categories.


from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_CATEGORIES = [
    "Holiday & Celebrations",
    "Social & People",
    "Travel & Active & Sports",
    "Fashion",
    "Weather & Season",
    "Food",
]

# Dataset category names may omit spaces and may contain the original typo
CATEGORY_ALIASES = {
    "holiday&celebrations": "Holiday&Celebrations",
    "holidayandcelebrations": "Holiday&Celebrations",
    "social&people": "Social&People",
    "socialandpeople": "Social&People",
    "travel&active&sports": "Travel&Active&Sports",
    "travelandactiveandsports": "Travel&Active&Sports",
    "fashion": "Fashion",
    "weather&season": "Whether&Season",   # dataset spelling
    "weatherandseason": "Whether&Season",  # dataset spelling
    "whether&season": "Whether&Season",
    "whetherandseason": "Whether&Season",
    "food": "Food",
}


def default_data_dir() -> Path:
    """Prefer the same default folder style used by the original script."""
    here = Path(__file__).resolve().parent
    bundled = here / "train_allmetadata_json"
    return bundled if bundled.is_dir() else here


def normalize_category_name(name: str) -> str:
    """Normalize category names for matching user input to dataset labels."""
    cleaned = name.strip().replace(" ", "")
    cleaned = cleaned.replace(
        "and", "&") if "&" not in cleaned.lower() else cleaned
    key = cleaned.lower()
    return CATEGORY_ALIASES.get(key, cleaned)


def safe_filename(name: str) -> str:
    """Convert a display category into a safe file stem."""
    s = name.strip().replace("&", "and")
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()
    return s or "category"


def load_labels(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.float64)
    if values.ndim != 1:
        values = values.ravel()
    return values


def load_categories(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        data: Any = json.load(f)

    if not isinstance(data, list):
        raise TypeError(
            f"Expected train_category.json to contain a list, got {type(data).__name__}")

    categories: list[str] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise TypeError(
                f"Expected object at category row {i}, got {type(item).__name__}")
        if "Category" not in item:
            raise KeyError(f"Category row {i} is missing the 'Category' field")
        categories.append(str(item["Category"]))
    return categories


def plot_distribution(
    values: np.ndarray,
    *,
    display_category: str,
    dataset_category: str,
    save_path: Path | None = None,
    show: bool = False,
    bins: int = 80,
) -> None:
    if len(values) == 0:
        print(
            f"Skipped {display_category!r}: no matching rows for dataset category {dataset_category!r}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.hist(values, bins=bins, color="#4C72B0",
            edgecolor="white", linewidth=0.3, alpha=0.9)
    ax.set_xlabel("Popularity score / log views")
    ax.set_ylabel("Count")
    ax.set_title(f"Histogram (n = {len(values):,})")
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    ax.boxplot(values, vert=True, widths=0.5)
    ax.set_ylabel("Popularity score / log views")
    ax.set_title("Box plot")
    ax.set_xticklabels([display_category])
    ax.grid(True, axis="y", alpha=0.25)

    stats = (
        f"min={values.min():.2f}  max={values.max():.2f}  "
        f"mean={values.mean():.2f}  median={np.median(values):.2f}  "
        f"std={values.std():.2f}"
    )
    fig.suptitle(
        f"{display_category} log-views distribution\n{stats}", fontsize=11)
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "display_category",
        "dataset_category",
        "count",
        "min",
        "max",
        "mean",
        "median",
        "std",
        "save_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved summary: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir(),
        help="Directory containing train_label.txt and train_category.json.",
    )
    parser.add_argument(
        "--label-file",
        type=Path,
        default=None,
        help="Optional explicit path to train_label.txt.",
    )
    parser.add_argument(
        "--category-file",
        type=Path,
        default=None,
        help="Optional explicit path to train_category.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for saved category plots.",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=DEFAULT_CATEGORIES,
        help="Categories to plot. Names may use spaces around '&'.",
    )
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Optional path for a CSV summary of per-category statistics.",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    label_file = args.label_file or data_dir / "train_label.txt"
    category_file = args.category_file or data_dir / "train_category.json"
    output_dir = args.output_dir or data_dir / "category_label_distributions"
    summary_csv = args.summary_csv or output_dir / \
        "category_label_distribution_summary.csv"

    if not label_file.is_file():
        raise FileNotFoundError(f"Missing label file: {label_file}")
    if not category_file.is_file():
        raise FileNotFoundError(f"Missing category file: {category_file}")

    labels = load_labels(label_file)
    categories = np.array(load_categories(category_file), dtype=object)

    if len(labels) != len(categories):
        raise ValueError(
            f"Length mismatch: {len(labels):,} labels but {len(categories):,} category rows. "
            "These files must be row-aligned."
        )

    summary_rows: list[dict[str, Any]] = []
    for display_category in args.categories:
        dataset_category = normalize_category_name(display_category)
        mask = categories == dataset_category
        values = labels[mask]

        save_path = output_dir / \
            f"{safe_filename(display_category)}_log_views_distribution.png"
        plot_distribution(
            values,
            display_category=display_category,
            dataset_category=dataset_category,
            save_path=save_path,
            show=args.show,
            bins=args.bins,
        )

        if len(values) > 0:
            summary_rows.append(
                {
                    "display_category": display_category,
                    "dataset_category": dataset_category,
                    "count": len(values),
                    "min": f"{values.min():.6g}",
                    "max": f"{values.max():.6g}",
                    "mean": f"{values.mean():.6g}",
                    "median": f"{np.median(values):.6g}",
                    "std": f"{values.std():.6g}",
                    "save_path": str(save_path),
                }
            )

    if summary_rows:
        write_summary_csv(summary_rows, summary_csv)

    print(f"Loaded {len(labels):,} labels from {label_file}")
    print(f"Loaded {len(categories):,} categories from {category_file}")


if __name__ == "__main__":
    main()
