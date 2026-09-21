#!/usr/bin/env python3
"""Plot distribution of popularity scores in train_label.txt."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_labels(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.float64)
    if values.ndim != 1:
        values = values.ravel()
    return values


def plot_distribution(
    values: np.ndarray,
    *,
    save_path: Path | None = None,
    show: bool = False,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.hist(values, bins=80, color="#4C72B0", edgecolor="white", linewidth=0.3, alpha=0.9)
    ax.set_xlabel("Popularity score")
    ax.set_ylabel("Count")
    ax.set_title(f"Histogram (n = {len(values):,})")
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    ax.boxplot(values, vert=True, widths=0.5)
    ax.set_ylabel("Popularity score")
    ax.set_title("Box plot")
    ax.set_xticklabels(["train_label"])
    ax.grid(True, axis="y", alpha=0.25)

    stats = (
        f"min={values.min():.2f}  max={values.max():.2f}  "
        f"mean={values.mean():.2f}  median={np.median(values):.2f}  "
        f"std={values.std():.2f}"
    )
    fig.suptitle(f"train_label.txt distribution\n{stats}", fontsize=11)
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label-file",
        type=Path,
        default=Path(__file__).resolve().parent
        / "train_allmetadata_json"
        / "train_label.txt",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=Path(__file__).resolve().parent
        / "train_allmetadata_json"
        / "train_label_distribution.png",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    values = load_labels(args.label_file)
    plot_distribution(values, save_path=args.save, show=args.show)
    print(f"Loaded {len(values):,} values from {args.label_file}")


if __name__ == "__main__":
    main()
