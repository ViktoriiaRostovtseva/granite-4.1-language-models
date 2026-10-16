"""Create side-by-side metric boxplots from pipeline result JSON files.

Usage:
    python3 plot_metrics.py

Requirements:
    pip install matplotlib
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_INPUTS = {
    "Original": "results_orig_truth.json",
    "Moderated": "results_mod_truth.json",
    "Corrupted": "results_cor_truth.json",
    "Very Corrupted": "results_very_cor_truth.json",
}
METRICS = {
    "strong_match_rate": "Strong match rate",
    "avg_rouge_l": "Average ROUGE-L",
    "none_rate": "None rate",
}
COLORS = ["#3b82d4", "#7c5cd8", "#57606a", "#e05c5c"]


def load_metric_values(path: Path, metric: str) -> list[float]:
    """Return one metric value per evaluated document in a result file."""
    with path.open() as file:
        results = json.load(file)

    return [
        document["metrics"][metric]
        for document in results["documents"]
        if "metrics" in document
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Create boxplots comparing extraction metrics by sample type."
    )
    parser.add_argument(
        "--original",
        default=DEFAULT_INPUTS["Original"],
        help="Results JSON for original samples.",
    )
    parser.add_argument(
        "--moderated",
        default=DEFAULT_INPUTS["Moderated"],
        help="Results JSON for moderated samples.",
    )
    parser.add_argument(
        "--corrupted",
        default=DEFAULT_INPUTS["Corrupted"],
        help="Results JSON for corrupted samples.",
    )
    parser.add_argument(
        "--cor",
        default=DEFAULT_INPUTS["Very Corrupted"],
        help="Results JSON for very corrupted samples.",
    )
    parser.add_argument(
        "--output",
        default="figures/metric_boxplots.png",
        help="Output image path.",
    )
    args = parser.parse_args()

    input_paths = {
        "Original": Path(args.original),
        "Moderated": Path(args.moderated),
        "Corrupted": Path(args.corrupted),
        "Very Corrupted": Path(args.cor),
    }
    metric_values = {
        metric: [load_metric_values(path, metric) for path in input_paths.values()]
        for metric in METRICS
    }

    figure, axes = plt.subplots(1, len(METRICS), figsize=(18, 5), sharey=True)
    for axis, (metric, title) in zip(axes, METRICS.items()):
        boxes = axis.boxplot(
            metric_values[metric],
            tick_labels=input_paths.keys(),
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 1.5},
        )
        for box, color in zip(boxes["boxes"], COLORS):
            box.set_facecolor(color)

        axis.set_title(title)
        axis.set_ylim(-0.05, 1.05)
        axis.set_ylabel("Score")
        axis.grid(axis="y", alpha=0.3)

    figure.suptitle("Extraction Metrics by Sample Type")
    figure.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved boxplots to {output_path}")


if __name__ == "__main__":
    main()
