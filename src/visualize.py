"""
Visualization and Benchmark Plotting for FieldAI.

Generates comparative bar charts:
- OCR + Regex vs. BERT (Text Only) vs. LayoutLM (Text + 2D Layout)
- Field-wise breakdown: Company, Date, Total, and Overall Macro F1
- Annotated ΔF1 = F1(LayoutLM) - F1(BERT)

Supports Matplotlib (if installed) and pure-Python standalone SVG rendering
so visual artifacts can be generated in any environment without external packages.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List


def load_json_metrics(file_path: str) -> Optional[Dict[str, Any]]:
    """Safely load JSON metrics dictionary from disk."""
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load {file_path}: {e}")
        return None


def generate_svg_chart(
    data: Dict[str, Dict[str, float]],
    output_path: str,
    fields: Optional[List[str]] = None,
) -> None:
    """
    Generate an SVG comparative bar chart without any external dependencies.
    
    data format:
    {
        "OCR + Regex": {"company": 0.45, "date": 0.62, "total": 0.53, "macro_avg": 0.533},
        "BERT (Text)": {"company": 0.82, "date": 0.91, "total": 0.76, "macro_avg": 0.830},
        "LayoutLM": {"company": 0.93, "date": 0.94, "total": 0.91, "macro_avg": 0.927},
    }
    """
    if fields is None:
        fields = ["company", "date", "total", "macro_avg"]

    field_labels = {
        "company": "Company",
        "date": "Date",
        "total": "Total",
        "macro_avg": "Macro Avg",
    }

    models = list(data.keys())
    colors = ["#4A90E2", "#F5A623", "#7ED321", "#BD10E0"]

    # Chart dimensions
    width = 800
    height = 500
    margin = {"top": 80, "right": 50, "bottom": 80, "left": 70}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]

    n_groups = len(fields)
    n_bars = len(models)
    group_width = plot_w / n_groups
    bar_width = min(40, (group_width * 0.75) / max(1, n_bars))
    gap = 4

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" style="background-color: #FFFFFF; font-family: -apple-system, BlinkMacSystemFont, sans-serif;">',
        f'  <!-- Title -->',
        f'  <text x="{width / 2}" y="35" text-anchor="middle" font-size="20" font-weight="bold" fill="#1A1A1A">Field-Wise F1 Benchmark: Text vs. Layout</text>',
        f'  <text x="{width / 2}" y="55" text-anchor="middle" font-size="12" fill="#666666">ICDAR 2019 SROIE Dataset — Exact Entity Match Evaluation</text>',
    ]

    # Y-axis gridlines and labels (0% to 100%)
    for pct in range(0, 101, 20):
        y = margin["top"] + plot_h - (pct / 100.0) * plot_h
        svg_parts.append(
            f'  <line x1="{margin["left"]}" y1="{y}" x2="{width - margin["right"]}" y2="{y}" stroke="#E0E0E0" stroke-width="1" stroke-dasharray="3,3" />'
        )
        svg_parts.append(
            f'  <text x="{margin["left"] - 10}" y="{y + 4}" text-anchor="end" font-size="12" fill="#777777">{pct}%</text>'
        )

    # Bars
    for g_idx, field in enumerate(fields):
        g_center = margin["left"] + g_idx * group_width + group_width / 2
        start_x = g_center - (n_bars * bar_width + (n_bars - 1) * gap) / 2

        for m_idx, model in enumerate(models):
            val = data[model].get(field, 0.0)
            bar_h = val * plot_h
            bx = start_x + m_idx * (bar_width + gap)
            by = margin["top"] + plot_h - bar_h
            color = colors[m_idx % len(colors)]

            # Draw bar
            svg_parts.append(
                f'  <rect x="{bx:.1f}" y="{by:.1f}" width="{bar_width:.1f}" height="{bar_h:.1f}" fill="{color}" rx="4" />'
            )
            # Label value on top of bar
            svg_parts.append(
                f'  <text x="{bx + bar_width / 2:.1f}" y="{by - 6:.1f}" text-anchor="middle" font-size="11" font-weight="600" fill="#333333">{val * 100:.1f}%</text>'
            )

        # Group label on X-axis
        svg_parts.append(
            f'  <text x="{g_center:.1f}" y="{height - margin["bottom"] + 25}" text-anchor="middle" font-size="14" font-weight="bold" fill="#222222">{field_labels.get(field, field)}</text>'
        )

    # Legend
    legend_y = height - 25
    legend_x_start = margin["left"] + 60
    for m_idx, model in enumerate(models):
        lx = legend_x_start + m_idx * 200
        color = colors[m_idx % len(colors)]
        svg_parts.append(
            f'  <rect x="{lx}" y="{legend_y - 12}" width="16" height="16" fill="{color}" rx="3" />'
        )
        svg_parts.append(
            f'  <text x="{lx + 24}" y="{legend_y}" font-size="12" font-weight="500" fill="#333333">{model}</text>'
        )

    svg_parts.append("</svg>")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_parts))
    print(f"Chart saved to {output_path}")


def generate_matplotlib_plot(
    data: Dict[str, Dict[str, float]],
    output_path: str,
    fields: Optional[List[str]] = None,
) -> bool:
    """Generate PNG / PDF plot using Matplotlib if available."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return False

    if fields is None:
        fields = ["company", "date", "total", "macro_avg"]

    field_labels = ["Company", "Date", "Total", "Macro Avg"]
    models = list(data.keys())

    x = np.arange(len(fields))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, model in enumerate(models):
        scores = [data[model].get(f, 0.0) * 100 for f in fields]
        rects = ax.bar(x + i * width, scores, width, label=model)
        for rect in rects:
            h = rect.get_height()
            ax.annotate(
                f"{h:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_ylabel("F1 Score (%)", fontsize=12)
    ax.set_title("Receipt Information Extraction: Layout vs. Text Benchmark", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(field_labels, fontsize=12)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Matplotlib figure saved to {output_path}")
    return True


if __name__ == "__main__":
    # Test generation with sample benchmark data
    sample_data = {
        "OCR + Regex": {"company": 0.421, "date": 0.654, "total": 0.542, "macro_avg": 0.539},
        "BERT (Text)": {"company": 0.812, "date": 0.923, "total": 0.771, "macro_avg": 0.835},
        "LayoutLM (Layout)": {"company": 0.941, "date": 0.952, "total": 0.928, "macro_avg": 0.940},
    }
    generate_svg_chart(sample_data, "results/benchmark_preview.svg")
