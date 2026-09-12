"""
Benchmark Logging and Results Table Formatting for Ilādṛṣṭi.
Maintains synchronized JSON, CSV, and Markdown results tables for hackathon reporting.
"""

import csv
import json
from pathlib import Path
from typing import Any, Optional


def log_benchmark_result(
    experiment_name: str,
    architecture: str,
    backbone: str,
    in_channels: int,
    loss_function: str,
    val_dice: float,
    val_iou: float,
    test_loss: float,
    test_dice: float,
    test_iou: float,
    seed: int = 42,
    notes: str = "",
    output_dir: str = "outputs",
) -> dict[str, Any]:
    """
    Safely log a benchmark run into outputs/benchmark_results.csv,
    outputs/benchmark_results.json, and outputs/benchmark_results.md.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    entry = {
        "experiment_name": experiment_name,
        "architecture": architecture,
        "backbone": backbone,
        "in_channels": in_channels,
        "loss_function": loss_function,
        "val_dice": round(val_dice, 4),
        "val_iou": round(val_iou, 4),
        "test_loss": round(test_loss, 4),
        "test_dice": round(test_dice, 4),
        "test_iou": round(test_iou, 4),
        "seed": seed,
        "notes": notes,
    }

    # 1. Update JSON list
    json_file = out_path / "benchmark_results.json"
    all_entries = []
    if json_file.exists():
        try:
            with open(json_file, "r") as f:
                all_entries = json.load(f)
        except json.JSONDecodeError:
            all_entries = []

    # Replace if duplicate experiment_name, else append
    existing_idx = next(
        (i for i, e in enumerate(all_entries) if e.get("experiment_name") == experiment_name),
        None,
    )
    if existing_idx is not None:
        all_entries[existing_idx] = entry
    else:
        all_entries.append(entry)

    with open(json_file, "w") as f:
        json.dump(all_entries, f, indent=2)

    # 2. Update CSV table
    csv_file = out_path / "benchmark_results.csv"
    fieldnames = list(entry.keys())
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_entries)

    # 3. Update Markdown table
    md_file = out_path / "benchmark_results.md"
    md_content = generate_markdown_table(all_entries)
    with open(md_file, "w") as f:
        f.write(md_content)

    print(f"[Ilādṛṣṭi] Benchmark metrics safely logged to:")
    print(f"            - {json_file}")
    print(f"            - {csv_file}")
    print(f"            - {md_file}")

    return entry


def generate_markdown_table(entries: list[dict[str, Any]]) -> str:
    """Generate GitHub-flavored markdown table for final reporting."""
    lines = [
        "# Ilādṛṣṭi: Benchmark Results & Ablation Study",
        "Track 6 (AI for Science & Society) — Deep Learning Hackathon, Amrita Vishwa Vidyapeetham\n",
        "| Experiment | Architecture | Backbone | Bands | Loss Function | Val Dice | Val IoU | Test Loss | Test Dice | Test IoU | Seed | Notes |",
        "| :--- | :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]
    for e in entries:
        bands_str = "4 (RGB+NIR)" if e["in_channels"] == 4 else "3 (RGB)"
        lines.append(
            f"| **{e['experiment_name']}** | {e['architecture']} | {e['backbone']} | {bands_str} | "
            f"{e['loss_function']} | {e['val_dice']:.4f} | {e['val_iou']:.4f} | {e['test_loss']:.4f} | "
            f"**{e['test_dice']:.4f}** | **{e['test_iou']:.4f}** | {e['seed']} | {e['notes']} |"
        )
    lines.append("")
    return "\n".join(lines)
