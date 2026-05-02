"""Visualize Crazyflow benchmark variants."""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

logger = logging.getLogger(__name__)

RESULTS_PATH = Path(__file__).parents[1] / "results/performance/crazyflow.json"
PLOTS_DIR = Path(__file__).parents[1] / "plots/performance"
PIPELINE_LABELS = {"baseline": "Baseline", "uwb_imu_ekf": "With UWB/IMU/EKF"}
COLORS = {
    ("baseline", "cpu"): "#1f77b4",
    ("uwb_imu_ekf", "cpu"): "#ff7f0e",
    ("baseline", "gpu"): "#aec7e8",
    ("uwb_imu_ekf", "gpu"): "#ffbb78",
}


def _pow2_fmt(x: float, _: object) -> str:
    """Format powers of two for log-base-2 x-axes."""
    if x <= 0:
        return ""
    exp = np.log2(x)
    return f"$2^{{{round(exp)}}}$" if abs(exp - round(exp)) < 0.01 else ""


def calculate_fps(rows: list[dict]) -> list[dict]:
    """Calculate FPS metrics from timing data."""
    processed = []
    for row in rows:
        timings = np.asarray(row["timings"], dtype=float)
        processed.append(
            {
                **row,
                "pipeline": row.get("pipeline", "baseline"),
                "min_time": float(np.min(timings)),
                "mean_time": float(np.mean(timings)),
                "max_time": float(np.max(timings)),
                "fps_mean": float(row["n_worlds"] / np.mean(timings)),
                "fps_std": float(np.std(row["n_worlds"] / timings)),
            }
        )
    return processed


def plot_performance(rows: list[dict], output_dir: Path, n_drones: int = 1) -> None:
    """Plot FPS performance for Crazyflow benchmark variants and save to file."""
    filtered = [row for row in rows if row["n_drones"] == n_drones]
    if not filtered:
        logger.warning("No benchmark rows found for n_drones=%s.", n_drones)
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    keys = sorted({(row["pipeline"], row["device"]) for row in filtered})
    for pipeline, device in keys:
        series = sorted(
            (row for row in filtered if row["pipeline"] == pipeline and row["device"] == device),
            key=lambda row: row["n_worlds"],
        )
        n_worlds = np.array([row["n_worlds"] for row in series], dtype=float)
        fps = np.array([row["fps_mean"] for row in series], dtype=float)
        fps_std = np.array([row["fps_std"] for row in series], dtype=float)
        color = COLORS.get((pipeline, device), "#333333")
        label = f"{PIPELINE_LABELS.get(pipeline, pipeline)} {device.upper()}"
        ax.plot(n_worlds, fps, marker="o", color=color, label=label, linewidth=2, markersize=6)
        ax.fill_between(
            n_worlds,
            np.clip(fps - 3 * fps_std, 1e-6, None),
            fps + 3 * fps_std,
            color=color,
            alpha=0.2,
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    xticks = sorted({int(row["n_worlds"]) for row in filtered if row["n_worlds"] > 0})
    ax.set_xticks(xticks)
    ax.xaxis.set_major_formatter(FuncFormatter(_pow2_fmt))
    ax.set_xlabel("Number of Worlds", fontsize=12)
    ax.set_ylabel("Steps Per Second", fontsize=12)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=10, loc="best")

    plt.tight_layout()
    plt.show()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"performance_{n_drones:03d}.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / f"performance_{n_drones:03d}.pdf", bbox_inches="tight")
    plt.close()


def main(n_drones: int = 1) -> None:
    """Load data, calculate metrics, and create visualizations."""
    if not RESULTS_PATH.exists():
        logger.warning("No benchmark data file at %s, skipping plot generation.", RESULTS_PATH)
        return
    with RESULTS_PATH.open() as f:
        rows = json.load(f)
    plot_performance(calculate_fps(rows), PLOTS_DIR, n_drones=n_drones)


if __name__ == "__main__":
    import fire

    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)
