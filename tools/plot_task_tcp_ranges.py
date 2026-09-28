"""Render one TCP clipping/reset table per active task.

Each table contains only policy-activated TCP axes. Cartesian coordinates are
reported in mm; local tool yaw is reported in degrees.
"""

from __future__ import annotations

import argparse
from importlib import import_module
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Exclude legacy/calibration-only task profiles from the hardware overview.
ACTIVE_TASKS = (
    "hang_double_strings_2",
    "push_t",
    "insert_gear",
    "plug_power_socket",
    "rotate_knob",
)
AXES = ("X", "Y", "Z", "Rx", "Ry", "Rz")


def mm(value: float) -> str:
    return f"{value * 1000.0:.1f} mm"


def mm_range(low: float, high: float) -> str:
    return f"[{low * 1000.0:.1f}, {high * 1000.0:.1f}] mm"


def deg_range(low: float, high: float) -> str:
    return f"[{math.degrees(low):.0f}, {math.degrees(high):.0f}] deg"


def reset_xyz(cfg: object) -> tuple[tuple[float, float], ...]:
    """Return reset sampler intervals from its configured centre and spread."""
    return tuple(
        (center - half_width, center + half_width)
        for center, half_width in zip(cfg.reset_pos, cfg.reset_rnd_abs, strict=True)
    )


def table_rows(task_name: str) -> list[list[str]]:
    cfg = import_module(f"tasks.{task_name}.config").TaskConfig()
    reset = reset_xyz(cfg)
    rows: list[list[str]] = []
    for index, active in enumerate(cfg.movable_axes):
        if not active:
            continue
        axis = AXES[index]
        if index < 3:
            reset_value = (
                mm_range(*reset[index])
                if reset[index][0] != reset[index][1]
                else mm(reset[index][0])
            )
            rows.append([
                axis,
                mm_range(cfg.safety_pos_min[index], cfg.safety_pos_max[index]),
                reset_value,
            ])
        elif index == 5:
            yaw = getattr(cfg, "yaw_range_rad", None)
            if yaw is None:
                raise ValueError(f"{task_name} activates Rz without yaw_range_rad")
            rows.append([
                axis,
                deg_range(*yaw),
                f"{math.degrees(getattr(cfg, 'reset_yaw_rad', 0.0)):.0f} deg",
            ])
        else:
            half_range = cfg.safety_rot_half_range[index]
            reset_rot = cfg.reset_rot_rnd_abs[index - 3]
            rows.append([axis, deg_range(-half_range, half_range), deg_range(-reset_rot, reset_rot)])
    return rows


def render(task_name: str, output: Path) -> None:
    rows = table_rows(task_name)
    fig, ax = plt.subplots(figsize=(6.2, max(2.2, 0.65 * (len(rows) + 1))))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=("Axis", "Clip range", "Reset"),
        cellLoc="center",
        colLoc="center",
        colWidths=(0.16, 0.42, 0.42),
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.7)
    for (row, _column), cell in table.get_celld().items():
        cell.set_edgecolor("#c8c8c8")
        if row == 0:
            cell.set_facecolor("#1f4e79")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2:
            cell.set_facecolor("#f4f7fa")
    ax.set_title(task_name.replace("_", " "), fontsize=15, weight="bold", pad=18)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "analysis" / "task_tcp_ranges",
    )
    args = parser.parse_args()
    for task_name in ACTIVE_TASKS:
        output = args.output_dir / f"{task_name}.png"
        render(task_name, output)
        print(output)


if __name__ == "__main__":
    main()
