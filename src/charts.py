"""Figures. Matplotlib on Agg -- no display needed, runs anywhere."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments import BufferPoint, ImprovementPoint, LeverPoint  # noqa: E402
from line import RunResult  # noqa: E402
from metrics import Interval  # noqa: E402

STATE_COLOURS = {
    "busy": "#2a7f3f",      # making parts
    "blocked": "#c44444",   # waiting on downstream
    "starved": "#e8a33d",   # waiting on upstream
    "down": "#444444",      # broken
}


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def station_states(result: RunResult, out_dir: Path) -> Path:
    """Where every station's time goes, as a share of the window.

    The chart to read first. The pace-setter is the bar with almost no orange
    and almost no red; everything upstream of it is red, everything downstream
    of it is orange.
    """
    names = [s.name for s in result.stations]
    fig, ax = plt.subplots(figsize=(9, 0.62 * len(names) + 1.8))

    left = [0.0] * len(names)
    for bucket, colour in STATE_COLOURS.items():
        values = [s.shares(result.window)[bucket] * 100 for s in result.stations]
        ax.barh(names, values, left=left, color=colour, label=bucket,
                height=0.62, edgecolor="white", linewidth=0.6)
        for i, (value, start) in enumerate(zip(values, left)):
            if value >= 4.0:
                ax.text(start + value / 2, i, f"{value:.0f}%", ha="center",
                        va="center", color="white", fontsize=8.5, weight="bold")
        left = [a + b for a, b in zip(left, values)]

    ax.set_xlim(0, 100)
    ax.set_xlabel("share of the observed window (%)")
    ax.set_title(
        f"{result.config.name} - where station time goes\n"
        f"{result.throughput_per_hour:.1f} parts/h against a "
        f"{result.config.theoretical_rate:.1f} parts/h bottleneck ceiling",
        fontsize=11)
    ax.invert_yaxis()
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.30),
              frameon=False)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    return _save(fig, out_dir, "station_states.png")


def buffer_sweep(points: list[BufferPoint], ceiling: float, out_dir: Path,
                 knee_capacity: int | None = None) -> Path:
    """The trade-off: throughput and WIP against buffer size."""
    caps = [p.capacity for p in points]
    thr = [p.throughput.mean for p in points]
    err = [p.throughput.half_width for p in points]
    wip = [p.wip.mean for p in points]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(caps, thr, yerr=err, marker="o", color="#2a6ebb", capsize=3,
                linewidth=1.8, label="throughput (95% CI)")
    ax.axhline(ceiling, color="#c44444", linestyle="--", linewidth=1.2,
               label=f"bottleneck ceiling {ceiling:.1f}/h")
    if knee_capacity is not None:
        ax.axvline(knee_capacity, color="#777777", linestyle=":", linewidth=1.2)
        ax.text(knee_capacity, min(thr),
                f"  80% of the gain\n  at capacity {knee_capacity}",
                fontsize=8.5, color="#555555", va="bottom")
    ax.set_xlabel("buffer capacity between every pair of stations (parts)")
    ax.set_ylabel("throughput (parts/h)")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)

    twin = ax.twinx()
    twin.plot(caps, wip, marker="s", color="#e8a33d", linewidth=1.5,
              label="mean WIP")
    twin.set_ylabel("mean WIP (parts)", color="#b8791d")
    twin.tick_params(axis="y", colors="#b8791d")

    handles, labels = ax.get_legend_handles_labels()
    extra_handles, extra_labels = twin.get_legend_handles_labels()
    ax.legend(handles + extra_handles, labels + extra_labels,
              loc="lower right", frameon=False, fontsize=9)
    ax.set_title("Buffer buys throughput and pays in WIP", fontsize=11)
    return _save(fig, out_dir, "buffer_sweep.png")


def improvement(baseline: Interval, points: list[ImprovementPoint],
                out_dir: Path, factor: float) -> Path:
    """Throughput gain from speeding up each station in turn."""
    names = [p.station for p in points]
    deltas = [p.delta for p in points]
    # Error bars are the PAIRED interval: the uncertainty on the change, not
    # the much larger uncertainty on either line's absolute throughput.
    errs = [p.difference.half_width for p in points]
    colours = ["#2a7f3f" if p.is_bottleneck else
               ("#2a6ebb" if p.material else "#bbbbbb") for p in points]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(names, deltas, yerr=errs, color=colours, capsize=4, width=0.6)
    ax.axhline(0, color="#333333", linewidth=0.9)
    for i, point in enumerate(points):
        label = "bottleneck" if point.is_bottleneck else point.verdict
        if label:
            ax.text(i, point.delta + point.difference.half_width + 0.08, label,
                    ha="center", fontsize=8.5, color="#555555")
    ax.set_ylabel("change in line throughput (parts/h)")
    ax.set_title(f"Cutting one station's cycle time by "
                 f"{(1 - factor) * 100:.0f}%, one station at a time\n"
                 f"baseline {baseline.mean:.1f} parts/h", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_ylim(top=max(d + e for d, e in zip(deltas, errs)) * 1.25)
    return _save(fig, out_dir, "improvement.png")


def levers(baseline: Interval, points: list[LeverPoint], out_dir: Path) -> Path:
    """Three ways to spend money on the bottleneck."""
    names = [p.label for p in points]
    deltas = [p.delta for p in points]
    errs = [p.difference.half_width for p in points]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.barh(names, deltas, xerr=errs, color="#5b7fa6", capsize=4, height=0.55)
    ax.axvline(0, color="#333333", linewidth=0.9)
    for i, point in enumerate(points):
        ax.text(point.delta + point.difference.half_width + 0.06, i,
                f"eff. cycle {point.effective_cycle:.1f} s", va="center",
                fontsize=8.5, color="#555555")
    ax.set_xlabel("change in line throughput (parts/h)")
    ax.set_title(f"Levers at the bottleneck, baseline "
                 f"{baseline.mean:.1f} parts/h\n"
                 "the first two are matched on effective cycle time",
                 fontsize=11)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_xlim(right=max(d + e for d, e in zip(deltas, errs)) * 1.45)
    return _save(fig, out_dir, "levers.png")
