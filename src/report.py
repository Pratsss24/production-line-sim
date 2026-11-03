"""Build the simulation report as a printable PDF.

`paper.py` owns the page furniture; the argument below is this project's
own, in the order the argument runs.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from experiments import (BufferPoint, ImprovementPoint, LeverPoint,
                         marginal_gain)
from line import RunResult
from metrics import Interval, littles_law, starvation_signature
from paper import Paper

FOOTER = ("Seeded throughout, so re-running reproduces every figure and "
          "every interval above.")


def _interval(value: Interval) -> str:
    return f"{value.mean:.2f} ± {value.half_width:.2f}"


def _states(result: RunResult,
            signature: dict) -> tuple[list[str], list[list[str]]]:
    rows = []
    for one in result.stations:
        share = one.shares(result.window)
        rows.append([one.name, f"{share['busy']:.1%}",
                     f"{share['blocked']:.1%}", f"{share['starved']:.1%}",
                     f"{share['down']:.1%}", str(one.failures),
                     str(one.completed), signature[one.name]])
    return (["station", "busy", "blocked", "starved", "down", "failures",
             "parts", "waits on"], rows)


def _sweep(sweep: list[BufferPoint],
           knee: BufferPoint) -> tuple[list[str], list[list[str]]]:
    marginals = dict(marginal_gain(sweep))
    rows = []
    for one in sweep:
        gain = marginals.get(one.capacity, float("nan"))
        mark = "  ← knee" if one.capacity == knee.capacity else ""
        rows.append([f"{one.capacity}{mark}",
                     _interval(one.throughput), f"{one.wip.mean:.2f}",
                     f"{one.flow_time.mean / 60:.1f}",
                     "—" if gain != gain else f"{gain:.3f}"])
    return (["capacity", "throughput, parts/h", "mean WIP",
             "flow time, min", "marginal parts/h per WIP part"], rows)


def _paired(points, first: str) -> tuple[list[str], list[list[str]]]:
    rows = []
    for one in points:
        label = getattr(one, "station", None) or one.label
        if getattr(one, "is_bottleneck", False):
            label += "  ← bottleneck"
        cells = [label]
        if hasattr(one, "effective_cycle"):
            cells.append(f"{one.effective_cycle:.2f}")
        cells += [_interval(one.throughput),
                  f"{one.delta:+.2f} ± {one.difference.half_width:.2f}",
                  one.verdict]
        rows.append(cells)
    head = [first]
    if points and hasattr(points[0], "effective_cycle"):
        head.append("effective cycle, s")
    head += ["throughput, parts/h", "paired change", "verdict"]
    return (head, rows)


def build(result: RunResult, sweep: list[BufferPoint], knee: BufferPoint,
          improvement_baseline: Interval, improvements: list[ImprovementPoint],
          lever_baseline: Interval, levers: list[LeverPoint],
          figures: dict[str, Path], out_path: Path,
          replications: int, experiment_hours: float, factor: float) -> Path:
    config = result.config
    little = littles_law(result)
    signature = starvation_signature(result)
    worst = max(result.stations,
                key=lambda one: one.shares(result.window)["starved"])
    out = Path(out_path)
    out = out.with_name("report.pdf") if out.suffix != ".pdf" else out
    out.parent.mkdir(parents=True, exist_ok=True)

    with Paper(out, config.name,
               f"Discrete-event simulation · {len(config.stations)} "
               f"stations · buffers {config.buffers()} · "
               f"{config.run_seconds / 3600:.0f} h observed after a "
               f"{config.warmup_seconds / 3600:.0f} h warm-up · "
               f"generated {datetime.now():%Y-%m-%d %H:%M}") as paper:

        paper.cover(
            headline=f"{result.throughput_per_hour:.1f} parts an hour "
                     f"against a {config.theoretical_rate:.1f} ceiling",
            small=f"Line efficiency {result.line_efficiency:.1%} of the "
                  f"ceiling set by {config.bottleneck.name}. Mean WIP "
                  f"{result.mean_wip:.1f} parts, flow time "
                  f"{result.mean_flow_time / 60:.1f} minutes.",
            tiles=[("throughput, parts/h",
                    f"{result.throughput_per_hour:.1f}"),
                   ("ceiling, parts/h", f"{config.theoretical_rate:.1f}"),
                   ("line efficiency", f"{result.line_efficiency:.1%}"),
                   ("mean WIP, parts", f"{result.mean_wip:.1f}")],
            blurb=f"Little's law check: mean WIP {little.wip:.2f} parts "
                  f"against throughput × flow time = "
                  f"{little.predicted_wip:.2f} "
                  f"({little.relative_error:.2%} apart). This is a "
                  f"conservation identity, not a model output — a run "
                  f"that fails it has broken bookkeeping. The small "
                  f"residual is the parts already in the line when the "
                  f"warm-up window closed.",
            figure_path=figures.get("states"))

        paper.section(
            "Where station time goes",
            warn=f"No station is overloaded, yet the line makes "
                 f"{result.throughput_per_hour:.1f} parts/h against a "
                 f"{config.theoretical_rate:.1f} parts/h ceiling. The gap "
                 f"is not downtime — the downtime is already inside "
                 f"the ceiling. It is the cost of variability plus finite "
                 f"buffers, and it cannot be fixed by fixing machines. "
                 f"{worst.name} idles "
                 f"{worst.shares(result.window)['starved']:.0%} of the "
                 f"window waiting for work the bottleneck has not released "
                 f"yet.",
            table=_states(result, signature),
            after="Busy, blocked, starved and down are shares of the whole "
                  "observed window, so they sum to one for each station.")

        paper.section(
            "How much buffer?",
            note=f"{replications} replications per point, "
                 f"{experiment_hours:.0f} h each.",
            body="The marginal column is the decision: every extra part of "
                 "WIP buys less than the one before, and flow time — "
                 "how long a quality problem stays undetected on the line "
                 "— rises with it the whole way.",
            figure_path=figures.get("sweep"), table=_sweep(sweep, knee))

        paper.section(
            "Where does improvement pay?",
            note=f"Baseline {_interval(improvement_baseline)} parts/h.",
            body="The change column is the paired difference: every variant "
                 "is run on the same seeds, so replication k of the "
                 "improved line shares every failure time at every station "
                 "the change did not touch. The shared variation cancels "
                 "inside each pair, which is why the interval on a change "
                 "is far tighter than the interval on either line's "
                 "absolute throughput. \"Within noise\" means that paired "
                 "interval still contains zero.",
            figure_path=figures.get("improvement"),
            table=_paired(improvements,
                          f"station made {(1 - factor) * 100:.0f}% faster"))

        paper.section(
            "Speed, reliability or variability?",
            note=f"Baseline {_interval(lever_baseline)} parts/h.",
            body="The first two levers are matched on effective cycle time, "
                 "so a capacity calculation would price them identically. "
                 "The third does not change effective cycle time at all "
                 "— cycle variability is invisible to that "
                 "calculation, which is precisely why it gets overlooked.",
            figure_path=figures.get("levers"),
            table=_paired(levers, f"lever at {config.bottleneck.name}"),
            after=FOOTER)

    return out
