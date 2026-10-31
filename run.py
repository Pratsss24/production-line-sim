"""Run the line, run the experiments, write the figures and the report.

    python run.py                 # everything, ~1 min
    python run.py --quick         # fewer replications, shorter horizon
    python run.py --line data/line.json --seed 7
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import charts  # noqa: E402
import experiments as ex  # noqa: E402
import metrics  # noqa: E402
import report  # noqa: E402
from line import LineConfig, simulate  # noqa: E402

IMPROVEMENT_FACTOR = 0.90


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--line", default=str(ROOT / "data" / "line.json"))
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--out", default=str(ROOT / "out"))
    parser.add_argument("--replications", type=int, default=12)
    parser.add_argument("--experiment-hours", type=float, default=40.0,
                        help="horizon per replication in the experiments; the "
                             "headline run always uses the shift length in "
                             "the line file")
    parser.add_argument("--quick", action="store_true",
                        help="4 replications over 8 h - for a smoke test, not "
                             "for a number you intend to quote")
    args = parser.parse_args()

    if args.quick:
        args.replications, args.experiment_hours = 4, 8.0

    out_dir = Path(args.out)
    config = LineConfig.from_json(args.line)

    # --- the headline run: one shift, reported as a shift ------------------
    result = simulate(config, seed=args.seed)
    print(metrics.summarise(result))
    print()
    print(f"analytic bottleneck : {metrics.bottleneck_by_capacity(config).name} "
          f"(slowest effective cycle)")
    print(f"observed bottleneck : {metrics.bottleneck_by_waiting(result)} "
          f"(waits on nobody)")
    for name, role in metrics.starvation_signature(result).items():
        print(f"  {name:14s} {role}")

    # The experiments run longer than one shift on purpose. Eight hours is the
    # right window to report a shift in and far too short to separate two
    # configurations: the failure process alone puts a couple of parts/h of
    # noise on the answer.
    long_config = replace(config, run_seconds=args.experiment_hours * 3600)
    common = dict(replications=args.replications, base_seed=args.seed)
    print(f"\n{args.replications} replications x {args.experiment_hours:.0f} h "
          f"per experiment point\n")

    sweep = ex.buffer_sweep(long_config, **common)
    knee = ex.knee(sweep)
    marginals = dict(ex.marginal_gain(sweep))
    print("BUFFER SWEEP")
    for point in sweep:
        mark = "  <- 80% of the gain" if point.capacity == knee.capacity else ""
        print(f"  capacity {point.capacity:3d}   {point.throughput} parts/h   "
              f"WIP {point.wip.mean:6.2f}   flow {point.flow_time.mean / 60:5.1f} min"
              f"   marginal {marginals.get(point.capacity, float('nan')):+.3f}{mark}")

    improve_base, improvements = ex.improvement_sensitivity(
        long_config, factor=IMPROVEMENT_FACTOR, **common)
    print(f"\nCYCLE TIME {(1 - IMPROVEMENT_FACTOR) * 100:.0f}% FASTER, ONE "
          f"STATION AT A TIME   baseline {improve_base} parts/h")
    for point in improvements:
        tag = "BOTTLENECK" if point.is_bottleneck else ""
        verdict = point.verdict
        print(f"  {point.station:14s} {point.throughput} parts/h   "
              f"paired {point.delta:+6.2f} +/- {point.difference.half_width:.2f}   "
              f"{verdict:20s} {tag}")

    lever_base, levers = ex.speed_vs_reliability(long_config, **common)
    print(f"\nLEVERS AT THE BOTTLENECK   baseline {lever_base} parts/h")
    for point in levers:
        verdict = point.verdict
        print(f"  {point.label:34s} eff. cycle {point.effective_cycle:5.2f} s   "
              f"{point.throughput} parts/h   "
              f"paired {point.delta:+6.2f} +/- {point.difference.half_width:.2f}   "
              f"{verdict}")

    figures = {
        "states": charts.station_states(result, out_dir),
        "sweep": charts.buffer_sweep(sweep, config.theoretical_rate, out_dir,
                                     knee_capacity=knee.capacity),
        "improvement": charts.improvement(improve_base, improvements, out_dir,
                                          IMPROVEMENT_FACTOR),
        "levers": charts.levers(lever_base, levers, out_dir),
    }
    html_path = report.build(
        result, sweep, knee, improve_base, improvements, lever_base, levers,
        figures, out_dir / "report.html", args.replications,
        args.experiment_hours, IMPROVEMENT_FACTOR)

    print(f"\nreport  {html_path}")
    for path in figures.values():
        print(f"figure  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
