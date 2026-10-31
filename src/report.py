"""A single self-contained HTML page: the run, the experiments, the figures."""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path

from experiments import (BufferPoint, ImprovementPoint, LeverPoint,
                         marginal_gain)
from line import RunResult
from metrics import Interval, littles_law, starvation_signature

CSS = """
body { font: 15px/1.55 -apple-system, Segoe UI, Roboto, sans-serif;
       max-width: 980px; margin: 2.2rem auto; padding: 0 1.2rem;
       color: #1d2227; }
h1 { font-size: 1.6rem; margin-bottom: 0.2rem; }
h2 { font-size: 1.15rem; margin-top: 2.2rem; border-bottom: 1px solid #e3e6ea;
     padding-bottom: 0.3rem; }
.sub { color: #666; margin-top: 0; }
table { border-collapse: collapse; width: 100%; margin: 0.9rem 0;
        font-size: 0.92rem; }
th, td { padding: 0.42rem 0.6rem; border-bottom: 1px solid #e8eaed;
         text-align: right; }
th:first-child, td:first-child { text-align: left; }
th { background: #f6f7f9; font-weight: 600; }
tr.highlight td { background: #eef7f0; font-weight: 600; }
td.noise { color: #999; }
img { max-width: 100%; margin: 0.8rem 0; border: 1px solid #e3e6ea;
      border-radius: 4px; }
.kpi { display: flex; gap: 1.6rem; flex-wrap: wrap; margin: 1rem 0 0.4rem; }
.kpi div { background: #f6f7f9; border-radius: 6px; padding: 0.7rem 1.1rem;
           min-width: 150px; }
.kpi b { display: block; font-size: 1.5rem; font-weight: 600; }
.kpi span { color: #666; font-size: 0.83rem; text-transform: uppercase;
            letter-spacing: 0.04em; }
p.note { background: #fbf7ec; border-left: 3px solid #e8a33d;
         padding: 0.6rem 0.9rem; margin: 1rem 0; font-size: 0.93rem; }
footer { margin-top: 2.5rem; color: #888; font-size: 0.85rem; }
"""


def _interval(value: Interval) -> str:
    return f"{value.mean:.2f} &plusmn; {value.half_width:.2f}"


def build(result: RunResult, sweep: list[BufferPoint], knee: BufferPoint,
          improvement_baseline: Interval, improvements: list[ImprovementPoint],
          lever_baseline: Interval, levers: list[LeverPoint],
          figures: dict[str, Path], out_path: Path,
          replications: int, experiment_hours: float, factor: float) -> Path:
    config = result.config
    little = littles_law(result)
    signature = starvation_signature(result)
    marginals = dict(marginal_gain(sweep))

    def fig(name: str) -> str:
        path = figures.get(name)
        return (f'<img src="{html.escape(path.name)}" alt="{html.escape(name)}">'
                if path else "")

    rows_states = "".join(
        f"<tr><td>{html.escape(s.name)}</td>"
        f"<td>{s.shares(result.window)['busy']:.1%}</td>"
        f"<td>{s.shares(result.window)['blocked']:.1%}</td>"
        f"<td>{s.shares(result.window)['starved']:.1%}</td>"
        f"<td>{s.shares(result.window)['down']:.1%}</td>"
        f"<td>{s.failures}</td><td>{s.completed}</td>"
        f"<td style='text-align:left'>{html.escape(signature[s.name])}</td></tr>"
        for s in result.stations)

    def row(cells: list[str], highlight: bool = False) -> str:
        klass = ' class="highlight"' if highlight else ""
        return f"<tr{klass}>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"

    rows_sweep = "".join(
        row([str(p.capacity), _interval(p.throughput), f"{p.wip.mean:.2f}",
             f"{p.flow_time.mean / 60:.1f}",
             f"{marginals.get(p.capacity, float('nan')):.3f}"],
            highlight=p.capacity == knee.capacity)
        for p in sweep)

    rows_improve = "".join(
        row([html.escape(p.station), _interval(p.throughput),
             f"{p.delta:+.2f} &plusmn; {p.difference.half_width:.2f}",
             p.verdict if p.material
             else f'<span style="color:#999">{p.verdict}</span>'],
            highlight=p.is_bottleneck)
        for p in improvements)

    rows_levers = "".join(
        row([html.escape(p.label), f"{p.effective_cycle:.2f}",
             _interval(p.throughput),
             f"{p.delta:+.2f} &plusmn; {p.difference.half_width:.2f}",
             p.verdict if p.material
             else f'<span style="color:#999">{p.verdict}</span>'])
        for p in levers)

    worst = max(result.stations,
                key=lambda s: s.shares(result.window)["starved"])

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{html.escape(config.name)} - line simulation</title>
<style>{CSS}</style></head><body>

<h1>{html.escape(config.name)}</h1>
<p class="sub">Discrete-event simulation &middot;
{len(config.stations)} stations &middot; buffers {config.buffers()} &middot;
{config.run_seconds / 3600:.0f} h observed after a
{config.warmup_seconds / 3600:.0f} h warm-up &middot;
generated {datetime.now():%Y-%m-%d %H:%M}</p>

<div class="kpi">
  <div><span>throughput</span><b>{result.throughput_per_hour:.1f}</b>parts/h</div>
  <div><span>ceiling</span><b>{config.theoretical_rate:.1f}</b>parts/h
       ({html.escape(config.bottleneck.name)})</div>
  <div><span>line efficiency</span><b>{result.line_efficiency:.1%}</b>of the ceiling</div>
  <div><span>mean WIP</span><b>{result.mean_wip:.1f}</b>parts</div>
  <div><span>flow time</span><b>{result.mean_flow_time / 60:.1f}</b>min</div>
</div>

<p class="note"><b>Little's law check.</b> Mean WIP {little.wip:.2f} parts
against throughput &times; flow time = {little.predicted_wip:.2f}
({little.relative_error:.2%} apart). This is a conservation identity, not a
model output: a run that fails it has broken bookkeeping. The small residual
is the parts already in the line when the warm-up window closed.</p>

<h2>Where station time goes</h2>
{fig('states')}
<table>
<tr><th>station</th><th>busy</th><th>blocked</th><th>starved</th><th>down</th>
    <th>failures</th><th>parts</th><th>waits on</th></tr>
{rows_states}
</table>
<p>No station is overloaded, yet the line makes
{result.throughput_per_hour:.1f} parts/h against a
{config.theoretical_rate:.1f} parts/h ceiling. The gap is not downtime - the
downtime is already inside the ceiling. It is the cost of variability plus
finite buffers, and it cannot be fixed by fixing machines.
{html.escape(worst.name)} idles
{worst.shares(result.window)['starved']:.0%} of the window waiting for work
that the bottleneck has not released yet.</p>

<h2>How much buffer?</h2>
{fig('sweep')}
<table>
<tr><th>capacity</th><th>throughput (parts/h)</th><th>mean WIP</th>
    <th>flow time (min)</th><th>marginal parts/h per WIP part</th></tr>
{rows_sweep}
</table>
<p>{replications} replications per point,
{experiment_hours:.0f} h each. The marginal column is the decision: every
extra part of WIP buys less than the one before, and flow time - how long a
quality problem stays undetected on the line - rises with it the whole way.</p>

<h2>Where does improvement pay?</h2>
{fig('improvement')}
<table>
<tr><th>station made {(1 - factor) * 100:.0f}% faster</th>
    <th>throughput (parts/h)</th><th>paired change</th><th>verdict</th></tr>
{rows_improve}
</table>
<p>Baseline {_interval(improvement_baseline)} parts/h. The change column is
the <b>paired</b> difference: every variant is run on the same seeds, so
replication <i>k</i> of the improved line shares every failure time at every
station the change did not touch. The shared variation cancels inside each
pair, which is why the interval on a change is far tighter than the interval
on either line's absolute throughput. "Within noise" means that paired
interval still contains zero.</p>

<h2>Speed, reliability or variability?</h2>
{fig('levers')}
<table>
<tr><th>lever at {html.escape(config.bottleneck.name)}</th>
    <th>effective cycle (s)</th><th>throughput (parts/h)</th>
    <th>paired change</th><th>verdict</th></tr>
{rows_levers}
</table>
<p>Baseline {_interval(lever_baseline)} parts/h. The first two levers are
matched on effective cycle time, so a capacity calculation would price them
identically. The third does not change effective cycle time at all - cycle
variability is invisible to that calculation, which is precisely why it gets
overlooked.</p>

<footer>Generated by <code>python run.py</code>. Seeded, so re-running
reproduces every figure above.</footer>
</body></html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")
    return out_path
