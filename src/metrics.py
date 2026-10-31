"""Reading a simulation run: bottleneck, Little's law, confidence intervals.

Kept apart from the model so every number here can be checked against a
hand-computed case without starting the SimPy engine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from line import LineConfig, RunResult, Station

# Two-sided 95% t critical values, df = n-1. A normal 1.96 on five
# replications understates the interval by about a quarter, which is exactly
# the margin an "improvement" tends to hide in.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086,
        30: 2.042}


def t95(df: int) -> float:
    if df <= 0:
        return float("nan")
    if df in _T95:
        return _T95[df]
    smaller = max(k for k in _T95 if k <= df)
    return _T95[smaller]


@dataclass
class Interval:
    """A mean with a 95% confidence half-width."""

    mean: float
    half_width: float
    n: int

    @property
    def low(self) -> float:
        return self.mean - self.half_width

    @property
    def high(self) -> float:
        return self.mean + self.half_width

    def overlaps(self, other: "Interval") -> bool:
        """Whether a difference between two configurations could be noise.

        Overlapping intervals are not proof of no difference, but they are
        proof that this many replications cannot show one.
        """
        return self.low <= other.high and other.low <= self.high

    def __str__(self) -> str:
        return f"{self.mean:.2f} +/- {self.half_width:.2f}"


def paired_difference(treated: list[float], baseline: list[float]) -> Interval:
    """Confidence interval on the per-replication difference.

    The variants in `experiments.py` are run on the same seeds, so replication
    k of the improved line and replication k of the baseline share every
    failure time at every station the change did not touch. The right
    comparison is therefore the mean of the differences, not the difference of
    the means with two independent intervals: the shared variation cancels
    inside each pair, and the interval collapses to the size of the effect
    being measured. Comparing the two independent intervals instead throws
    that away and will call a real improvement "noise".
    """
    if len(treated) != len(baseline):
        raise ValueError("paired comparison needs equal-length samples")
    return confidence([a - b for a, b in zip(treated, baseline)])


def confidence(values: list[float]) -> Interval:
    n = len(values)
    if n == 0:
        return Interval(float("nan"), float("nan"), 0)
    mean = sum(values) / n
    if n == 1:
        return Interval(mean, float("nan"), 1)
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    half = t95(n - 1) * math.sqrt(variance / n)
    return Interval(mean, half, n)


# --- bottleneck -----------------------------------------------------------

def bottleneck_by_capacity(config: LineConfig) -> Station:
    """The analytic bottleneck: slowest effective cycle time.

    What you can work out from a process sheet before building anything.
    """
    return config.bottleneck


def bottleneck_by_waiting(result: RunResult) -> str:
    """The observed bottleneck: the station that waits on others least.

    On a real line nobody hands you the cycle times. What you can measure is
    which machine is never starved and never blocked -- it is the one setting
    the pace, because everything else is waiting on it. This is the shop-floor
    version of the same answer and it needs no process sheet.
    """
    worst = min(result.stations,
                key=lambda s: (s.starved + s.blocked) / result.window)
    return worst.name


def starvation_signature(result: RunResult) -> dict[str, str]:
    """Which side each station waits on: upstream, downstream, or neither.

    Reading this across the line locates the bottleneck without any arithmetic:
    stations upstream of it are blocked, stations downstream of it are starved,
    and the bottleneck itself is neither.
    """
    pace = bottleneck_by_waiting(result)
    out: dict[str, str] = {}
    for stat in result.stations:
        if stat.name == pace:
            out[stat.name] = "pace-setter (waits on nobody)"
        elif stat.blocked > stat.starved:
            out[stat.name] = "blocked (downstream full)"
        else:
            out[stat.name] = "starved (upstream empty)"
    return out


# --- Little's law ---------------------------------------------------------

@dataclass
class LittleCheck:
    wip: float
    throughput_per_second: float
    flow_time: float
    predicted_wip: float
    error: float

    @property
    def relative_error(self) -> float:
        return abs(self.error) / self.wip if self.wip else float("nan")


def littles_law(result: RunResult) -> LittleCheck:
    """WIP = throughput x flow time, checked against the measured values.

    This is not a model output -- it is a conservation identity that any
    correct run must satisfy. If it fails by more than a percent or two, the
    WIP integral or the flow-time bookkeeping is wrong, so it is the single
    most useful test of the simulation itself.
    """
    # Departures, not good parts: the identity is about parts leaving the
    # system. Using good parts only leaves the scrap unaccounted for and the
    # balance fails by roughly the scrap rate -- which looks like a bug in the
    # WIP integral and is not one.
    lam = result.departures / result.window if result.window else 0.0
    flow = result.mean_departure_flow_time
    predicted = lam * flow
    return LittleCheck(
        wip=result.mean_wip,
        throughput_per_second=lam,
        flow_time=flow,
        predicted_wip=predicted,
        error=predicted - result.mean_wip,
    )


# --- summary --------------------------------------------------------------

def summarise(result: RunResult) -> str:
    config = result.config
    lines = [
        f"{config.name}   buffers {config.buffers()}   "
        f"{config.run_seconds / 3600:.0f} h after {config.warmup_seconds / 3600:.0f} h warm-up",
        "",
        f"{'station':14s} {'busy':>7s} {'blocked':>8s} {'starved':>8s} {'down':>7s} {'parts':>7s}",
    ]
    for stat in result.stations:
        share = stat.shares(result.window)
        lines.append(
            f"{stat.name:14s} {share['busy']:6.1%} {share['blocked']:8.1%} "
            f"{share['starved']:8.1%} {share['down']:7.1%} {stat.completed:7d}"
        )

    little = littles_law(result)
    lines += [
        "",
        f"throughput        {result.throughput_per_hour:6.2f} parts/h",
        f"bottleneck ceiling{config.theoretical_rate:6.2f} parts/h "
        f"({config.bottleneck.name})",
        f"line efficiency   {result.line_efficiency:6.1%}",
        f"good parts        {result.completed:6d}  scrap {result.scrapped}",
        "",
        f"mean WIP          {result.mean_wip:6.2f} parts",
        f"mean flow time    {result.mean_flow_time:6.1f} s "
        f"({result.mean_flow_time / 60:.1f} min)",
        f"Little's law      WIP {little.wip:.2f} vs lambda*W {little.predicted_wip:.2f} "
        f"({little.relative_error:.2%})",
    ]
    return "\n".join(lines)
