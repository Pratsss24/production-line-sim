"""The questions a line simulation exists to answer.

Three of them, each with a result that is hard to get from a spreadsheet:

1. **How much buffer?** Throughput rises with buffer size and so does WIP.
   The curve has a knee, and the knee is the decision.
2. **Where does improvement pay?** Making a non-bottleneck station faster
   changes nothing. This is Goldratt's point, and seeing it come out of a
   model is more convincing than being told it.
3. **Speed or reliability?** A cheaper cycle time and a better MTBF can buy
   the same capacity on paper. They do not buy the same throughput.

Every number carries a confidence interval, because a single replication of a
stochastic model can show an improvement that is not there.
"""

from __future__ import annotations

from dataclasses import dataclass

from line import LineConfig, simulate_replications
from metrics import Interval, confidence, paired_difference


def _measure(config: LineConfig, replications: int, base_seed: int):
    runs = simulate_replications(config, replications, base_seed)
    return (
        confidence([r.throughput_per_hour for r in runs]),
        confidence([r.mean_wip for r in runs]),
        confidence([r.mean_flow_time for r in runs]),
    )


def _throughputs(config: LineConfig, replications: int,
                 base_seed: int) -> list[float]:
    """Per-replication throughput, in seed order, so pairs line up."""
    return [r.throughput_per_hour
            for r in simulate_replications(config, replications, base_seed)]


# --- 1. buffer sweep ------------------------------------------------------

@dataclass
class BufferPoint:
    capacity: int
    throughput: Interval
    wip: Interval
    flow_time: Interval


def buffer_sweep(config: LineConfig, capacities: list[int] | None = None,
                 replications: int = 10, base_seed: int = 20261031
                 ) -> list[BufferPoint]:
    """Throughput and WIP against buffer size.

    A SimPy Store needs capacity >= 1, so capacity 1 here is the minimum
    decoupling -- one part may wait between stations. True zero-buffer
    operation (a part can only move if the next machine is free) is a
    different model, not a capacity of zero.
    """
    capacities = capacities or [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]
    points = []
    for cap in capacities:
        throughput, wip, flow = _measure(
            config.with_buffers(cap), replications, base_seed)
        points.append(BufferPoint(cap, throughput, wip, flow))
    return points


def knee(points: list[BufferPoint], capture: float = 0.8) -> BufferPoint:
    """Smallest buffer capturing `capture` of the gain the sweep achieves.

    Not "within a percent of the best": on a line whose buffer curve is
    gradual rather than kneed, that criterion picks the largest capacity
    tested and calls it a knee. Measuring against the gain actually on offer
    -- worst to best across the range swept -- says something either way, and
    if the answer is "most of the range", the honest reading is that buffer is
    a poor lever on this line, not that the knee lies further right.
    """
    low = min(p.throughput.mean for p in points)
    high = max(p.throughput.mean for p in points)
    target = low + capture * (high - low)
    for point in points:
        if point.throughput.mean >= target:
            return point
    return points[-1]


def marginal_gain(points: list[BufferPoint]) -> list[tuple[int, float]]:
    """Parts/h bought per extra part of WIP, step by step along the sweep.

    The number the decision actually turns on. Buffer is not free: it is WIP,
    floor space and flow time, and flow time is how long a quality problem
    stays undetected on the line.
    """
    out = []
    for prev, point in zip(points, points[1:]):
        d_wip = point.wip.mean - prev.wip.mean
        d_thr = point.throughput.mean - prev.throughput.mean
        out.append((point.capacity, d_thr / d_wip if d_wip else float("nan")))
    return out


# --- 2. where improvement pays -------------------------------------------

# An effect smaller than this share of baseline throughput is not worth a
# capital request, however cleanly the statistics resolve it.
MATERIAL_SHARE = 0.01


@dataclass
class ImprovementPoint:
    station: str
    is_bottleneck: bool
    throughput: Interval
    difference: Interval  # paired against the baseline
    baseline: float = 0.0

    @property
    def delta(self) -> float:
        return self.difference.mean

    @property
    def significant(self) -> bool:
        """Whether the paired interval excludes zero."""
        return self.difference.low > 0 or self.difference.high < 0

    @property
    def material(self) -> bool:
        """Whether the effect is big enough to act on.

        Pairing is sensitive enough to resolve a tenth of a part per hour,
        which makes "statistically significant" a weak standard: on this line
        every station clears it, because with finite buffers a faster
        neighbour does refill the bottleneck's buffer slightly sooner after a
        repair. Significance says the effect is real; materiality says whether
        anyone should care. Reporting only the first is how a 0.2% gain gets
        written up as an improvement.
        """
        return abs(self.delta) >= MATERIAL_SHARE * self.baseline

    @property
    def verdict(self) -> str:
        if not self.significant:
            return "within noise"
        if not self.material:
            return f"real but under {MATERIAL_SHARE:.0%}"
        return "material"


def improvement_sensitivity(config: LineConfig, factor: float = 0.90,
                            replications: int = 10, base_seed: int = 20261031
                            ) -> tuple[Interval, list[ImprovementPoint]]:
    """Speed up one station at a time by `factor` and see what the line does."""
    base_values = _throughputs(config, replications, base_seed)
    baseline = confidence(base_values)
    bottleneck = config.bottleneck.name

    points = []
    for index, station in enumerate(config.stations):
        improved = config.with_station(index, mean_cycle=station.mean_cycle * factor)
        values = _throughputs(improved, replications, base_seed)
        points.append(ImprovementPoint(
            station=station.name,
            is_bottleneck=station.name == bottleneck,
            throughput=confidence(values),
            difference=paired_difference(values, base_values),
            baseline=baseline.mean,
        ))
    return baseline, points


# --- 3. speed against reliability ----------------------------------------

@dataclass
class LeverPoint:
    label: str
    effective_cycle: float
    throughput: Interval
    difference: Interval
    baseline: float = 0.0

    @property
    def delta(self) -> float:
        return self.difference.mean

    @property
    def significant(self) -> bool:
        return self.difference.low > 0 or self.difference.high < 0

    @property
    def material(self) -> bool:
        return abs(self.delta) >= MATERIAL_SHARE * self.baseline

    @property
    def verdict(self) -> str:
        if not self.significant:
            return "within noise"
        if not self.material:
            return f"real but under {MATERIAL_SHARE:.0%}"
        return "material"


def speed_vs_reliability(config: LineConfig, replications: int = 10,
                         base_seed: int = 20261031
                         ) -> tuple[Interval, list[LeverPoint]]:
    """Three ways to spend money on the bottleneck, matched on paper capacity.

    Cutting the cycle time, halving the repair time and halving the cycle-time
    variability are not interchangeable even when two of them give the same
    effective cycle: variability does not show up in the capacity calculation
    at all, and it is the one that costs nothing in capital.
    """
    index = config.stations.index(config.bottleneck)
    station = config.stations[index]
    base_values = _throughputs(config, replications, base_seed)
    baseline = confidence(base_values)

    # Match the cycle-time cut to the capacity that halving MTTR buys, so the
    # two levers are compared at equal effective cycle rather than equal
    # wishful thinking.
    halved_mttr = config.with_station(index, mttr=station.mttr / 2)
    target_cycle = station.mean_cycle * (
        halved_mttr.stations[index].availability / station.availability) ** -1
    variants = {
        f"cycle {station.mean_cycle:.1f} -> {target_cycle:.1f} s":
            config.with_station(index, mean_cycle=target_cycle),
        f"MTTR {station.mttr:.0f} -> {station.mttr / 2:.0f} s": halved_mttr,
        f"cycle CV {station.cycle_cv:.2f} -> {station.cycle_cv / 2:.2f}":
            config.with_station(index, cycle_cv=station.cycle_cv / 2),
    }

    points = []
    for label, variant in variants.items():
        values = _throughputs(variant, replications, base_seed)
        points.append(LeverPoint(
            label=label,
            effective_cycle=variant.stations[index].effective_cycle,
            throughput=confidence(values),
            difference=paired_difference(values, base_values),
            baseline=baseline.mean,
        ))
    return baseline, points
