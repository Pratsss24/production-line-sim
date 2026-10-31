"""Discrete-event model of a serial production line with finite buffers.

The line is a chain of stations separated by buffers of limited size. Two
phenomena that only appear once buffers are finite drive everything here:

* **Starving** -- a station waits because its input buffer is empty.
* **Blocking** -- a station has finished a part but cannot hand it on because
  the downstream buffer is full, so it stands still holding the part.

A line of stations each capable of 60 parts an hour does not make 60 parts an
hour. Variability plus finite buffers costs throughput even when no station is
overloaded and nothing breaks, and that loss is what this model measures.

Failures consume *operating* time, not calendar time: a machine that is
starved is not wearing out. Time to failure is sampled in operating seconds
and decremented only while a station is actually cutting, and a failure that
lands mid-part interrupts it, repairs, and then finishes the remaining work.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import simpy


@dataclass
class Station:
    """One machine. Cycle time is gamma-distributed about `mean_cycle`.

    `cycle_cv` is the coefficient of variation (sigma/mean); 0 means a
    deterministic machine, 0.2 is typical of a manual-load operation.
    `mtbf_operating` is mean time between failures in *operating* seconds.
    """

    name: str
    mean_cycle: float
    cycle_cv: float = 0.0
    mtbf_operating: float = float("inf")
    mttr: float = 0.0
    scrap_rate: float = 0.0

    @property
    def availability(self) -> float:
        if self.mtbf_operating == float("inf"):
            return 1.0
        return self.mtbf_operating / (self.mtbf_operating + self.mttr)

    @property
    def effective_cycle(self) -> float:
        """Mean cycle inflated by downtime: the station's real capacity.

        Ranking stations by this rather than by nameplate cycle time is what
        stops a fast-but-unreliable machine from hiding behind its spec sheet.
        """
        return self.mean_cycle / self.availability

    @property
    def capacity_per_hour(self) -> float:
        return 3600.0 / self.effective_cycle

    def sample_cycle(self, rng: np.random.Generator) -> float:
        if self.cycle_cv <= 0:
            return self.mean_cycle
        # Gamma keeps cycle times positive and right-skewed, which is how real
        # cycle times behave: a cycle can run long, never shorter than zero.
        shape = 1.0 / (self.cycle_cv**2)
        scale = self.mean_cycle / shape
        return float(rng.gamma(shape, scale))

    def sample_ttf(self, rng: np.random.Generator) -> float:
        if self.mtbf_operating == float("inf"):
            return float("inf")
        return float(rng.exponential(self.mtbf_operating))

    def sample_repair(self, rng: np.random.Generator) -> float:
        return float(rng.exponential(self.mttr)) if self.mttr > 0 else 0.0


# Each station draws from its own independent streams, one per purpose.
STREAMS = ("cycle", "ttf", "repair", "scrap")


def station_streams(seed: int, count: int) -> list[dict[str, np.random.Generator]]:
    """Independent generators per station per purpose.

    This is the whole reason comparisons in `experiments.py` are readable. With
    one shared generator, making station 2 faster changes how many draws it
    takes, which shifts every later draw -- so station 5's failures land at
    different times and the throughput difference you measure is partly that
    reshuffle rather than the change you made. Separate streams keep the
    failure sequence at every untouched station identical between the two
    runs, which is the variance reduction known as common random numbers.
    """
    roots = np.random.SeedSequence(seed).spawn(count)
    return [
        {name: np.random.default_rng(child)
         for name, child in zip(STREAMS, root.spawn(len(STREAMS)))}
        for root in roots
    ]


@dataclass
class LineConfig:
    name: str
    stations: list[Station]
    buffer_capacity: int | list[int] = 2
    run_seconds: float = 8 * 3600.0
    warmup_seconds: float = 3600.0

    def buffers(self) -> list[int]:
        """Buffer sizes between consecutive stations: len(stations) - 1 of them."""
        n = len(self.stations) - 1
        if isinstance(self.buffer_capacity, int):
            return [self.buffer_capacity] * n
        if len(self.buffer_capacity) != n:
            raise ValueError(
                f"{len(self.stations)} stations need {n} buffers, "
                f"got {len(self.buffer_capacity)}"
            )
        return list(self.buffer_capacity)

    @classmethod
    def from_json(cls, path: str | Path) -> "LineConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f for f in Station.__dataclass_fields__}
        stations = []
        for entry in raw["stations"]:
            unknown = set(entry) - known
            if unknown:
                # A silently ignored key is a tolerance or an MTBF that the
                # engineer thinks is in the model and is not.
                raise ValueError(f"unknown station keys: {sorted(unknown)}")
            stations.append(Station(**entry))
        return cls(
            name=raw["name"],
            stations=stations,
            buffer_capacity=raw.get("buffer_capacity", 2),
            run_seconds=raw.get("run_seconds", 8 * 3600.0),
            warmup_seconds=raw.get("warmup_seconds", 3600.0),
        )

    def with_buffers(self, capacity: int | list[int]) -> "LineConfig":
        return replace(self, buffer_capacity=capacity)

    def with_station(self, index: int, **changes) -> "LineConfig":
        stations = list(self.stations)
        stations[index] = replace(stations[index], **changes)
        return replace(self, stations=stations)

    @property
    def bottleneck(self) -> Station:
        """Slowest station by effective cycle time."""
        return max(self.stations, key=lambda s: s.effective_cycle)

    @property
    def theoretical_rate(self) -> float:
        """Parts per hour if only the bottleneck mattered.

        This is the ceiling the line can never beat and, with finite buffers
        and variability, never quite reaches.
        """
        return self.bottleneck.capacity_per_hour


@dataclass
class StationStats:
    name: str
    busy: float = 0.0
    blocked: float = 0.0
    starved: float = 0.0
    down: float = 0.0
    completed: int = 0
    scrapped: int = 0
    failures: int = 0

    def accounted(self) -> float:
        return self.busy + self.blocked + self.starved + self.down

    def shares(self, window: float) -> dict[str, float]:
        if window <= 0:
            return {}
        return {
            "busy": self.busy / window,
            "blocked": self.blocked / window,
            "starved": self.starved / window,
            "down": self.down / window,
        }


@dataclass
class Part:
    index: int
    entered: float
    exited: float | None = None

    @property
    def flow_time(self) -> float | None:
        return None if self.exited is None else self.exited - self.entered


@dataclass
class RunResult:
    config: LineConfig
    stations: list[StationStats]
    window: float
    completed: int
    scrapped: int
    flow_times: list[float] = field(default_factory=list)
    # Every part that LEFT the system, good or scrapped. Throughput is good
    # parts, but Little's law is about departures: a part scrapped at the last
    # station occupied the line all the way there and must not vanish from the
    # WIP balance.
    departure_flow_times: list[float] = field(default_factory=list)
    wip_area: float = 0.0  # integral of WIP over time, for the time-average

    @property
    def throughput_per_hour(self) -> float:
        return self.completed / self.window * 3600.0 if self.window > 0 else 0.0

    @property
    def mean_wip(self) -> float:
        return self.wip_area / self.window if self.window > 0 else 0.0

    @property
    def mean_flow_time(self) -> float:
        return float(np.mean(self.flow_times)) if self.flow_times else float("nan")

    @property
    def departures(self) -> int:
        return self.completed + self.scrapped

    @property
    def mean_departure_flow_time(self) -> float:
        if not self.departure_flow_times:
            return float("nan")
        return float(np.mean(self.departure_flow_times))

    @property
    def line_efficiency(self) -> float:
        """Achieved throughput against the bottleneck's own capacity.

        The shortfall is the cost of variability and finite buffers -- it is
        not downtime, and no amount of fixing machines removes it.
        """
        theoretical = self.config.theoretical_rate
        return self.throughput_per_hour / theoretical if theoretical > 0 else 0.0


class Line:
    """The SimPy model itself."""

    def __init__(self, config: LineConfig, seed: int = 20261031) -> None:
        self.config = config
        self.streams = station_streams(seed, len(config.stations))
        self.env = simpy.Environment()
        self.stats = [StationStats(name=s.name) for s in config.stations]
        self.buffers = [
            simpy.Store(self.env, capacity=cap) for cap in config.buffers()
        ]
        self.parts: list[Part] = []
        self.completed_parts: list[Part] = []
        self.departed_parts: list[Part] = []
        self._next_index = 0
        self._wip = 0
        self._wip_area = 0.0
        self._last_wip_change = 0.0
        self._window_start = 0.0
        self._measuring = False

    # --- bookkeeping ----------------------------------------------------

    def _reset_stats(self) -> None:
        """Clear everything at the end of warm-up.

        Starting measurement at t=0 would average in an empty line filling up,
        which understates throughput and overstates starvation. The warm-up is
        discarded, which is standard practice for a terminating-to-steady-state
        DES run.
        """
        for stat in self.stats:
            stat.busy = stat.blocked = stat.starved = stat.down = 0.0
            stat.completed = stat.scrapped = stat.failures = 0
        self.completed_parts.clear()
        self.departed_parts.clear()
        self._wip_area = 0.0
        self._last_wip_change = self.env.now
        self._window_start = self.env.now
        self._measuring = True

    def _track_wip(self, delta: int) -> None:
        if self._measuring:
            self._wip_area += self._wip * (self.env.now - self._last_wip_change)
            self._last_wip_change = self.env.now
        self._wip += delta

    def _charge(self, index: int, bucket: str, start: float) -> None:
        """Charge the time from `start` until now, clipped to the window.

        Clipping matters: an activity that began during the warm-up and ended
        inside the measurement window would otherwise contribute its whole
        duration to a window it was only partly in. On a station whose repairs
        run into the minutes that is a percent or more of the window, charged
        to whichever bucket happened to straddle the boundary.
        """
        if not self._measuring:
            return
        seconds = self.env.now - max(start, self._window_start)
        if seconds > 0:
            setattr(self.stats[index], bucket,
                    getattr(self.stats[index], bucket) + seconds)

    # --- processes ------------------------------------------------------

    def _process_part(self, index: int, station: Station, ttf: float):
        """Run one part through a station, interrupting for failures.

        Returns the remaining time-to-failure. Processing is consumed in
        segments so a failure can land in the middle of a part, which is what
        actually happens -- and the part is finished after the repair rather
        than restarted, since the material is still in the fixture.
        """
        streams = self.streams[index]
        remaining = station.sample_cycle(streams["cycle"])
        while remaining > 1e-9:
            segment = min(remaining, ttf)
            start = self.env.now
            yield self.env.timeout(segment)
            self._charge(index, "busy", start)
            remaining -= segment
            ttf -= segment

            if ttf <= 1e-9 and remaining > 1e-9:
                repair = station.sample_repair(streams["repair"])
                start = self.env.now
                yield self.env.timeout(repair)
                self._charge(index, "down", start)
                if self._measuring:
                    self.stats[index].failures += 1
                ttf = station.sample_ttf(streams["ttf"])
        return ttf

    def _station_process(self, index: int):
        station = self.config.stations[index]
        is_first = index == 0
        is_last = index == len(self.config.stations) - 1
        streams = self.streams[index]
        ttf = station.sample_ttf(streams["ttf"])

        while True:
            # --- take a part in
            if is_first:
                # Raw material is always available: the first station cannot
                # starve, so the line's losses are internal, not supply.
                part = Part(index=self._next_index, entered=self.env.now)
                self._next_index += 1
                self.parts.append(part)
                self._track_wip(+1)
            else:
                start = self.env.now
                part = yield self.buffers[index - 1].get()
                self._charge(index, "starved", start)

            # --- work on it
            ttf = yield from self._process_part(index, station, ttf)

            if self._measuring:
                self.stats[index].completed += 1

            # --- scrap check
            if station.scrap_rate > 0 and streams["scrap"].random() < station.scrap_rate:
                part.exited = self.env.now
                if self._measuring:
                    self.stats[index].scrapped += 1
                    self.departed_parts.append(part)
                self._track_wip(-1)
                continue

            # --- hand it on
            if is_last:
                part.exited = self.env.now
                self._track_wip(-1)
                if self._measuring:
                    self.completed_parts.append(part)
                    self.departed_parts.append(part)
            else:
                start = self.env.now
                yield self.buffers[index].put(part)
                self._charge(index, "blocked", start)

    def _warmup_process(self):
        yield self.env.timeout(self.config.warmup_seconds)
        self._reset_stats()

    def run(self) -> RunResult:
        for index in range(len(self.config.stations)):
            self.env.process(self._station_process(index))
        if self.config.warmup_seconds > 0:
            self.env.process(self._warmup_process())
        else:
            self._reset_stats()

        total = self.config.warmup_seconds + self.config.run_seconds
        self.env.run(until=total)

        # Close the WIP integral at the end of the run.
        self._wip_area += self._wip * (self.env.now - self._last_wip_change)

        window = self.config.run_seconds
        last = self.stats[-1]
        return RunResult(
            config=self.config,
            stations=self.stats,
            window=window,
            completed=last.completed - last.scrapped,
            scrapped=sum(s.scrapped for s in self.stats),
            flow_times=[p.flow_time for p in self.completed_parts
                        if p.flow_time is not None],
            departure_flow_times=[p.flow_time for p in self.departed_parts
                                  if p.flow_time is not None],
            wip_area=self._wip_area,
        )


def simulate(config: LineConfig, seed: int = 20261031) -> RunResult:
    return Line(config, seed=seed).run()


def simulate_replications(config: LineConfig, replications: int = 5,
                          base_seed: int = 20261031) -> list[RunResult]:
    """Several independent runs.

    One run of a stochastic model is one sample, not an answer. Comparing two
    configurations on single runs is how a difference that is pure noise gets
    presented as an improvement.
    """
    return [simulate(config, seed=base_seed + 1000 * i) for i in range(replications)]
