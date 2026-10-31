"""Checks against closed-form results, not against golden output.

Run with `python tests/test_line.py` (no pytest needed) or `pytest tests/`.

A simulation is easy to test badly: record what it printed, assert it prints
that again, and the suite passes forever while the model is wrong. Everything
here is anchored on a value worked out independently -- a deterministic line's
throughput, the availability formula, Little's law -- or on an invariant that
any correct run must satisfy whatever the numbers come out as.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import experiments as ex  # noqa: E402
import metrics  # noqa: E402
from line import LineConfig, Station, simulate, station_streams  # noqa: E402

LINE_FILE = Path(__file__).resolve().parent.parent / "data" / "line.json"


def deterministic(cycles: list[float], buffer_capacity: int = 2,
                  hours: float = 4.0, warmup: float = 0.5) -> LineConfig:
    """A line with no variability and no failures: fully predictable."""
    return LineConfig(
        name="test",
        stations=[Station(name=f"S{i}", mean_cycle=c)
                  for i, c in enumerate(cycles)],
        buffer_capacity=buffer_capacity,
        run_seconds=hours * 3600,
        warmup_seconds=warmup * 3600,
    )


# --- closed-form throughput ----------------------------------------------

def test_single_deterministic_station_hits_its_nameplate_rate() -> None:
    """One station, 36 s cycle, nothing to wait for: exactly 100 parts/h."""
    result = simulate(deterministic([36.0]))
    assert abs(result.throughput_per_hour - 100.0) < 0.01, \
        result.throughput_per_hour


def test_deterministic_line_runs_at_its_slowest_station() -> None:
    """Cycles 10/20/15 -> the 20 s station sets the rate: 180 parts/h.

    The whole point of a line: capacity is not the average of the stations,
    it is the worst of them.
    """
    result = simulate(deterministic([10.0, 20.0, 15.0]))
    assert abs(result.throughput_per_hour - 180.0) < 0.1, \
        result.throughput_per_hour


def test_a_fast_station_before_a_slow_one_is_blocked_not_busy() -> None:
    """S0 at 10 s feeding S1 at 20 s can only be busy half the time.

    The rest is blocking, and a report that called it "50% utilised" without
    saying why would hide the only fact that matters.
    """
    result = simulate(deterministic([10.0, 20.0]))
    shares = result.stations[0].shares(result.window)
    assert abs(shares["busy"] - 0.5) < 0.02, shares
    assert shares["blocked"] > 0.45, shares
    assert abs(result.stations[1].shares(result.window)["busy"] - 1.0) < 0.02


def test_downtime_costs_exactly_the_availability_formula() -> None:
    """MTBF 100 s of operating time, MTTR 100 s -> availability 0.50.

    A 10 s cycle at 50% availability is 180 parts/h, not 360. This is the
    closed-form check on the failure model itself: time to failure is
    consumed in operating seconds, so over a long run the station must settle
    at MTBF/(MTBF+MTTR) regardless of how the failures happen to fall.
    """
    config = LineConfig(
        name="one unreliable machine",
        stations=[Station(name="S0", mean_cycle=10.0, mtbf_operating=100.0,
                          mttr=100.0)],
        run_seconds=200 * 3600.0,
        warmup_seconds=3600.0,
    )
    assert abs(config.stations[0].availability - 0.5) < 1e-12

    # Averaged over replications, not asserted on one run. A single 200 h run
    # sees about 3 600 failures, so its throughput carries a standard
    # deviation near 2.4 parts/h: a tolerance tight enough to be worth
    # asserting would fail on a fifth of seeds, and one loose enough to pass
    # would not catch a 2% error in the failure model. Ten replications bring
    # the standard error under 0.8 and the bound below the error worth
    # catching.
    rates = [simulate(config, seed=seed).throughput_per_hour
             for seed in range(1, 11)]
    mean = sum(rates) / len(rates)
    assert abs(mean - 180.0) < 1.6, (mean, rates)


def test_capacity_ranking_uses_effective_not_nameplate_cycle() -> None:
    """A fast unreliable machine can be slower than a slow reliable one."""
    fast_fragile = Station(name="fast", mean_cycle=40.0,
                           mtbf_operating=600.0, mttr=600.0)   # A = 0.5
    slow_solid = Station(name="slow", mean_cycle=50.0)          # A = 1.0
    config = LineConfig(name="x", stations=[fast_fragile, slow_solid])
    assert fast_fragile.mean_cycle < slow_solid.mean_cycle
    assert fast_fragile.effective_cycle == 80.0
    assert config.bottleneck.name == "fast"


# --- invariants any correct run must satisfy -----------------------------

def test_littles_law_holds_on_the_real_line() -> None:
    """WIP = throughput x flow time. A conservation identity, not an output.

    The residual is the parts already inside the line when the warm-up window
    closed: their flow time started before the window the WIP integral covers.
    """
    result = simulate(LineConfig.from_json(LINE_FILE))
    check = metrics.littles_law(result)
    assert check.relative_error < 0.02, check


def test_station_time_accounts_for_the_whole_window() -> None:
    """busy + blocked + starved + down must fill the window.

    Short by at most the one activity still in flight when the window closed.
    Anything larger is time the model lost track of.
    """
    config = LineConfig.from_json(LINE_FILE)
    result = simulate(config)
    longest = max(max(s.mean_cycle * 4, s.mttr * 4) for s in config.stations)
    for stat in result.stations:
        gap = result.window - stat.accounted()
        assert -1e-6 <= gap < longest, (stat.name, gap)


def test_a_station_is_never_charged_for_two_things_at_once() -> None:
    """No bucket may exceed the window, individually or together."""
    result = simulate(LineConfig.from_json(LINE_FILE))
    for stat in result.stations:
        for bucket, share in stat.shares(result.window).items():
            assert 0.0 <= share <= 1.0 + 1e-9, (stat.name, bucket, share)


def test_parts_are_conserved() -> None:
    """Everything the last station finished either shipped or was scrapped."""
    result = simulate(LineConfig.from_json(LINE_FILE))
    last = result.stations[-1]
    assert result.completed == last.completed - last.scrapped
    assert result.departures == result.completed + result.scrapped
    assert len(result.departure_flow_times) == result.departures


def test_throughput_never_beats_the_bottleneck_ceiling() -> None:
    config = LineConfig.from_json(LINE_FILE)
    result = simulate(config)
    assert result.throughput_per_hour <= config.theoretical_rate
    assert 0.0 < result.line_efficiency <= 1.0


# --- the seeding and stream structure ------------------------------------

def test_a_seed_reproduces_a_run_exactly() -> None:
    config = LineConfig.from_json(LINE_FILE)
    a = simulate(config, seed=5)
    b = simulate(config, seed=5)
    c = simulate(config, seed=6)
    assert a.throughput_per_hour == b.throughput_per_hour
    assert a.flow_times == b.flow_times
    assert a.throughput_per_hour != c.throughput_per_hour


def test_streams_are_independent_between_stations() -> None:
    """Changing one station must not reshuffle another station's failures.

    This is the property the paired comparisons depend on. With one shared
    generator it does not hold: a faster station takes a different number of
    draws, every later draw shifts, and the measured difference is partly
    that reshuffle.
    """
    config = LineConfig.from_json(LINE_FILE)
    base = simulate(config, seed=11)
    # Speed up station 0 only.
    changed = simulate(config.with_station(0, mean_cycle=20.0), seed=11)
    for index in range(1, len(config.stations)):
        assert base.stations[index].failures == changed.stations[index].failures, \
            index


def test_streams_differ_by_station_and_by_purpose() -> None:
    streams = station_streams(123, 3)
    draws = [s[name].random() for s in streams for name in ("cycle", "ttf")]
    assert len(set(draws)) == len(draws), draws


# --- the experiments ------------------------------------------------------

def test_more_buffer_never_reduces_throughput() -> None:
    """Monotone by construction: extra buffer cannot make a line worse.

    It is the sharpest available check that the sweep is measuring the effect
    of buffer rather than run-to-run noise -- without common random numbers
    this test fails, which is how the need for them showed up.
    """
    config = deterministic([30.0, 34.0, 31.0], hours=20.0)
    config = config.with_station(0, cycle_cv=0.3).with_station(1, cycle_cv=0.3)
    config = config.with_station(2, cycle_cv=0.3)
    points = ex.buffer_sweep(config, capacities=[1, 2, 4, 8, 16],
                             replications=5)
    rates = [p.throughput.mean for p in points]
    for lower, higher in zip(rates, rates[1:]):
        assert higher >= lower - 1e-9, rates
    assert rates[-1] > rates[0], rates


def test_improving_a_non_bottleneck_changes_nothing() -> None:
    """Goldratt, as an assertion.

    The bottleneck gain must be significant and every non-bottleneck gain
    must not be -- on a line with one clear bottleneck and enough slack
    elsewhere.
    """
    config = deterministic([30.0, 50.0, 32.0], hours=20.0)
    config = config.with_station(1, cycle_cv=0.2)
    _, points = ex.improvement_sensitivity(config, factor=0.9, replications=5)
    by_name = {p.station: p for p in points}
    assert by_name["S1"].is_bottleneck
    assert by_name["S1"].significant, by_name["S1"].difference
    assert by_name["S1"].delta > 1.0, by_name["S1"].delta
    for name in ("S0", "S2"):
        assert abs(by_name[name].delta) < 0.5, (name, by_name[name].delta)


def test_the_saving_at_a_non_bottleneck_turns_into_starvation() -> None:
    """Where the removed work goes: idle time, not output.

    Not a statistical claim -- a direct accounting one. Cut a non-bottleneck's
    cycle time and its busy time falls by that much, its starved time rises by
    the same amount, and the part count does not move.
    """
    config = LineConfig.from_json(LINE_FILE)
    base = simulate(config, seed=20261031)
    faster = simulate(config.with_station(2, mean_cycle=41.0 * 0.9),
                      seed=20261031)
    saved = base.stations[2].busy - faster.stations[2].busy
    extra_idle = faster.stations[2].starved - base.stations[2].starved
    assert saved > 1500, saved
    assert abs(extra_idle - saved) < 50, (saved, extra_idle)
    assert faster.completed == base.completed


def test_knee_reports_the_gain_actually_captured() -> None:
    config = deterministic([30.0, 34.0, 31.0], hours=10.0)
    config = config.with_station(1, cycle_cv=0.3)
    points = ex.buffer_sweep(config, capacities=[1, 2, 4, 8], replications=4)
    chosen = ex.knee(points, capture=0.8)
    low = min(p.throughput.mean for p in points)
    high = max(p.throughput.mean for p in points)
    assert chosen.throughput.mean >= low + 0.8 * (high - low)
    # And it must be the smallest capacity that clears the bar.
    smaller = [p for p in points if p.capacity < chosen.capacity]
    for point in smaller:
        assert point.throughput.mean < low + 0.8 * (high - low)


def test_marginal_gain_is_the_slope_between_sweep_points() -> None:
    config = deterministic([30.0, 34.0], hours=5.0)
    config = config.with_station(1, cycle_cv=0.3)
    points = ex.buffer_sweep(config, capacities=[1, 4], replications=3)
    (capacity, slope), = ex.marginal_gain(points)
    expected = ((points[1].throughput.mean - points[0].throughput.mean)
                / (points[1].wip.mean - points[0].wip.mean))
    assert capacity == 4
    assert abs(slope - expected) < 1e-12


# --- statistics -----------------------------------------------------------

def test_confidence_interval_matches_the_textbook_formula() -> None:
    """Mean 3, s = 1.5811, n = 5, t(4) = 2.776 -> half-width 1.9630."""
    interval = metrics.confidence([1.0, 2.0, 3.0, 4.0, 5.0])
    assert abs(interval.mean - 3.0) < 1e-12
    assert abs(interval.half_width - 2.776 * 1.5811388 / 5**0.5) < 1e-5
    assert interval.n == 5


def test_small_samples_use_t_not_1_96() -> None:
    """On five replications a normal interval is a quarter too narrow."""
    assert metrics.t95(4) == 2.776
    assert metrics.t95(100) == 2.042  # falls back to the largest df tabulated


def test_paired_difference_cancels_shared_variation() -> None:
    """Two samples that differ by a constant have a zero-width interval.

    The point of pairing: the run-to-run variation is identical in both
    samples, so the difference is the effect and nothing else.
    """
    baseline = [55.0, 61.0, 58.0, 64.0, 59.0]
    treated = [v + 3.0 for v in baseline]
    paired = metrics.paired_difference(treated, baseline)
    assert abs(paired.mean - 3.0) < 1e-12
    assert paired.half_width < 1e-9
    # The unpaired comparison cannot see it at all.
    assert metrics.confidence(treated).overlaps(metrics.confidence(baseline))


def test_paired_difference_rejects_mismatched_samples() -> None:
    try:
        metrics.paired_difference([1.0, 2.0], [1.0])
    except ValueError:
        return
    raise AssertionError("accepted a paired comparison of unequal samples")


def test_intervals_know_when_they_overlap() -> None:
    a = metrics.Interval(10.0, 1.0, 5)
    b = metrics.Interval(11.5, 1.0, 5)
    c = metrics.Interval(20.0, 1.0, 5)
    assert a.overlaps(b) and b.overlaps(a)
    assert not a.overlaps(c)


# --- configuration --------------------------------------------------------

def test_buffer_count_must_match_the_station_count() -> None:
    config = deterministic([10.0, 10.0, 10.0])
    assert len(config.with_buffers([1, 2]).buffers()) == 2
    try:
        config.with_buffers([1, 2, 3]).buffers()
    except ValueError:
        return
    raise AssertionError("accepted 3 buffers for 3 stations")


def test_unknown_keys_in_the_line_file_are_rejected() -> None:
    """A misspelt MTBF must fail loudly, not be silently dropped."""
    import json
    import tempfile
    raw = json.loads(LINE_FILE.read_text(encoding="utf-8"))
    raw["stations"][0]["mtbf"] = 5000  # the real field is mtbf_operating
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        try:
            LineConfig.from_json(path)
        except ValueError as exc:
            assert "mtbf" in str(exc)
            return
    raise AssertionError("accepted an unknown station key")


def test_the_shipped_line_file_loads_and_has_one_clear_bottleneck() -> None:
    config = LineConfig.from_json(LINE_FILE)
    assert len(config.stations) == 5
    assert config.bottleneck.name == "OP20 Mill"
    others = [s.effective_cycle for s in config.stations
              if s.name != config.bottleneck.name]
    assert config.bottleneck.effective_cycle > max(others) * 1.2


def test_warmup_is_discarded() -> None:
    """Without it the empty line filling up is averaged into the answer."""
    cold = simulate(deterministic([10.0, 20.0], hours=1.0, warmup=0.0))
    warm = simulate(deterministic([10.0, 20.0], hours=1.0, warmup=1.0))
    assert warm.throughput_per_hour > cold.throughput_per_hour
    assert abs(warm.throughput_per_hour - 180.0) < 0.1


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {test.__name__}  {exc}")
        else:
            print(f"pass  {test.__name__}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
