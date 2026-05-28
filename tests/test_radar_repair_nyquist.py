"""Tests for ``radar_repair.ensure_nyquist_velocity``.

The function repairs ``radar.instrument_parameters['nyquist_velocity']``
so PyART's region-based dealiaser doesn't trip on bad/missing/non-
uniform values. Two paths matter:

  - **bad data** (missing / all-zero / non-finite): derive per-sweep
    Nyquist from observed velocities (``max(|v|) × safety_margin``).
  - **non-uniform within a sweep** (PRF stagger): use the lowest non-
    zero stated Nyquist for that sweep — a conservative choice that
    matches the slowest PRF the radar recorded, and doesn't overshoot
    on archives where velocities had already been dealiased.

We don't need a real PyART NEXRAD file for any of this; a lightweight
synthetic-radar harness exercises the array logic directly.
"""

from __future__ import annotations

import numpy as np

from radar_warning_game.data.radar_repair import (
    _DEFAULT_NYQUIST_MS,
    _NYQUIST_SAFETY_MARGIN,
    ensure_nyquist_velocity,
)


class _Radar:
    """Minimal PyART-radar stand-in for the bits ``ensure_nyquist_velocity`` reads."""

    def __init__(
        self,
        *,
        nyquists: list[list[float]] | None,
        velocities: list[list[float]],
    ) -> None:
        # nyquists is per-sweep per-ray; None means "instrument_parameters absent".
        sweeps = velocities
        self.nsweeps = len(sweeps)
        rays_per_sweep = [len(s) for s in sweeps]
        starts: list[int] = []
        ends: list[int] = []
        cursor = 0
        for n in rays_per_sweep:
            starts.append(cursor)
            ends.append(cursor + n - 1)
            cursor += n
        self.nrays = cursor
        self.sweep_start_ray_index = {"data": np.asarray(starts, dtype=int)}
        self.sweep_end_ray_index = {"data": np.asarray(ends, dtype=int)}
        flat_v = np.concatenate([np.asarray(s, dtype=float) for s in sweeps])
        self.fields = {"velocity": {"data": flat_v}}
        if nyquists is not None:
            flat_ny = np.concatenate([np.asarray(s, dtype=float) for s in nyquists])
            self.instrument_parameters = {
                "nyquist_velocity": {"data": flat_ny},
            }
        else:
            self.instrument_parameters = None


def _nyquist_array(radar: _Radar) -> np.ndarray:
    return np.asarray(radar.instrument_parameters["nyquist_velocity"]["data"],
                      dtype=np.float64)


# ---- non-uniform: lowest non-zero per-ray Nyquist -----------------------

def test_nonuniform_sweep_uses_lowest_nonzero_stated_nyquist():
    """A sweep with mixed Nyquist (PRF stagger) gets repaired to a
    uniform value equal to the *lowest non-zero* stated value. Bigger
    values from the same sweep don't sneak in, and we don't fall back
    to a velocity-derived estimate."""
    sweep_nyquists = [25.0, 35.0, 25.0, 35.0]   # dual-PRF stagger
    sweep_velocities = [-50.0, 30.0, 90.0, -10.0]   # already-dealiased, |v|=90
    radar = _Radar(
        nyquists=[sweep_nyquists],
        velocities=[sweep_velocities],
    )
    repaired = ensure_nyquist_velocity(radar)
    assert repaired is True
    repaired_arr = _nyquist_array(radar)
    assert repaired_arr.shape == (4,)
    assert np.allclose(repaired_arr, 25.0), (
        "non-uniform repair should pick the lowest non-zero stated "
        "value (25), not derive from observed |v|=90"
    )


def test_nonuniform_with_some_zero_rays_ignores_the_zeros():
    """If a non-uniform sweep mixes zeros with valid PRF values, the
    repair picks the lowest *non-zero* value — zeros would be the
    "bad data" signal, not the slowest PRF."""
    sweep_nyquists = [0.0, 35.0, 28.0, 35.0]
    radar = _Radar(
        nyquists=[sweep_nyquists],
        velocities=[[10.0, 12.0, 8.0, 11.0]],
    )
    assert ensure_nyquist_velocity(radar) is True
    assert np.allclose(_nyquist_array(radar), 28.0)


def test_uniform_good_data_is_left_alone():
    """Uniform per-sweep Nyquists (within float tolerance) don't
    trigger repair at all."""
    radar = _Radar(
        nyquists=[[25.0, 25.0, 25.0]],
        velocities=[[10.0, 12.0, 8.0]],
    )
    repaired = ensure_nyquist_velocity(radar)
    assert repaired is False


# ---- bad-data path: velocity-derived -------------------------------------

def test_all_zero_nyquists_fall_back_to_velocity_derived():
    """When every ray in a sweep has Nyquist=0 (bad metadata), the
    repair must use ``max(|v|) × safety_margin`` — there's no stated
    value worth trusting."""
    radar = _Radar(
        nyquists=[[0.0, 0.0, 0.0, 0.0]],
        velocities=[[-32.0, 18.0, -25.0, 4.0]],
    )
    assert ensure_nyquist_velocity(radar) is True
    expected = 32.0 * _NYQUIST_SAFETY_MARGIN
    assert np.allclose(_nyquist_array(radar), expected, atol=1e-3)


def test_missing_instrument_parameters_triggers_velocity_derived():
    """No ``instrument_parameters`` at all → bad-data path."""
    radar = _Radar(
        nyquists=None,
        velocities=[[20.0, 22.0, -18.0]],
    )
    assert ensure_nyquist_velocity(radar) is True
    arr = _nyquist_array(radar)
    assert np.allclose(arr, 22.0 * _NYQUIST_SAFETY_MARGIN, atol=1e-3)


def test_clear_air_sweep_uses_default_floor():
    """A sweep whose observed |v| is sub-1 m/s (clear-air noise) is
    too weak to derive a Nyquist from — fall back to the default
    floor so the dealias algorithm doesn't see a fake 0.5 m/s
    unambiguous range."""
    radar = _Radar(
        nyquists=[[0.0, 0.0, 0.0]],
        velocities=[[0.2, -0.1, 0.3]],
    )
    assert ensure_nyquist_velocity(radar) is True
    assert np.allclose(_nyquist_array(radar), _DEFAULT_NYQUIST_MS)


# ---- negative-Nyquist handling -----------------------------------------

def test_negative_nyquists_are_treated_as_magnitudes():
    """Some archives encode the Nyquist limit with the wrong sign — a
    ``-25`` value still means a 25 m/s unambiguous bound. The repair
    must take the magnitude before picking the lowest non-zero, so
    PyART never gets handed a negative Nyquist (which would unfold
    velocities in the wrong direction)."""
    # Sweep with mixed-sign per-ray Nyquists: +35 and -25. The
    # absolute lowest non-zero is 25 — that's what the repair should
    # store, NOT -25 (negative) and NOT 35 (the larger of the two).
    radar = _Radar(
        nyquists=[[35.0, -25.0, 35.0, -25.0]],
        velocities=[[10.0, -12.0, 8.0, 11.0]],
    )
    assert ensure_nyquist_velocity(radar) is True
    repaired = _nyquist_array(radar)
    assert np.all(repaired > 0.0)
    assert np.allclose(repaired, 25.0)


def test_all_negative_nyquists_still_recover():
    """A sweep whose stated Nyquists are *all* negative was previously
    silently falling through to the velocity-derived path because the
    ``> 0`` filter dropped every ray. Taking abs() first means the
    magnitude is used as-stated."""
    radar = _Radar(
        nyquists=[[-25.0, -35.0, -25.0]],
        velocities=[[20.0, -22.0, 18.0]],
    )
    assert ensure_nyquist_velocity(radar) is True
    repaired = _nyquist_array(radar)
    assert np.allclose(repaired, 25.0)
    assert np.all(repaired > 0.0)


def test_uniform_signed_sweep_is_not_treated_as_nonuniform():
    """A sweep that's effectively uniform when you take the magnitude
    (``[+25, -25]``) should NOT trigger the non-uniform repair — the
    spread check now compares absolute values, so signed-but-uniform
    Nyquists are left alone."""
    radar = _Radar(
        nyquists=[[25.0, -25.0, 25.0]],
        velocities=[[10.0, 12.0, 8.0]],
    )
    repaired = ensure_nyquist_velocity(radar)
    assert repaired is False


# ---- mixed sweeps --------------------------------------------------------

def test_nonuniform_first_sweep_does_not_affect_second_uniform_sweep():
    """The trigger fires globally (any sweep non-uniform → repair the
    whole radar), but each sweep is repaired independently. A
    non-uniform sweep uses its own lowest non-zero value; an all-zero
    sweep on the same radar falls to the velocity-derived branch."""
    radar = _Radar(
        nyquists=[[24.0, 30.0, 24.0], [0.0, 0.0, 0.0]],   # sw0 nonuniform, sw1 bad
        velocities=[[5.0, 8.0, 6.0], [20.0, -15.0, 18.0]],
    )
    assert ensure_nyquist_velocity(radar) is True
    arr = _nyquist_array(radar)
    # Sweep 0: lowest non-zero stated = 24
    assert np.allclose(arr[:3], 24.0)
    # Sweep 1: velocity-derived from max(|v|)=20 × margin
    assert np.allclose(arr[3:], 20.0 * _NYQUIST_SAFETY_MARGIN, atol=1e-3)
