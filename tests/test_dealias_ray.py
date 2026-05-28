"""Tests for the in-house ray-by-ray phase-unwrap dealiaser.

The algorithm processes each ray with its own Nyquist value so it
handles PRF-staggered sweeps that PyART's dealiasers refuse. These
tests exercise the unwrapping math directly via lightweight synthetic
radars built without PyART — we don't need the full file pipeline to
verify the per-ray phase-unwrap logic.
"""

from __future__ import annotations

import numpy as np
import pytest

from radar_warning_game.data.dealias_ray import dealias_ray_by_ray


class _Radar:
    """Minimal PyART-radar stand-in: only the bits ``dealias_ray_by_ray`` reads.

    ``sweep_starts``, when provided, lets a test split its rays into
    multiple sweeps — necessary for cases that want to test per-sweep
    behavior (e.g. each ray in its own sweep gets its own Nyquist
    honored, since the disuniform-sweep derivation only fires within
    a single sweep). Without it, all rays land in one big sweep.

    ``reflectivity`` lets a test set up a Z field; when present, the
    dealiaser uses it to mask sub-20 dBZ gates from the unwrap.
    """

    def __init__(
        self,
        velocity: np.ndarray,
        nyquist_per_ray: np.ndarray | None,
        mask: np.ndarray | None = None,
        sweep_starts: list[int] | None = None,
        reflectivity: np.ndarray | None = None,
    ) -> None:
        if mask is None:
            data = np.ma.array(velocity, mask=np.zeros_like(velocity, dtype=bool))
        else:
            data = np.ma.array(velocity, mask=mask)
        self.fields = {"velocity": {"data": data, "units": "meters_per_second"}}
        if reflectivity is not None:
            self.fields["reflectivity"] = {
                "data": np.ma.array(reflectivity, mask=np.zeros_like(reflectivity, dtype=bool)),
                "units": "dBZ",
            }
        if nyquist_per_ray is None:
            self.instrument_parameters = None
        else:
            self.instrument_parameters = {
                "nyquist_velocity": {"data": np.asarray(nyquist_per_ray, dtype=np.float64)},
            }
        if sweep_starts is not None:
            nrays = velocity.shape[0]
            starts = list(sweep_starts)
            ends = [(starts[i + 1] - 1) if i + 1 < len(starts) else nrays - 1
                    for i in range(len(starts))]
            self.nsweeps = len(starts)
            self.sweep_start_ray_index = {"data": np.asarray(starts, dtype=int)}
            self.sweep_end_ray_index = {"data": np.asarray(ends, dtype=int)}


def _fold(v: np.ndarray, nyq: float) -> np.ndarray:
    """Fold a true velocity to the [-V_ny, V_ny] interval the radar
    actually measures. Inverse of what the dealiaser must undo."""
    return ((v + nyq) % (2 * nyq)) - nyq


# ---- basic single-ray correctness ------------------------------------

def test_unwraps_one_fold_at_nyquist_edge():
    """A ray that rolls smoothly from +V_ny−ε through +V_ny back into
    -V_ny+ε should be reconstructed back into a continuous +V_ny
    plateau on the dealiased side."""
    nyq = 25.0
    true_v = np.array([23.0, 24.0, 26.0, 28.0, 30.0])
    folded = _fold(true_v, nyq)
    # Sanity: the middle samples are now in the negative side
    assert folded[2] < 0
    radar = _Radar(folded[None, :], np.array([nyq]))
    out = dealias_ray_by_ray(radar)["data"]
    assert np.allclose(np.asarray(out)[0], true_v, atol=0.1)


def test_unwraps_two_folds():
    """Two cumulative folds within a single ray should both unwind."""
    nyq = 25.0
    true_v = np.array([10.0, 20.0, 30.0, 45.0, 55.0, 65.0, 75.0])
    folded = _fold(true_v, nyq)
    radar = _Radar(folded[None, :], np.array([nyq]))
    out = np.asarray(dealias_ray_by_ray(radar)["data"])[0]
    assert np.allclose(out, true_v, atol=0.1)


def test_unwraps_negative_folds():
    """Symmetric: a ray that progresses below -V_ny gets reconstructed
    upward into the correct negative values."""
    nyq = 25.0
    true_v = np.array([-10.0, -20.0, -35.0, -45.0, -55.0])
    folded = _fold(true_v, nyq)
    radar = _Radar(folded[None, :], np.array([nyq]))
    out = np.asarray(dealias_ray_by_ray(radar)["data"])[0]
    assert np.allclose(out, true_v, atol=0.1)


# ---- per-ray Nyquist (the whole point) ---------------------------------

def test_two_sweeps_with_different_nyquists():
    """Per-sweep Nyquists are still honored: a radar with two
    SWEEPS at different Nyquists dealiases each sweep against its
    own value. (The cross-ray-within-sweep case is handled by the
    disuniform-sweep derivation in a separate test.)"""
    nyq_per_ray = np.array([25.0, 35.0])
    true_v = np.array([
        [10.0, 20.0, 30.0, 40.0],   # sweep 0 (V_ny=25): crosses ±25
        [10.0, 20.0, 30.0, 34.0],   # sweep 1 (V_ny=35): stays within ±35
    ])
    folded = np.array([_fold(true_v[0], 25.0), _fold(true_v[1], 35.0)])
    # Each ray is its own sweep → each gets uniform-Nyquist treatment.
    radar = _Radar(folded, nyq_per_ray, sweep_starts=[0, 1])
    out = np.asarray(dealias_ray_by_ray(radar)["data"])
    assert np.allclose(out, true_v, atol=0.1)


def test_phase_unwrap_anchored_at_first_gate():
    """Phase-unwrap dealiasing reconstructs velocity by accumulating
    fold-step indicators along the ray, with fold count 0 at the
    first gate. A ray that starts already-folded (no diff inside the
    ray itself reveals it) reconstructs to the wrong absolute value
    by design — this is a fundamental limitation shared with PyART's
    1D unwrap. We pin this so a future "improvement" doesn't claim
    to solve it without an inter-ray anchor."""
    nyq = 25.0
    # Constant 50 m/s — folded to constant 0 m/s. No diff signal to
    # detect a fold. Output stays at 0.
    folded = np.zeros((1, 5))
    radar = _Radar(folded, np.array([nyq]))
    out = np.asarray(dealias_ray_by_ray(radar)["data"])
    assert np.allclose(out, folded, atol=0.1)


def test_negative_metadata_nyquist_is_taken_as_magnitude():
    """A ray with a stored Nyquist of -25 m/s should be treated as 25
    m/s — sign-flipped metadata shouldn't reverse the unfold direction."""
    nyq = 25.0
    true_v = np.array([20.0, 30.0, 45.0])
    folded = _fold(true_v, nyq)
    radar = _Radar(folded[None, :], np.array([-nyq]))   # negative metadata
    out = np.asarray(dealias_ray_by_ray(radar)["data"])[0]
    assert np.allclose(out, true_v, atol=0.1)


# ---- masked-gate handling ---------------------------------------------

def test_masked_gaps_dont_create_phantom_folds():
    """A masked gate in the middle of a ray must not look like a huge
    velocity jump to the unwrapper. The forward-fill ensures the
    diff across the mask is zero, so no fold is registered there."""
    nyq = 25.0
    true_v = np.array([15.0, 20.0, 22.0, 24.0, 22.0])
    folded = _fold(true_v, nyq)
    mask = np.array([False, False, True, False, False])
    radar = _Radar(folded[None, :], np.array([nyq]), mask=mask[None, :])
    out_masked = dealias_ray_by_ray(radar)["data"]
    out = np.asarray(out_masked)
    # Unmasked gates should match the true values
    assert np.allclose(out[0, [0, 1, 3, 4]], true_v[[0, 1, 3, 4]], atol=0.1)
    # The mask itself must survive the round-trip
    assert np.ma.getmaskarray(out_masked)[0, 2] == True


def test_zero_nyquist_ray_passes_through_unchanged():
    """A ray whose stored Nyquist is 0 (degenerate metadata) should
    fall through to raw velocity — no fold detection. Otherwise we'd
    divide by zero or produce garbage."""
    nyq_per_ray = np.array([25.0, 0.0])
    true_v = np.array([[10.0, 20.0, 30.0], [5.0, 7.0, 9.0]])
    folded = np.array([_fold(true_v[0], 25.0), true_v[1]])
    radar = _Radar(folded, nyq_per_ray)
    out = np.asarray(dealias_ray_by_ray(radar)["data"])
    # Ray 0 dealiased
    assert np.allclose(out[0], true_v[0], atol=0.1)
    # Ray 1 unchanged (still the raw, "folded" — same as true here)
    assert np.allclose(out[1], true_v[1], atol=0.1)


# ---- edge cases -------------------------------------------------------

def test_no_velocity_field_raises():
    radar = _Radar(np.array([[10.0]]), np.array([25.0]))
    del radar.fields["velocity"]
    with pytest.raises(KeyError):
        dealias_ray_by_ray(radar)


def test_missing_nyquist_returns_raw():
    """When no instrument_parameters at all, the dealiaser logs and
    returns the raw velocities verbatim — never crashes."""
    radar = _Radar(np.array([[10.0, 20.0, 30.0]]), None)
    field = dealias_ray_by_ray(radar)
    assert np.allclose(np.asarray(field["data"]), [[10.0, 20.0, 30.0]])


def test_short_ray_handled():
    """1-gate rays have no diff to compute. Output equals input."""
    radar = _Radar(np.array([[42.0]]), np.array([25.0]))
    out = np.asarray(dealias_ray_by_ray(radar)["data"])
    assert out.shape == (1, 1)
    assert np.allclose(out, [[42.0]])


# ---- single-gate fold-in-precip dealiases correctly --------------------

def test_single_gate_fold_in_high_z_is_corrected():
    """Inside a real high-reflectivity precipitation cell a single
    aliased gate (sandwiched between unfolded neighbors) used to be
    masked by the generic spike filter and so kept its raw folded
    value in the output. The generic spike filter is gone now — only
    the sub-20-dBZ mask runs — so a high-Z single-gate fold flows
    through the cumulative-fold counter normally: the count bumps +1
    at the folded gate, then back to 0 at the next gate, so the
    correction is applied to that one gate and downstream values
    are untouched."""
    nyq = 25.0
    # Storm cell at +20 m/s with one gate truly at +28 (folds to -22
    # at Nyq=25). All gates are 40 dBZ (precipitation, not noise).
    true_v = np.array([20.0, 18.0, 28.0, 20.0, 22.0])
    folded = np.array([true_v[0], true_v[1], _fold(true_v[2:3], nyq)[0],
                       true_v[3], true_v[4]])
    z = np.full_like(folded, 40.0)
    radar = _Radar(folded[None, :], np.array([nyq]),
                   reflectivity=z[None, :])
    out = np.asarray(dealias_ray_by_ray(radar)["data"])[0]
    # The single folded gate is restored to +28.
    assert out[2] == pytest.approx(28.0, abs=0.1)
    # Adjacent gates stay at their original values (no spurious propagation).
    assert np.allclose(out[[0, 1, 3, 4]], true_v[[0, 1, 3, 4]], atol=0.1)


# ---- local-window azimuthal consensus --------------------------------

def test_local_consensus_corrects_bulk_aliased_ray():
    """A whole ray of constant folded velocity has no within-ray
    gradient — the radial pass sees diffs of zero and detects no
    folds, leaving the ray uncorrected. The local-window
    consensus pass picks up the slack: each gate of the folded
    ray sees its same-range azimuthal neighbors at the unfolded
    value (median = -20), notices it's ~2·Nyq off, and snaps.

    Scenario: 5 adjacent rays, 3 gates each. The middle ray's
    every gate is at +24 (true value -26, folded at Nyq=25); the
    other four rays are uniformly at -20 (no fold).
    """
    nyq = 25.0
    raw = np.array([
        [-20.0, -20.0, -20.0],
        [-20.0, -20.0, -20.0],
        [+24.0, +24.0, +24.0],   # whole ray folded; no within-ray gradient
        [-20.0, -20.0, -20.0],
        [-20.0, -20.0, -20.0],
    ])
    radar = _Radar(raw, np.full(5, nyq))
    out = np.asarray(dealias_ray_by_ray(radar)["data"])
    # The folded ray is lifted to -26 across all of its gates.
    assert np.allclose(out[2], -26.0, atol=0.1)
    # Unfolded neighbor rays unchanged.
    assert np.allclose(out[[0, 1, 3, 4]], -20.0, atol=0.1)


def test_local_consensus_skips_across_prf_boundaries():
    """The same-Nyq filter in the local-window median means a ray
    sandwiched between two different-Nyq neighbors won't be
    snapped by either — there's no consensus to use that's on the
    same PRF footing. Pin that the value stays untouched in this
    case."""
    nyq = np.array([25.0, 25.0, 35.0])   # disuniform
    raw = np.array([
        [-20.0, -20.0],
        [+24.0, +24.0],
        [-20.0, -20.0],
    ])
    radar = _Radar(raw, nyq)
    out = np.asarray(dealias_ray_by_ray(radar)["data"])
    # Ray 1's window contains ray 0 (Nyq=25) and ray 2 (Nyq=35);
    # only ray 0 matches its own Nyq=25, leaving fewer than two
    # same-Nyq neighbors → snap skips → +24 retained.
    assert np.allclose(out[1], +24.0, atol=0.1)


def test_local_consensus_does_not_propagate_around_the_sweep():
    """Regression for the constant-range-arc artifact: a single
    false-fold detection in the radial pass for one ray must NOT
    propagate ±2·Nyq across the whole sweep. With a local-window
    consensus, propagation is bounded to roughly ±5° of the
    triggering ray — anything farther around the sweep is
    untouched by the snap pass.

    Scenario: 360 rays at 1° spacing, all at -20 m/s. Ray 50 is
    artificially pre-shifted by +50 m/s (a faux 1D cascade).
    Snap should pull ray 50 back into agreement with its nearby
    neighbors (rays 45-55) but rays 180-190 (~half a sweep away)
    are untouched.
    """
    nyq = 25.0
    nrays = 360
    raw = np.full((nrays, 3), -20.0)
    raw[50] = +30.0   # +50 offset cascade on a single ray
    az = np.arange(nrays, dtype=np.float64) * (360.0 / nrays)
    radar = _Radar(raw, np.full(nrays, nyq))
    radar.azimuth = {"data": az}
    out = np.asarray(dealias_ray_by_ray(radar)["data"])
    # Ray 50 is corrected by its nearby neighbors.
    assert np.allclose(out[50], -20.0, atol=0.1)
    # Far-side rays are untouched (still at -20).
    assert np.allclose(out[180:190], -20.0, atol=0.1)


# ---- inter-ray cascade correction -------------------------------------

def test_false_fold_cascade_is_snapped_back_by_neighbors():
    """1D phase unwrap is vulnerable to ONE spurious fold detection
    cascading ±2·Nyq across every downrange gate of that ray. The
    inter-ray sanity pass catches this by comparing each gate to
    its same-range azimuthal neighbors: when both neighbors agree
    on a value and the current gate is off by ~2·Nyq, the cascade
    is snapped back.

    Scenario: three adjacent rays. The middle ray has been
    erroneously offset by +2·Nyq from gate 2 onward (simulating a
    cascade); both neighbors carry the correct value. The
    correction should pull the middle ray's downrange gates back
    into agreement."""
    nyq = 25.0
    correct = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
    cascaded = np.array([5.0, 5.0, 55.0, 55.0, 55.0])   # off by +2·Nyq from gate 2
    raw = np.stack([correct, cascaded, correct], axis=0)
    radar = _Radar(raw, np.array([nyq, nyq, nyq]))
    out = np.asarray(dealias_ray_by_ray(radar)["data"])
    # Middle ray should be snapped back to ~5 from gate 2 onward.
    assert np.allclose(out[1], correct, atol=0.1)
    # Outer rays untouched.
    assert np.allclose(out[0], correct, atol=0.1)
    assert np.allclose(out[2], correct, atol=0.1)


def test_real_azimuthal_gradient_is_not_snapped():
    """A real velocity field with an azimuthal gradient (modest,
    < Nyq from ray to ray) must NOT be flattened by the snap pass.
    Pin that adjacent rays at +10 → +20 → +30 (Nyq=25) stay at
    those values."""
    nyq = 25.0
    rays = np.array([
        [10.0, 10.0, 10.0],
        [20.0, 20.0, 20.0],
        [30.0, 30.0, 30.0],
    ])
    radar = _Radar(rays, np.array([nyq, nyq, nyq]))
    out = np.asarray(dealias_ray_by_ray(radar)["data"])
    assert np.allclose(out, rays, atol=0.1)


# ---- per-ray Nyquist derivation for disuniform sweeps -----------------

def test_disuniform_sweep_dealiases_per_ray_with_own_nyquist():
    """Disuniform per-ray stated Nyquists are honored ray-by-ray —
    each ray uses its own value as the unfold interval, regardless
    of what neighboring rays say. The per-ray value is what the
    radar physically used to fold that ray's data, so it's the
    right thing to use for unwrap. Uniformity across the sweep is
    irrelevant to the 1D-along-the-ray algorithm."""
    stated = np.array([25.0, 35.0])
    # Ray 0 (Nyq=25): true v ramps from 10 → 40, fold at gate 3.
    # Ray 1 (Nyq=35): true v ramps from 10 → 34, no fold (within ±35).
    true_v = np.array([
        [10.0, 20.0, 30.0, 40.0],
        [10.0, 20.0, 30.0, 34.0],
    ])
    folded = np.array([_fold(true_v[0], 25.0), _fold(true_v[1], 35.0)])
    radar = _Radar(folded, stated)
    out = np.asarray(dealias_ray_by_ray(radar)["data"])
    assert np.allclose(out, true_v, atol=0.1)


def test_uniform_sweep_keeps_stated_nyquist_untouched():
    """Uniform-stated sweeps must NOT trigger the derive-from-observation
    path — that would slow things down and risk widening the Nyquist
    away from the radar's authoritative figure on healthy data. Pin
    that a perfectly uniform sweep dealiases identically to the
    stated-Nyquist case."""
    nyq = 25.0
    true_v = np.array([
        [10.0, 20.0, 30.0, 40.0],
        [-10.0, -20.0, -30.0, -40.0],
    ])
    folded = np.array([_fold(true_v[0], nyq), _fold(true_v[1], nyq)])
    stated = np.array([nyq, nyq])
    radar = _Radar(folded, stated)
    out = np.asarray(dealias_ray_by_ray(radar)["data"])
    assert np.allclose(out, true_v, atol=0.1)


# ---- low-reflectivity spike filter ------------------------------------

def test_low_reflectivity_gate_does_not_anchor_false_fold():
    """A low-Z gate (bird / insect / noise) sandwiched in the middle
    of a ray used to anchor the cumulative fold count if its value
    happened to look like a fold transition. With the sub-20-dBZ
    aggressive spike filter that gate is now transparent for the
    unwrap, so downstream gates stay aligned with the real
    meteorology."""
    nyq = 25.0
    # Surrounding gates show a low real velocity (consistent storm
    # at +5 m/s) but one mid-ray gate has a noise spike at -20 m/s
    # (would-be fold trigger) AND low reflectivity.
    ray_v = np.array([5.0, 5.0, -20.0, 5.0, 5.0, 5.0])
    ray_z = np.array([35.0, 35.0, 12.0, 35.0, 35.0, 35.0])   # noise gate at 12 dBZ
    radar = _Radar(ray_v[None, :], np.array([nyq]),
                   reflectivity=ray_z[None, :])
    out = np.asarray(dealias_ray_by_ray(radar)["data"])[0]
    # Gates after the noise spike should still read ~+5, not be
    # shifted by 2·Nyq=50 m/s as the simple unwrap would do.
    assert np.all(np.abs(out[3:] - 5.0) < 1.0)


def test_high_reflectivity_real_fold_still_unwraps():
    """The low-Z filter must not interfere with high-Z real folds.
    A genuine velocity ramp through the Nyquist edge — with full
    reflectivity throughout — still unfolds correctly."""
    nyq = 25.0
    true_v = np.array([20.0, 24.0, 30.0, 40.0])
    folded = _fold(true_v, nyq)
    # All gates above 20 dBZ → unwrap engages on the whole ray.
    z = np.full_like(folded, 40.0)
    radar = _Radar(folded[None, :], np.array([nyq]),
                   reflectivity=z[None, :])
    out = np.asarray(dealias_ray_by_ray(radar)["data"])[0]
    assert np.allclose(out, true_v, atol=0.1)


def test_no_reflectivity_field_does_not_crash():
    """Radar objects without a reflectivity field still dealias —
    the low-Z filter just becomes a no-op."""
    nyq = 25.0
    true_v = np.array([10.0, 20.0, 30.0])
    folded = _fold(true_v, nyq)
    radar = _Radar(folded[None, :], np.array([nyq]))   # no reflectivity
    assert "reflectivity" not in radar.fields
    out = np.asarray(dealias_ray_by_ray(radar)["data"])[0]
    assert np.allclose(out, true_v, atol=0.1)


def test_low_z_gate_keeps_its_own_velocity_in_output():
    """The aggressive filter masks low-Z gates for the **unwrap pass
    only** — their actual measured velocity must still appear in the
    output (with whatever fold count the unwrap reached by that
    point). Previously the carry-forward value from the previous
    valid gate got written into the spike position, erasing the
    user-visible reading."""
    nyq = 25.0
    # Mid-ray noise spike at -15 (low Z); surrounding storm at +5.
    ray_v = np.array([5.0, 5.0, -15.0, 5.0, 5.0])
    ray_z = np.array([35.0, 35.0, 8.0, 35.0, 35.0])
    radar = _Radar(ray_v[None, :], np.array([nyq]),
                   reflectivity=ray_z[None, :])
    out = np.asarray(dealias_ray_by_ray(radar)["data"])[0]
    # The low-Z gate at index 2 must still read its own -15 value
    # (no folds happened by that point — the unwrap walked
    # transparently across it).
    assert out[2] == pytest.approx(-15.0)


def test_low_z_does_not_change_user_visible_mask():
    """The aggressive filter must mask the unwrap pass only — the
    *output* mask is the original input mask, so the user still sees
    the low-Z gate (just at its raw, un-corrected velocity)."""
    nyq = 25.0
    ray_v = np.array([5.0, 5.0, 5.0, 5.0])
    ray_z = np.array([35.0, 5.0, 35.0, 35.0])   # one sub-20-dBZ gate
    radar = _Radar(ray_v[None, :], np.array([nyq]),
                   reflectivity=ray_z[None, :])
    out_field = dealias_ray_by_ray(radar)
    out = out_field["data"]
    # The output mask reflects the INPUT mask (all-unmasked here),
    # not the augmented unwrap mask.
    assert isinstance(out, np.ma.MaskedArray)
    assert not np.any(np.ma.getmaskarray(out))


def test_output_is_pyart_compatible_field_dict():
    """The returned dict mirrors the shape PyART's dealiasers produce
    so the caller can ``radar.add_field("corrected_velocity", ...)``
    without special-casing the new dealiaser."""
    nyq = 25.0
    radar = _Radar(np.array([[10.0, 20.0]]), np.array([nyq]))
    field = dealias_ray_by_ray(radar)
    assert "data" in field
    assert field.get("units") == "meters_per_second"
    assert "standard_name" in field
    assert isinstance(field["data"], np.ma.MaskedArray)
    assert field["data"].shape == (1, 2)
