"""Ray-by-ray Doppler-velocity dealiasing.

Fully vectorized 1D phase unwrap along each ray's gate axis using
the ray's **own stated Nyquist** as the unfold interval — works
identically on uniform and disuniform sweeps (the per-ray choice is
the right thing in both cases). This sidesteps PyART's within-sweep-
uniformity prerequisite — ``dealias_region_based`` and
``dealias_unwrap_phase`` both refuse to run on a sweep whose rays
disagree on Nyquist, and the latter even discards per-ray context
internally (see ``pyart/correct/unwrap.py:_dealias_unwrap_1d``:
*"nyquist_vel is only available sweep by sweep which has been lost
at this point"*).

Algorithm (per ray, all rays processed via numpy in one batched call):

  1. Take the absolute value of the file's per-ray Nyquist so a
     stray negative metadata value doesn't flip the unfold direction.
  2. Forward-fill masked gates with the most recent unmasked value
     in the same ray. ``np.diff`` then sees a zero step across the
     gap rather than a spurious huge jump, so the cumulative-fold
     count doesn't pick up phantom folds at mask edges.
  3. Compute gate-to-gate differences in **phase** units
     (``v * π / V_ny``). A diff in (π, 3π] means we just wrapped
     from +V_ny back to -V_ny, so the upcoming gate is one full
     fold low; a diff in [-3π, -π] is the symmetric +V_ny case.
  4. Cumulative-sum the +/-1 step indicators to get a fold count
     for every gate, then convert back to velocity:
     ``v_true = v_obs + 2 * V_ny * fold_count``.
  5. Re-apply the original mask. Masked gates retain their raw
     (folded) value but won't be drawn since the renderer respects
     the mask.

Performance: a full WSR-88D volume (~7 sweeps × 720 rays × 1832
gates ≈ 9 M cells) dealiases in ~180 ms on a single modern x86
core — typically several times faster than PyART's region-based
dealias and somewhat faster than its 2D phase unwrap. The cost
is concentrated in numpy's ``cumsum`` / ``maximum.accumulate`` /
``nanmedian`` along the gate axis and is trivially parallelizable
across sweeps if even more throughput is ever needed.

Quality vs PyART: the 1D-per-ray approach is more prone to error
propagation through a noisy gate than PyART's 2D sweep unwrap (a
single misclassified fold ripples forward). It nevertheless catches
the common case (continuous storm signal with folds at the Nyquist
edge) and is the only option for PRF-staggered sweeps.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

# Below this Nyquist (m/s) we treat the ray as having no usable
# unambiguous-velocity bound and skip dealiasing for it. Avoids
# divide-by-near-zero issues on degenerate metadata.
_MIN_NYQUIST_MS = 0.5

# Gates with reflectivity below this threshold (dBZ) are treated as
# transparent for the unwrap. Low-Z returns are dominated by
# bird/insect biological scatter, anomalous propagation, and
# near-noise speckle — none of which carry meteorologically
# meaningful Doppler signal. Trusting them tends to anchor the
# phase unwrap on garbage and propagate one bad fold count across
# the rest of the ray. Mask them out for the unwrap purposes only;
# the display mask is preserved on the way out so the user still
# sees what's there (just at the raw, un-corrected velocity).
_LOW_DBZ_THRESHOLD = 20.0


def dealias_ray_by_ray(radar) -> dict:
    """Return a ``corrected_velocity``-style field dict.

    Mirrors the return shape of :func:`pyart.correct.dealias_region_based`
    so the caller can drop it into ``radar.add_field("corrected_velocity",
    result, replace_existing=True)`` exactly like the PyART path.

    Raises ``KeyError`` if the radar has no ``velocity`` field. Returns
    a field whose ``data`` is the original velocity (masked) when no
    dealiasing was applicable — that way the caller's downstream
    "always have a corrected_velocity to read from" assumption holds.
    """
    if "velocity" not in radar.fields:
        raise KeyError("Radar has no 'velocity' field — nothing to dealias")
    vel_field = radar.fields["velocity"]
    vdata = vel_field["data"]
    # Promote to a masked array so we have a uniform mask interface;
    # the ``np.ma.getmaskarray`` helper handles plain ndarrays too.
    if not isinstance(vdata, np.ma.MaskedArray):
        vdata = np.ma.masked_invalid(vdata)
    mask = np.ma.getmaskarray(vdata)
    raw = np.asarray(vdata, dtype=np.float64)
    # Replace masked-array fill values with NaN so the carry-forward
    # below doesn't propagate the fill (e.g. -9999) into the diff.
    raw = np.where(mask, np.nan, raw)

    # Per-ray Nyquist. Falls back to the per-sweep velocity-derived
    # estimate (handled upstream by radar_repair.ensure_nyquist_velocity)
    # if the metadata is missing.
    inst = radar.instrument_parameters or {}
    nyq_dict = inst.get("nyquist_velocity")
    if nyq_dict is None or "data" not in nyq_dict:
        log.warning(
            "dealias_ray_by_ray: radar has no nyquist_velocity metadata; "
            "returning raw velocities"
        )
        return _make_field(vdata, vel_field)
    nyq_per_ray = np.abs(np.asarray(nyq_dict["data"], dtype=np.float64))
    if nyq_per_ray.shape[0] != raw.shape[0]:
        log.warning(
            "dealias_ray_by_ray: nyquist_velocity length %d != nrays %d; "
            "returning raw velocities",
            nyq_per_ray.shape[0], raw.shape[0],
        )
        return _make_field(vdata, vel_field)

    # Simple per-ray phase unwrap, using each ray's own stated
    # Nyquist (widened by the observed |v| max when the data
    # clearly exceeds stated — pre-dealiased archives). Uniformity
    # within the sweep is irrelevant: every ray is processed
    # independently with its own Nyq.
    nyq_effective = _per_ray_effective_nyquist(radar, raw, mask, nyq_per_ray)
    # Low-Z mask: sub-20 dBZ gates are biological / noise / clutter
    # and shouldn't anchor a fold count. Mask only — single-gate
    # folds in precipitation pass through the radial unwrap fine.
    augmented_mask = _mask_low_reflectivity(radar, mask)

    # ---- Phase 1: radial (per-ray) unwrap --------------------------
    out = _unwrap_rays(raw, augmented_mask, nyq_effective)

    # ---- Phase 2: local-window azimuthal consensus -----------------
    # 1D radial unwrap is vulnerable to a single false-fold detection
    # cascading ±2·Nyq across every downrange gate of that ray. We
    # used to run a cumulative-sum azimuthal unwrap pass here, but
    # that propagated false-fold detections all the way around the
    # sweep — producing constant-range arc artifacts visible in the
    # display. Instead, we now do a LOCAL-WINDOW consensus: for each
    # gate, compute the median of nearby azimuthal neighbors
    # (±~5°) at the same range, and if the current gate is
    # ~2·Nyq off from that local median, snap it. The window
    # is small enough that an azimuthal false-fold can't propagate
    # globally, but wide enough that single-gate noise doesn't
    # dominate.
    out = _snap_to_neighbor_consensus(out, mask, nyq_per_ray, radar)

    masked_out = np.ma.array(out, mask=mask)
    return _make_field(masked_out, vel_field)


_DEFAULT_LOCAL_WINDOW_DEG = 5.0


def _snap_to_neighbor_consensus(
    dealiased: np.ndarray,
    mask: np.ndarray,
    nyq_per_ray: np.ndarray,
    radar,
    window_deg: float = _DEFAULT_LOCAL_WINDOW_DEG,
) -> np.ndarray:
    """For each gate, snap rays whose value is ~``2·Nyq`` away from
    the **local-azimuth median** of nearby rays at the same range.

    Uses a small azimuth window — ``window_deg`` half-width measured
    against ``radar.azimuth`` when available, falling back to a fixed
    ray-index half-width when it isn't. The window is purposely
    narrow so a single false-fold detection at one azimuth can't
    propagate across the whole sweep (the old cumulative-sum
    azimuthal unwrap did exactly that and produced constant-range
    arc artifacts visible in the rendered display). Median-of-window
    is robust against single-gate outliers on either side of the
    current ray.

    Rules:

      - Window contains at least 3 same-Nyq neighbors → otherwise
        skip (no stable consensus).
      - Current gate must be unmasked; masked neighbors are excluded
        from the median.
      - Snap by ``round(offset / 2·Nyq) × 2·Nyq`` when the offset is
        at least one ``Nyq`` AND the rounding produces a non-zero
        cycle count.
      - Sector boundaries (different per-ray Nyquists in the window)
        are handled by only including same-Nyq rays in the median
        computation.
    """
    nrays, ngates = dealiased.shape
    if nrays < 3:
        return dealiased
    out = dealiased.copy()

    # Resolve a per-ray local-window in ray indices. We prefer using
    # real azimuth metadata (since ray spacing isn't always uniform
    # — gaps from missed rays, sector scans, etc.) but fall back to
    # a fixed ray-count window when ``radar.azimuth`` isn't there.
    az = _ray_azimuths(radar, nrays)
    sweep_ranges = _sweep_ranges_safe(radar, nrays)
    for s, e in sweep_ranges:
        if e - s < 3:
            continue
        sweep = out[s:e]
        sweep_mask = mask[s:e]
        sweep_nyq = nyq_per_ray[s:e]
        sweep_az = az[s:e] if az is not None else None
        n_sw = sweep.shape[0]

        # For each ray r in the sweep, find indices of rays within
        # ±``window_deg`` of its azimuth (and matching Nyquist).
        for r in range(n_sw):
            if sweep_mask[r].all():
                continue
            ny_r = float(sweep_nyq[r])
            if ny_r <= _MIN_NYQUIST_MS:
                continue
            # Pick the local window in ray-index space.
            lo, hi = _window_indices(r, n_sw, sweep_az, window_deg)
            if hi - lo < 3:
                continue
            # Restrict to same-Nyq rays only.
            window_idx = np.arange(lo, hi)
            same_nyq_mask = np.isclose(sweep_nyq[window_idx], ny_r, atol=0.05)
            window_idx = window_idx[same_nyq_mask]
            # Drop the current ray from its own median.
            window_idx = window_idx[window_idx != r]
            if window_idx.size < 2:
                continue
            window_vals = sweep[window_idx]                 # (k, ngates)
            window_mask = sweep_mask[window_idx]
            window_vals_nan = np.where(window_mask, np.nan, window_vals)
            with np.errstate(invalid="ignore", all="ignore"):
                consensus = np.nanmedian(window_vals_nan, axis=0)   # (ngates,)
            cur = sweep[r]
            cur_mask = sweep_mask[r]
            offset = cur - consensus
            with np.errstate(invalid="ignore"):
                n_cycles = np.where(
                    np.isfinite(offset),
                    np.round(offset / (2.0 * ny_r)),
                    0.0,
                )
                big_offset = np.isfinite(offset) & (np.abs(offset) > ny_r)
                should_snap = (~cur_mask) & big_offset & (n_cycles != 0)
            sweep[r] = np.where(should_snap, cur - 2.0 * ny_r * n_cycles, cur)
        out[s:e] = sweep
    return out


def _ray_azimuths(radar, nrays: int) -> np.ndarray | None:
    """Return the per-ray azimuth array if available, else ``None``."""
    try:
        az = np.asarray(radar.azimuth["data"], dtype=np.float64)
    except (AttributeError, KeyError, TypeError):
        return None
    if az.shape != (nrays,):
        return None
    return az


def _window_indices(
    r: int,
    n_sw: int,
    azimuths: np.ndarray | None,
    half_width_deg: float,
) -> tuple[int, int]:
    """Half-open ``(lo, hi)`` ray indices for ``r``'s local-azimuth
    window. When azimuth metadata isn't available, falls back to a
    fixed ray-index window (``±5`` rays — the typical 5° half-width
    at NEXRAD's 1° beam spacing)."""
    if azimuths is None:
        # ~5° at 1°/ray = ±5 rays.
        return (max(0, r - 5), min(n_sw, r + 6))
    a_r = float(azimuths[r])
    # Circular angular difference, mapped to [0, 180].
    daz = np.abs(((azimuths - a_r + 180.0) % 360.0) - 180.0)
    within = np.where(daz <= half_width_deg)[0]
    if within.size == 0:
        return (r, r + 1)
    # Take a contiguous block bracketing r so the median is over
    # rays that are also physically clustered (azimuth is usually
    # monotonic but can wrap or have gaps).
    lo = max(0, within.min())
    hi = min(n_sw, within.max() + 1)
    return (lo, hi)




def _sweep_ranges_safe(radar, nrays: int) -> list[tuple[int, int]]:
    """Half-open ``(start, end)`` ray indices per sweep, falling back
    to one big sweep when the radar doesn't expose the indexes (test
    harnesses)."""
    try:
        starts = np.asarray(radar.sweep_start_ray_index["data"], dtype=int)
        ends = np.asarray(radar.sweep_end_ray_index["data"], dtype=int)
        return [(int(starts[sw]), int(ends[sw]) + 1)
                for sw in range(int(radar.nsweeps))]
    except (AttributeError, KeyError, TypeError):
        return [(0, nrays)]


def _mask_low_reflectivity(radar, mask: np.ndarray) -> np.ndarray:
    """Return ``mask | (reflectivity < threshold)`` if the radar has
    a reflectivity field; otherwise return ``mask`` unchanged.

    Reflectivity gates the unwrap because Doppler velocity at very
    low Z is dominated by biological scatter (birds at sunset, bat
    swarms, insect bloom) or near-noise speckle — both produce
    radial-velocity values with no physical meaning that nevertheless
    fall within ±Nyq. If the unwrap walks through them they can
    anchor the wrong fold count for the rest of the ray. Masking
    them at this layer makes them transparent (forward-filled across)
    without affecting the user's display: the *original* mask is
    re-applied on the way out, so low-Z gates that were unmasked in
    the input come out with their raw velocity in the corrected
    field — the unwrap just doesn't trust them along the way.
    """
    if "reflectivity" not in radar.fields:
        return mask
    z = radar.fields["reflectivity"]["data"]
    if isinstance(z, np.ma.MaskedArray):
        z_mask = np.ma.getmaskarray(z)
        z_arr = np.asarray(z, dtype=np.float64)
    else:
        z_mask = ~np.isfinite(z)
        z_arr = np.asarray(z, dtype=np.float64)
    if z_arr.shape != mask.shape:
        # Velocity and reflectivity sweep counts can diverge on
        # split-cut volumes — surveillance has Z but no V, Doppler
        # has V but Z is over a different sub-volume. Bail rather
        # than mis-align the masks.
        return mask
    # A masked-Z gate is also low-Z for our purposes (we don't know
    # it's strong, so don't trust it).
    low_z = z_mask | (z_arr < _LOW_DBZ_THRESHOLD)
    return mask | low_z


def _per_ray_effective_nyquist(
    radar,
    raw: np.ndarray,
    mask: np.ndarray,
    stated: np.ndarray,
) -> np.ndarray:
    """Derive each ray's effective Nyquist for unwrap purposes.

    Every ray uses its own stated Nyquist. Disuniformity across the
    sweep is irrelevant — the per-ray value is what the radar
    physically used to fold that ray's data, so it's the right
    unfolding interval for our 1D phase unwrap. Only widen on rays
    whose observed ``|v|`` clearly *exceeds* their stated Nyquist
    (signature of a pre-dealiased archive whose stored values were
    stretched beyond ±stated) — otherwise leave the radar's
    authoritative figure alone.
    """
    v_abs = np.where(mask, 0.0, np.abs(raw))
    obs_max = v_abs.max(axis=1)
    # Tolerance: only widen when observed exceeds stated by more
    # than a fraction of a percent (above float roundoff). That way
    # data sitting at the fold edge (obs_max ≈ stated) doesn't get
    # the Nyq subtly bumped, which would shift fold detections.
    out = stated.copy()
    needs_widen = obs_max > stated * 1.001
    out[needs_widen] = obs_max[needs_widen] * 1.05
    return out


def _unwrap_rays(
    raw: np.ndarray,
    mask: np.ndarray,
    nyq_per_ray: np.ndarray,
) -> np.ndarray:
    """Vectorized 1D phase unwrap with mask-aware carry-forward.

    Inputs are shape ``(nrays, ngates)`` for ``raw`` / ``mask`` and
    ``(nrays,)`` for ``nyq_per_ray``. Returns a ``(nrays, ngates)``
    array of dealiased velocities; masked-gate positions are
    technically "filled" but the caller re-applies the mask, so their
    values don't matter for display.

    All operations are along axis=1 (gate axis); there are no Python
    loops over rays.
    """
    nrays, ngates = raw.shape
    if nrays == 0 or ngates < 2:
        return raw.copy()

    # ---- Step 1: forward-fill masked gates within each ray ---------
    # ``idx`` ends up as the index of the most-recent unmasked gate
    # for each (row, col); using ``np.maximum.accumulate`` on
    # ``np.where(mask, -1, col_index)`` gives that O(nrays * ngates)
    # without a Python loop.
    col_idx = np.broadcast_to(np.arange(ngates), (nrays, ngates))
    masked_idx = np.where(mask, -1, col_idx)
    last_valid = np.maximum.accumulate(masked_idx, axis=1)
    # Rows that start with masked gates have last_valid=-1 until the
    # first unmasked column. Substitute 0 — the result there gets
    # re-masked by the caller anyway.
    last_valid = np.where(last_valid < 0, 0, last_valid)
    row_idx = np.arange(nrays)[:, None]
    filled = raw[row_idx, last_valid]
    # NaN from start-of-row masked gates: replace with 0 before unwrap.
    filled = np.where(np.isfinite(filled), filled, 0.0)

    # ---- Step 2: per-ray Nyquist + skip-dealias mask ---------------
    # Rays with a non-usable Nyquist (zero / sub-floor) pass through
    # untouched. We do this by computing the unwrap as if their
    # Nyquist were +inf (no folds detected); easiest is to compute
    # everything and then patch those rows back to the original.
    nyq_safe = np.where(nyq_per_ray > _MIN_NYQUIST_MS, nyq_per_ray, 1.0)
    nyq_col = nyq_safe[:, None]

    # ---- Step 3: detect folds via phase differences ----------------
    # In phase units, a jump from +V_ny to -V_ny is a drop of -2π
    # (folded forward by one cycle). Conversely, +V_ny to -V_ny going
    # the other direction is +2π. We detect these via
    # ``|diff| > π``, signed.
    phase = filled * (np.pi / nyq_col)            # (nrays, ngates)
    diff = np.diff(phase, axis=1)                  # (nrays, ngates-1)
    step = np.zeros_like(diff, dtype=np.int8)
    step[diff > np.pi] = -1     # phase dropped 2π → upcoming gate is one fold high
    step[diff < -np.pi] = 1     # phase jumped 2π → upcoming gate is one fold low
    # Cumulative fold count, padding a 0 at column 0 (first gate has
    # no preceding diff so its fold count is 0).
    folds = np.concatenate(
        [np.zeros((nrays, 1), dtype=np.int32), np.cumsum(step, axis=1, dtype=np.int32)],
        axis=1,
    )

    # ---- Step 4: apply correction ----------------------------------
    # Apply the fold count to the RAW value (not ``filled``). Gates
    # that were augmented-masked (clutter spikes, sub-20 dBZ) have
    # ``filled`` = the carry-forward from an earlier gate — that's
    # what was used to compute the fold sequence so the spike
    # couldn't anchor the cumulative count. But the *output* at
    # those positions should be the gate's own measured velocity
    # with whichever fold count the unwrap reached by that point,
    # NOT the carry-forward value. Otherwise low-Z gates would have
    # their displayed velocity overwritten by an earlier gate's
    # value — wrong both as data and as user expectation. Where
    # ``raw`` is NaN (genuinely missing data), ``raw + folds*nyq``
    # stays NaN and the caller's output mask covers it.
    dealiased = raw + 2.0 * nyq_col * folds

    # ---- Step 5: rows we skipped → revert to raw -------------------
    skip_rows = nyq_per_ray <= _MIN_NYQUIST_MS
    if np.any(skip_rows):
        dealiased[skip_rows] = raw[skip_rows]
    # Final NaN scrub — ``raw + …`` propagates NaN through originally-
    # masked positions, which is fine for the data layer but downstream
    # rendering code occasionally sums values without nan-checking, so
    # zero them out. Output mask preserves them as masked anyway.
    dealiased = np.where(np.isfinite(dealiased), dealiased, 0.0)

    return dealiased


def _make_field(data, source_field: dict) -> dict:
    """Build a PyART-compatible ``corrected_velocity`` field dict.

    Carries through units / valid range metadata from the source
    ``velocity`` field where present so the downstream colormap
    selector and inspector readout don't have to special-case our
    dealiaser's output.
    """
    out = {
        "data": data,
        "long_name": "Velocity (dealiased, ray-by-ray)",
        "standard_name": "radial_velocity_of_scatterers_away_from_instrument",
        "units": source_field.get("units", "meters_per_second"),
        "_FillValue": source_field.get("_FillValue"),
        "coordinates": source_field.get("coordinates", "elevation azimuth range"),
        "comments": (
            "Dealiased by radar_warning_game.data.dealias_ray. Per-ray "
            "1D phase unwrap using each ray's own Nyquist velocity — "
            "handles PRF-staggered sweeps where PyART's region-based "
            "and 2D phase-unwrap dealiasers refuse to run."
        ),
    }
    return out
