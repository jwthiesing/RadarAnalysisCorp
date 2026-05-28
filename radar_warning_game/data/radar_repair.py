"""Fix-ups for PyART :class:`pyart.core.Radar` objects loaded from
NEXRAD Level 2 archives — specifically, metadata that PyART's reader
leaves in a state that downstream algorithms (dealiasing in particular)
can't use.

Two failure modes covered:

1. **Zero / missing / non-finite Nyquist velocity.** TDWR Level 2
   files on Unidata's mirror (and some legacy WSR-88D volumes) come
   through with all-zero Nyquist. :func:`pyart.correct.dealias_region_based`
   then divides by ``nyquist_velocity * 2`` to compute aliasing bin
   counts → ``1/0 = inf`` → ``ValueError: cannot convert float infinity
   to integer`` when the bin counts get cast to int.

2. **Non-uniform Nyquist within a sweep.** NEXRAD radars often stagger
   their PRF mid-sweep to extend the unambiguous range, which leaves
   different rays in the same sweep with different Nyquist values
   ("Nyquist velocities are not uniform in sweep"). PyART's region-based
   dealias rejects the sweep outright in that case. We use the
   **lowest non-zero Nyquist** the file states for that sweep — a
   conservative choice that matches the slowest PRF actually recorded.
   Older WSR-88D archives sometimes contain pre-dealiased velocities,
   so deriving a Nyquist from ``max(|v|)`` would overshoot (the
   dealiased values can be much larger than the original unambiguous
   limit). Falling back to the file's lowest stated value avoids that
   mistake. If every per-ray value in the sweep is zero (a degenerate
   metadata case), we drop to the bad-data path's velocity-derived
   estimate.

In both cases the repair makes the per-sweep arrays uniform so PyART
will run.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

# Safe default Nyquist (m/s) when a sweep has no measurable velocity
# signal — high enough to swallow any plausible PRF setting without
# triggering false aliasing bins, low enough that the dealias
# algorithm's bin allocation stays small.
_DEFAULT_NYQUIST_MS = 25.0
# When derived from observed velocities, pad by this factor so the
# real Nyquist isn't undershot (region-based dealias is sensitive at
# the edges).
_NYQUIST_SAFETY_MARGIN = 1.05
# Two Nyquist values are treated as "the same" when their difference
# is below this fraction of the sweep's max — covers float roundoff
# without merging PRF-staggered runs together.
_NYQUIST_UNIFORM_TOL = 1e-3


def ensure_nyquist_velocity(radar) -> bool:
    """Ensure ``radar.instrument_parameters['nyquist_velocity']`` is a
    per-ray array that PyART's dealias algorithms accept. Returns
    ``True`` if a repair was performed.

    Two repair triggers:
      - **Missing / empty / non-finite / all-zero data** → no source
        values to start from; ``max(|v|) × safety_margin`` is the
        only signal available.
      - **Non-uniform within a sweep** (PRF stagger) → use the lowest
        non-zero Nyquist the file states for that sweep. That's a
        conservative choice that matches the slowest PRF recorded
        and won't overshoot like ``max(|v|)`` does on already-
        dealiased older archives.
    """
    if "velocity" not in radar.fields:
        return False
    if radar.instrument_parameters is None:
        radar.instrument_parameters = {}
    existing = radar.instrument_parameters.get("nyquist_velocity")
    existing_data = existing["data"] if existing is not None else None

    # Decide which repair path applies, and for the non-uniform case
    # remember the source per-ray array so the per-sweep loop below
    # can read the file's stated Nyquist values directly.
    bad_data = False
    nonuniform = False
    source_arr: np.ndarray | None = None
    if existing_data is None:
        bad_data = True
    else:
        arr = np.asarray(existing_data, dtype=np.float64)
        if arr.size == 0:
            bad_data = True
        elif not np.isfinite(arr).all():
            bad_data = True
        elif np.allclose(arr, 0.0):
            bad_data = True
        elif arr.size == radar.nrays:
            # Check uniformity sweep-by-sweep against the **magnitude**
            # of each ray's stated Nyquist. Sign is meaningless for an
            # unambiguous-velocity bound — a ``-25`` value just means
            # the file encoded the limit in the negative direction.
            # Without ``abs`` here a sweep like ``[+25, -25]`` would
            # look wildly non-uniform when it's effectively the same
            # PRF limit twice. We only repair when there's actually a
            # mismatch; uniform-good arrays are left untouched.
            arr_abs = np.abs(arr)
            for sw in range(radar.nsweeps):
                s = int(radar.sweep_start_ray_index["data"][sw])
                e = int(radar.sweep_end_ray_index["data"][sw]) + 1
                sweep_ny = arr_abs[s:e]
                if sweep_ny.size == 0:
                    continue
                spread = float(sweep_ny.max() - sweep_ny.min())
                ref = max(float(sweep_ny.max()), 1.0)
                if spread / ref > _NYQUIST_UNIFORM_TOL:
                    nonuniform = True
                    break
            if nonuniform:
                source_arr = arr_abs
    if not (bad_data or nonuniform):
        return False

    velocity = radar.fields["velocity"]["data"]
    per_ray_nyquist = np.full(radar.nrays, _DEFAULT_NYQUIST_MS, dtype=np.float32)
    for sw in range(radar.nsweeps):
        s = int(radar.sweep_start_ray_index["data"][sw])
        e = int(radar.sweep_end_ray_index["data"][sw]) + 1
        # Non-uniform path: trust the file's stated per-ray values
        # and pick the *lowest non-zero* Nyquist for this sweep.
        # That matches the slowest PRF the radar actually recorded
        # and avoids the ``max(|v|)`` mistake on archives that
        # already had their velocities dealiased (the inflated
        # observed |v| would push the Nyquist way above the true
        # unambiguous limit, defeating the dealias retry). A sweep
        # whose stated values are all zero falls through to the
        # velocity-derived path below.
        if source_arr is not None:
            # ``source_arr`` is the magnitude of the file's stated
            # Nyquist values (see the abs() applied where it's
            # captured). Filtering ``> 0`` drops bad-metadata rays
            # without depending on the original sign — a file that
            # encoded the limit as -25 still contributes 25 here, and
            # a sweep with all-negative Nyquists no longer silently
            # falls through to the velocity-derived path.
            sweep_stated = source_arr[s:e]
            nonzero = sweep_stated[sweep_stated > 0.0]
            if nonzero.size > 0:
                per_ray_nyquist[s:e] = float(nonzero.min())
                continue
        # Bad-data path (or non-uniform sweep with no non-zero values
        # to fall back on): take the sweep's observed ``max(|v|)``
        # and pad by the safety margin. ``max(|v|)`` is the only
        # signal we have when the file gave us nothing.
        sweep_v = velocity[s:e]
        if hasattr(sweep_v, "compressed"):
            samples = sweep_v.compressed()
        else:
            samples = np.asarray(sweep_v)
            samples = samples[np.isfinite(samples)]
        if samples.size == 0:
            # No signal in this sweep — leave the floor default
            # already filled into per_ray_nyquist.
            continue
        v_abs_max = float(np.abs(samples).max())
        # Very-low-magnitude max suggests clear-air / no real Doppler
        # signal — better to use the floor than report a fake 1-2 m/s
        # Nyquist that would alias real velocities later.
        if v_abs_max < 1.0:
            continue
        per_ray_nyquist[s:e] = v_abs_max * _NYQUIST_SAFETY_MARGIN

    reason = []
    if bad_data:
        reason.append("missing/zero/non-finite")
    if nonuniform:
        reason.append("non-uniform within sweep")
    radar.instrument_parameters["nyquist_velocity"] = {
        "data": per_ray_nyquist,
        "units": "meters_per_second",
        "long_name": "unambiguous_doppler_velocity",
        "comments": (
            "Repaired by radar_repair.ensure_nyquist_velocity "
            f"({', '.join(reason)}). Non-uniform sweeps use the lowest "
            "non-zero per-ray Nyquist the file stated; bad/empty/zero "
            "sweeps fall back to observed |velocity| max × safety margin."
        ),
    }
    log.debug(
        "ensure_nyquist_velocity repaired %d rays (%s); per-sweep "
        "range %.1f-%.1f m/s",
        per_ray_nyquist.size, " + ".join(reason),
        float(per_ray_nyquist.min()),
        float(per_ray_nyquist.max()),
    )
    return True
