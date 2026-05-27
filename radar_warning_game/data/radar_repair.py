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
   dealias rejects the sweep outright in that case. The per-ray values
   on disk are the unambiguous limit of each individual PRF pulse —
   *not* the effective Nyquist of the combined post-stagger return,
   which is set by the radar's slowest PRF and could be much higher.
   So we derive the per-sweep Nyquist from the observed velocity field
   the same way the bad-data path does — ``max(|v|)`` is by construction
   an upper bound on |v| and avoids both under-reporting (which would
   cause valid velocities to get unfolded incorrectly) and using the
   file's mixed-PRF values that don't represent the real ceiling.

In both cases the repair makes the per-sweep arrays uniform so PyART
will run; the value used is always a safe upper bound on |v|.
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

    Two repair triggers — **both resolved identically** by deriving
    each sweep's Nyquist from the observed ``velocity`` field:
      - **Missing / empty / non-finite / all-zero data** → no source
        values to start from; ``max(|v|) × safety_margin`` is the
        only signal available.
      - **Non-uniform within a sweep** (PRF stagger) → the per-ray
        values are individual-PRF unambiguous limits, not the
        effective post-stagger Nyquist. Deriving from
        ``max(|v|) × safety_margin`` gives the correct ceiling (|v|
        is by construction bounded by the *real* Nyquist).
    """
    if "velocity" not in radar.fields:
        return False
    if radar.instrument_parameters is None:
        radar.instrument_parameters = {}
    existing = radar.instrument_parameters.get("nyquist_velocity")
    existing_data = existing["data"] if existing is not None else None

    # Decide which repair path applies.
    bad_data = False
    nonuniform = False
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
            # Check uniformity sweep-by-sweep. We only repair when
            # there's actually a mismatch; uniform-good arrays are
            # left untouched.
            for sw in range(radar.nsweeps):
                s = int(radar.sweep_start_ray_index["data"][sw])
                e = int(radar.sweep_end_ray_index["data"][sw]) + 1
                sweep_ny = arr[s:e]
                if sweep_ny.size == 0:
                    continue
                spread = float(sweep_ny.max() - sweep_ny.min())
                ref = max(float(sweep_ny.max()), 1.0)
                if spread / ref > _NYQUIST_UNIFORM_TOL:
                    nonuniform = True
                    break
    if not (bad_data or nonuniform):
        return False

    velocity = radar.fields["velocity"]["data"]
    per_ray_nyquist = np.full(radar.nrays, _DEFAULT_NYQUIST_MS, dtype=np.float32)
    for sw in range(radar.nsweeps):
        s = int(radar.sweep_start_ray_index["data"][sw])
        e = int(radar.sweep_end_ray_index["data"][sw]) + 1
        # Single derivation strategy: take the sweep's observed
        # ``max(|v|)`` and pad by the safety margin. This is correct
        # for both repair triggers — see the module docstring for why
        # we don't trust the file's mixed-PRF values for the
        # non-uniform case either.
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
            f"({', '.join(reason)}). Each sweep's Nyquist is derived "
            "from the observed |velocity| max plus a small safety "
            "margin — correct for both missing/zero values and "
            "PRF-staggered values that don't represent the effective "
            "post-stagger Nyquist."
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
