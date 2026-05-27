"""Fix-ups for PyART :class:`pyart.core.Radar` objects loaded from
NEXRAD Level 2 archives — specifically, metadata that PyART's reader
leaves in a state that downstream algorithms (dealiasing in particular)
can't use.

The motivating case is **TDWR** Level 2 files on Unidata's mirror.
PyART's ``read_nexrad_archive`` reads them fine but populates
``radar.instrument_parameters['nyquist_velocity']`` with an all-zeros
array. :func:`pyart.correct.dealias_region_based` then divides by
``nyquist_velocity * 2`` to compute aliasing bin counts → ``1/0 = inf``
→ ``ValueError: cannot convert float infinity to integer`` when the
bin counts get cast to int. Some older WSR-88D files (legacy-resolution
volumes before the dual-pol upgrade) hit the same path.

The fix here derives a usable per-ray Nyquist from the observed
velocity field — for a Doppler velocity field, |v| is bounded by the
true Nyquist by construction, so taking the per-sweep max of |v| and
adding a small safety margin gives the correct unfolded interval.
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


def ensure_nyquist_velocity(radar) -> bool:
    """Ensure ``radar.instrument_parameters['nyquist_velocity']`` is a
    usable per-ray array. Returns ``True`` if a repair was performed.

    A repair is needed when the existing array is missing, empty,
    non-finite, or all zeros — any of which would poison PyART's
    dealiasing algorithms with a divide-by-zero or inf.

    Per-sweep Nyquist is derived from the observed ``velocity`` field's
    absolute maximum, padded by :data:`_NYQUIST_SAFETY_MARGIN`. Sweeps
    with no usable velocity signal fall back to :data:`_DEFAULT_NYQUIST_MS`."""
    if "velocity" not in radar.fields:
        return False
    if radar.instrument_parameters is None:
        radar.instrument_parameters = {}
    existing = radar.instrument_parameters.get("nyquist_velocity")
    existing_data = existing["data"] if existing is not None else None
    needs_fix = False
    if existing_data is None:
        needs_fix = True
    else:
        arr = np.asarray(existing_data, dtype=np.float64)
        if arr.size == 0:
            needs_fix = True
        elif not np.isfinite(arr).all():
            needs_fix = True
        elif np.allclose(arr, 0.0):
            needs_fix = True
    if not needs_fix:
        return False
    velocity = radar.fields["velocity"]["data"]
    per_ray_nyquist = np.full(radar.nrays, _DEFAULT_NYQUIST_MS, dtype=np.float32)
    for sw in range(radar.nsweeps):
        s = int(radar.sweep_start_ray_index["data"][sw])
        e = int(radar.sweep_end_ray_index["data"][sw]) + 1
        sweep_v = velocity[s:e]
        # Pull out valid (unmasked + finite) samples; if there's no
        # signal in this sweep, fall through to the default.
        if hasattr(sweep_v, "compressed"):
            samples = sweep_v.compressed()
        else:
            samples = np.asarray(sweep_v)
            samples = samples[np.isfinite(samples)]
        if samples.size == 0:
            continue
        v_abs_max = float(np.abs(samples).max())
        # Very-low-magnitude max suggests no real Doppler signal —
        # better to use the floor than report a fake 1-2 m/s Nyquist
        # that would alias real velocities later.
        if v_abs_max < 1.0:
            continue
        per_ray_nyquist[s:e] = v_abs_max * _NYQUIST_SAFETY_MARGIN
    radar.instrument_parameters["nyquist_velocity"] = {
        "data": per_ray_nyquist,
        "units": "meters_per_second",
        "long_name": "unambiguous_doppler_velocity",
        "comments": (
            "Derived from observed velocity-field magnitude — "
            "PyART's NEXRAD reader left nyquist_velocity unset / "
            "all-zero (typical for TDWR Level 2 files and some "
            "legacy-resolution WSR-88D volumes)."
        ),
    }
    log.debug(
        "ensure_nyquist_velocity repaired %d rays; per-sweep range %.1f-%.1f m/s",
        per_ray_nyquist.size,
        float(per_ray_nyquist.min()),
        float(per_ray_nyquist.max()),
    )
    return True
