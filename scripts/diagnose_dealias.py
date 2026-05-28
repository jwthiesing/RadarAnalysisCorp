"""Dump per-sweep dealiasing diagnostics for a single Level 2 file.

Usage::

    python scripts/diagnose_dealias.py path/to/KLSX20101231_173000_V06.gz

Reports, per sweep:
  - sweep number, elevation, ray count, gate count
  - stated Nyquist statistics (min / max / unique values rounded to 0.01)
  - whether the stated values are uniform within the sweep
  - observed |velocity| max (over unmasked gates) — the lower bound on
    the *real* Nyquist used to fold the data
  - what the ray-by-ray dealiaser would pick as its effective per-ray
    Nyquist for that sweep
  - count of detected folds (places where ``|phase_diff| > π``) before
    and after the clutter-spike mask

The goal is to pinpoint what's different about a problematic archive
vs one that dealiases correctly — usually it's either a metadata
quirk (e.g. stated Nyquist tiny vs observed max), a near-radar clutter
field that survives spike detection, or a velocity field whose fill
value isn't masked properly.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def main(path: str) -> int:
    import pyart
    radar = pyart.io.read_nexrad_archive(path)
    if "velocity" not in radar.fields:
        print(f"no velocity field in {path}", file=sys.stderr)
        return 1

    vel = radar.fields["velocity"]["data"]
    if not isinstance(vel, np.ma.MaskedArray):
        vel = np.ma.masked_invalid(vel)
    mask = np.ma.getmaskarray(vel)
    raw = np.asarray(vel, dtype=np.float64)

    inst = radar.instrument_parameters or {}
    nyq_dict = inst.get("nyquist_velocity")
    if nyq_dict is None:
        nyq_per_ray = None
    else:
        nyq_per_ray = np.abs(np.asarray(nyq_dict["data"], dtype=np.float64))

    starts = np.asarray(radar.sweep_start_ray_index["data"], dtype=int)
    ends = np.asarray(radar.sweep_end_ray_index["data"], dtype=int)
    angles = np.asarray(radar.fixed_angle["data"], dtype=float)

    print(f"file: {path}")
    print(f"  nrays={radar.nrays}  ngates={radar.ngates}  nsweeps={radar.nsweeps}")
    print(f"  velocity mask coverage: {mask.mean() * 100:.1f}%")
    print()
    print(f"{'sw':>2}  {'elev':>5}  {'nrays':>5}  "
          f"{'stated_min':>10}  {'stated_max':>10}  {'unique':>8}  "
          f"{'uniform?':>9}  {'obs|v|max':>10}  {'eff_min':>8}  "
          f"{'eff_max':>8}  {'folds_raw':>10}  {'folds_spk':>10}")
    for sw in range(radar.nsweeps):
        s, e = int(starts[sw]), int(ends[sw]) + 1
        sweep_nyq = nyq_per_ray[s:e] if nyq_per_ray is not None else None
        sweep_raw = raw[s:e]
        sweep_mask = mask[s:e]
        v_abs = np.where(sweep_mask, 0.0, np.abs(sweep_raw))
        obs_max = float(v_abs.max()) if v_abs.size else 0.0
        if sweep_nyq is None or sweep_nyq.size == 0:
            stmin = stmax = 0.0
            unique = 0
            uniform = "—"
            eff_min = eff_max = 0.0
        else:
            stmin, stmax = float(sweep_nyq.min()), float(sweep_nyq.max())
            unique = len(np.unique(np.round(sweep_nyq, 2)))
            spread = stmax - stmin
            ref = max(stmax, 1.0)
            uniform = "yes" if (spread / ref) <= 1e-3 else "NO"
            # effective per-ray nyquist the dealiaser would pick.
            # Disuniform sweep: pin to sweep's MAX stated (matches the
            # combined-PRF effective Nyquist) and widen by observed.
            v_abs_per_ray = np.where(sweep_mask, 0.0, np.abs(sweep_raw)).max(axis=1)
            if uniform == "yes":
                eff = sweep_nyq
            else:
                eff = np.maximum(stmax, v_abs_per_ray * 1.05)
            eff_min, eff_max = float(eff.min()), float(eff.max())
        # Fold count using the EFFECTIVE per-ray Nyquist (so the
        # numbers reflect what the actual dealiaser would do, not
        # the raw stated values).
        if obs_max > 0 and sweep_nyq is not None and sweep_nyq.size > 0:
            v_filled = sweep_raw.copy()
            v_filled[sweep_mask] = np.nan
            d = np.diff(v_filled, axis=1)
            # Broadcast each row's effective Nyquist against its diffs.
            eff_col = eff[:, None] if eff.ndim == 1 else np.full_like(d, eff)
            with np.errstate(invalid="ignore"):
                folds_raw = int(np.sum(np.abs(d) > eff_col))
            if v_filled.shape[1] >= 3:
                left = v_filled[:, :-2]; mid = v_filled[:, 1:-1]; right = v_filled[:, 2:]
                ld = np.abs(mid - left); rd = np.abs(mid - right); nd = np.abs(left - right)
                thr = 0.5 * (eff[:, None] if eff.ndim == 1 else eff)
                with np.errstate(invalid="ignore"):
                    spikes = (ld > thr) & (rd > thr) & (nd < thr)
                folds_spk = folds_raw - int(np.sum(spikes))
            else:
                folds_spk = folds_raw
        else:
            folds_raw = folds_spk = 0
        print(f"{sw:>2}  {angles[sw]:>5.1f}  {e-s:>5}  "
              f"{stmin:>10.2f}  {stmax:>10.2f}  {unique:>8}  "
              f"{uniform:>9}  {obs_max:>10.2f}  {eff_min:>8.2f}  "
              f"{eff_max:>8.2f}  {folds_raw:>10}  {folds_spk:>10}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("file")
    args = p.parse_args()
    raise SystemExit(main(args.file))
