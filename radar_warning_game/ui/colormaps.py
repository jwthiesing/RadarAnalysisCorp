"""Custom colormaps registered with matplotlib at import time.

PyART ships :data:`Carbone42` (and others) and registers them on its
own import; everything else the radar panel needs lives here. Keeping
the registration in a side-effect-on-import module means
``radar_panel`` can pull colormap names through the existing matplotlib
bridge (``pg.colormap.getFromMatplotlib(name)``) without special-casing.

Currently bundled:

  - **RadarCope** — Doppler-velocity colormap parsed from the bundled
    GR2Analyst-format palette ``resources/colormaps/samVELBV.pal``.
    Replace that file to swap palettes; the parser handles both
    single-RGB and dual-RGB-with-step stops in the .pal format.

GR2Analyst .pal format reference (`color:` lines):

    color: V R G B               # one RGB triplet at value V
    color: V R1 G1 B1 R2 G2 B2   # two triplets — hard step at V

The single-triplet form is a smooth stop: the same color is used both
for the gradient approaching V from below and leading away above. The
two-triplet form makes V a discontinuity: ``(R1,G1,B1)`` is the
"approach" color (used at V when interpolating from the lower stop),
``(R2,G2,B2)`` is the "leave" color (used at V when interpolating to
the upper stop). Lines are not required to be in any order in the
file — we sort by V on parse.
"""

from __future__ import annotations

import re
from pathlib import Path

from matplotlib.colors import LinearSegmentedColormap

try:
    from matplotlib import colormaps as _mpl_colormaps
except ImportError:   # matplotlib < 3.7 fallback
    _mpl_colormaps = None


_PAL_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "resources" / "colormaps" / "samVELBV.pal"
)


def _parse_pal(path: Path) -> list[tuple[float, tuple[int, int, int], tuple[int, int, int]]]:
    """Parse a GR2Analyst .pal file's ``color:`` lines.

    Returns a list of ``(V, (R_in, G_in, B_in), (R_out, G_out, B_out))``
    tuples sorted by V ascending. Single-RGB stops are stored with
    ``in == out``; dual-RGB stops preserve the discontinuity for the
    matplotlib segment-data path so the renderer reproduces the hard
    step at that value.
    """
    out: list[tuple[float, tuple[int, int, int], tuple[int, int, int]]] = []
    line_re = re.compile(
        r"""^\s*color:\s*
            ([-+]?\d+(?:\.\d+)?)\s+       # V
            (\d+)\s+(\d+)\s+(\d+)         # R1 G1 B1
            (?:\s+(\d+)\s+(\d+)\s+(\d+))? # optional R2 G2 B2
            \s*$""",
        re.X,
    )
    for raw in path.read_text(encoding="utf-8").splitlines():
        m = line_re.match(raw)
        if not m:
            continue
        v = float(m.group(1))
        r1, g1, b1 = int(m.group(2)), int(m.group(3)), int(m.group(4))
        if m.group(5) is not None:
            r2, g2, b2 = int(m.group(5)), int(m.group(6)), int(m.group(7))
        else:
            r2, g2, b2 = r1, g1, b1
        out.append((v, (r1, g1, b1), (r2, g2, b2)))
    out.sort(key=lambda s: s[0])
    return out


def _build_cmap_from_pal(name: str, path: Path) -> LinearSegmentedColormap | None:
    """Build a :class:`LinearSegmentedColormap` from a GR2Analyst pal.

    The pal's V range maps linearly to position ``[0, 1]``. Stops with
    distinct in/out RGBs become discontinuities in the segment-data —
    matplotlib's segment format supports this natively
    (``(pos, y0, y1)`` triplets, where ``y0`` is the value approaching
    ``pos`` from below and ``y1`` is the value approaching from above).
    """
    if not path.exists():
        return None
    stops = _parse_pal(path)
    if len(stops) < 2:
        return None
    v_min, v_max = stops[0][0], stops[-1][0]
    span = (v_max - v_min) or 1.0
    reds: list[tuple[float, float, float]] = []
    greens: list[tuple[float, float, float]] = []
    blues: list[tuple[float, float, float]] = []
    for v, (r_in, g_in, b_in), (r_out, g_out, b_out) in stops:
        pos = (v - v_min) / span
        reds.append((pos, r_in / 255.0, r_out / 255.0))
        greens.append((pos, g_in / 255.0, g_out / 255.0))
        blues.append((pos, b_in / 255.0, b_out / 255.0))
    return LinearSegmentedColormap(
        name, segmentdata={"red": reds, "green": greens, "blue": blues}, N=256,
    )


def _register(name: str, cmap: LinearSegmentedColormap | None) -> bool:
    """Register ``cmap`` under ``name`` with matplotlib, tolerating
    repeated imports (the radar_panel module imports us at top level,
    and pytest can re-import within a session). Returns whether
    registration succeeded."""
    if cmap is None or _mpl_colormaps is None:
        return False
    try:
        _mpl_colormaps.register(cmap, name=name)
    except ValueError:
        # Already registered — fine.
        pass
    return True


RADARCOPE = _build_cmap_from_pal("RadarCope", _PAL_PATH)
_register("RadarCope", RADARCOPE)


# Public list of velocity colormap choices the UI offers. ``RadarCope``
# only appears in the dropdown if the pal parsed successfully —
# Carbone42 (PyART built-in) is always available as a fallback.
VELOCITY_COLORMAP_CHOICES: list[str] = (
    ["RadarCope", "Carbone42"] if RADARCOPE is not None else ["Carbone42"]
)
