"""Board surface reconstruction and registration (Section III-B).

  rebin_line(...)          one raw scan line -> heights on the X grid (median of laser frames per bin)
  height_map(...)          stack lines into the height grid H(y, x), map frame, upward positive
  smooth(H)                local-median positioning field (21 x 5 cells) used for probe heights
  Registration             map frame <-> stage frame: planar shift (dx, dy), rotation about (87, 87) mm,
                           vertical offset dz and linear tilt (Eq. 1); fitted to laser spot heights
  candidates(...)          1 mm candidate grid with stage XY of the EM probe and registered height

Scan geometry of the paper: X 64-110 mm, Y 69-106 mm, Y line pitch 0.01 mm, X bins 0.0495 mm
(1.5 x frame spacing at 5 mm/s), unidirectional lines from large to small X.
`checks/surface_check.py` rebuilds archived rows and the 1748 candidate heights.
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import median_filter
from scipy.optimize import least_squares

X_RANGE, Y_RANGE = (64.0, 110.0), (69.0, 106.0)
X_BIN, Y_PITCH = 0.0495, 0.01
CENTER = np.array([87.0, 87.0])
MEDIAN_CELLS = (21, 5)


# ---------------------------------------------------------------- reconstruction
def x_edges(x_range=X_RANGE, x_bin=X_BIN):
    return np.arange(x_range[0], x_range[1] + x_bin, x_bin)


def rebin_line(x_mm, distance_mm, edges):
    """Median laser distance per X bin. x_mm is the X position of each laser frame, obtained from a
    constant-velocity fit of encoder samples taken during the line (scanner side)."""
    x, d = np.asarray(x_mm, float), np.asarray(distance_mm, float)
    ok = np.isfinite(x) & np.isfinite(d)
    idx = np.digitize(x[ok], edges) - 1
    keep = (idx >= 0) & (idx < len(edges) - 1)
    idx, d = idx[keep], d[ok][keep]
    out, counts = np.full(len(edges) - 1, np.nan), np.zeros(len(edges) - 1, int)
    order = np.argsort(idx, kind='stable')
    idx, d = idx[order], d[order]
    bounds = np.searchsorted(idx, np.arange(len(edges)))
    for c in range(len(edges) - 1):
        s, e = bounds[c], bounds[c + 1]
        if e > s:
            out[c], counts[c] = np.median(d[s:e]), e - s
    return out, counts


def height_map(lines, x_range=X_RANGE, x_bin=X_BIN):
    """lines: iterable of (x_mm, distance_mm) per Y row. Returns xs, H (rows x bins), counts.
    Height is the negative laser distance relative to the global median (board surface near 0)."""
    edges = x_edges(x_range, x_bin)
    rows = [rebin_line(x, d, edges) for x, d in lines]
    D = np.array([r[0] for r in rows])
    counts = np.array([r[1] for r in rows])
    return .5 * (edges[:-1] + edges[1:]), np.nanmedian(D) - D, counts


def smooth(H, cells=MEDIAN_CELLS):
    """Local-median field for positioning; raw maxima stay available as obstacle evidence."""
    return median_filter(H, size=cells, mode='nearest')


# ---------------------------------------------------------------- registration
class Registration:
    """p = [dx, dy, dz, rotation_deg, tilt_x, tilt_y]; stage laser-spot XY <-> map XY."""

    def __init__(self, xs, ys, field, params):
        self.interp = RegularGridInterpolator((ys, xs), field, bounds_error=False, fill_value=np.nan)
        self.p = np.asarray(params, float)
        a = np.deg2rad(self.p[3])
        self.rot = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])

    def to_map(self, laser_xy):
        return (np.asarray(laser_xy, float) - CENTER) @ self.rot + (CENTER + self.p[:2])

    def to_stage(self, map_xy):
        return (np.asarray(map_xy, float) - (CENTER + self.p[:2])) @ self.rot.T + CENTER

    def height(self, laser_xy):
        """Eq. 1: registered surface height under a stage laser-spot position."""
        xy = np.atleast_2d(np.asarray(laser_xy, float))
        tilt = self.p[2] + self.p[4] * (xy[:, 0] - CENTER[0]) + self.p[5] * (xy[:, 1] - CENTER[1])
        return self.interp(self.to_map(xy)[:, ::-1]) + tilt

    @classmethod
    def fit(cls, xs, ys, field, laser_xy, laser_h, start, weights=None):
        """Robust least squares on laser spot heights measured at known stage positions."""
        laser_xy, laser_h = np.asarray(laser_xy, float), np.asarray(laser_h, float)
        w = np.ones(len(laser_h)) if weights is None else np.asarray(weights, float)
        lo = np.array([-1., 1., -25.5, -1., -.015, -.015])
        hi = np.array([1., 3., -23.5, 1., .015, .015])
        best = None
        for shift in (0., -.25, .25):
            s = np.asarray(start, float).copy()
            s[0] += shift
            r = least_squares(lambda v: np.nan_to_num((cls(xs, ys, field, v).height(laser_xy) - laser_h) * np.sqrt(w), nan=10.),
                              s, bounds=(lo, hi), loss='soft_l1', f_scale=.04, max_nfev=250, diff_step=1e-4)
            if best is None or r.cost < best.cost:
                best = r
        return cls(xs, ys, field, best.x)


# ---------------------------------------------------------------- candidates
def candidates(reg, probe_offset_xy, x_range=X_RANGE, y_range=Y_RANGE, pitch=1.0, x0=64.02475):
    """1 mm grid in map coordinates; returns ids 'fXX_YY', EM-probe stage XY and registered height."""
    gx = np.arange(x0, x_range[1] + 1e-9, pitch)
    gy = np.arange(y_range[0], y_range[1] + 1e-9, pitch)
    ids, map_xy = [], []
    for j, y in enumerate(gy):
        for i, x in enumerate(gx):
            ids.append(f'f{i:02d}_{j:02d}')
            map_xy.append([x, y])
    map_xy = np.array(map_xy)
    laser_xy = reg.to_stage(map_xy)
    return ids, map_xy, laser_xy + np.asarray(probe_offset_xy, float), reg.height(laser_xy)
