"""Terrain classes from the height map (Section III-C).

The thresholds come from package geometry, not from leakage measured on the board:
  PART_MIN_MM     0.15  above the board plane counts as a part (copper and solder mask stay below 0.1 mm)
  IC_AREA_MM2     9     footprint of the smallest common IC packages (3 x 3 mm QFN)
  SMALL_AREA_MM2  6     largest two-terminal chip part (1206, 3.2 x 1.6 mm) with its pads
  TOP_SHARE       0.8   an IC top is its footprint above 80 % of its height (the flat package body)
  BAND_MM         1.0   IC boundary band: up to 1 mm outside the IC top (QFP leads, QFN/BGA package edge)
  RING_GAP_MM     0.5   near-IC parts are the first ring of small parts: their distances to the IC top, sorted,
                        are cut at the first gap wider than this
  PAD_MM          0.5   pads and terminals around a small part belong to the part
Classes: 0 IC boundary, 1 near-IC part, 2 IC top, 3 other small part, 4 other part, 5 bare board.
The target IC is the largest IC unless the evaluator names one. Each candidate takes the class at its centre.
"""
import numpy as np
from scipy import ndimage

PART_MIN_MM, IC_AREA_MM2, SMALL_AREA_MM2, TOP_SHARE = .15, 9.0, 6.0, .8
BAND_MM, RING_GAP_MM, PAD_MM = 1.0, .5, .5
CLASSES = ('IC boundary', 'near-IC part', 'IC top', 'other small part', 'other part', 'bare board')
PX = .05   # analysis pixel, mm


def relative_height(xs, ys, H):
    """0.05 mm square pixels, 3 x 3 median, minus a robust board plane."""
    k = int(round(PX / np.median(np.diff(ys))))
    n = H.shape[0] // k
    H = np.nanmedian(np.asarray(H, float)[:n * k].reshape(n, k, -1), axis=1)
    ys = np.asarray(ys)[:n * k].reshape(n, k).mean(1)
    H = ndimage.median_filter(np.nan_to_num(H, nan=np.nanmedian(H)), size=3)
    X, Y = np.meshgrid(xs, ys)
    low = H < np.percentile(H, 30)
    for _ in range(5):
        c = np.linalg.lstsq(np.c_[np.ones(low.sum()), X[low], Y[low]], H[low], rcond=None)[0]
        rel = H - (c[0] + c[1] * X + c[2] * Y)
        low = np.abs(rel) < .08
    return xs, ys, rel


def _largest(mask):
    lab, n = ndimage.label(mask)
    if n <= 1:
        return mask
    return lab == 1 + int(np.argmax(ndimage.sum(mask, lab, np.arange(1, n + 1))))


def segment(rel, target=None):
    """Class per pixel and the first-ring distance cut (mm)."""
    lab, n = ndimage.label(rel > PART_MIN_MM)
    idx = np.arange(1, n + 1)
    area = ndimage.sum(np.ones_like(rel), lab, idx) * PX * PX
    ics = [i for i, a in zip(idx, area) if a >= IC_AREA_MM2]
    target = target or max(ics, key=lambda i: area[i - 1])
    comp = lab == target
    top = _largest(ndimage.binary_opening(comp & (rel >= TOP_SHARE * np.percentile(rel[comp], 95)), iterations=4))
    out = ndimage.distance_transform_edt(~top) * PX
    small_ids = [i for i, a in zip(idx, area) if PX * PX * 20 <= a <= SMALL_AREA_MM2]
    other_ic = np.isin(lab, [i for i in ics if i != target])
    small = np.isin(lab, small_ids)
    pad = int(PAD_MM / PX)
    cls = np.full(rel.shape, 5)
    cls[ndimage.binary_dilation((lab > 0) & ~small & ~comp & ~other_ic | other_ic, iterations=pad)] = 4
    cls[ndimage.binary_dilation(small, iterations=pad)] = 3
    d = sorted(out[lab == i].min() for i in small_ids)
    cut = next((a for a, b in zip(d, d[1:]) if b - a > RING_GAP_MM), d[-1])
    near = [i for i in small_ids if out[lab == i].min() <= cut]
    cls[ndimage.binary_dilation(np.isin(lab, near), iterations=pad) & ~top] = 1
    cls[top] = 2
    cls[(out > 0) & (out <= BAND_MM)] = 0
    return cls, cut


def candidate_classes(xs, ys, cls, map_xy):
    """Class at the centre of each candidate (map coordinates)."""
    map_xy = np.asarray(map_xy, float)
    i = np.abs(ys[None, :] - map_xy[:, 1:2]).argmin(1)
    j = np.abs(xs[None, :] - map_xy[:, 0:1]).argmin(1)
    return cls[i, j].astype(int)


def classes_from_map(path, map_xy):
    z = np.load(path)
    xs, ys, rel = relative_height(z['xs_mm'], z['ys_mm'], z['height_mm'])
    cls, _ = segment(rel)
    return candidate_classes(xs, ys, cls, map_xy)
