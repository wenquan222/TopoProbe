"""On-the-fly laser surface scan (Section III-B, scan side).

Each Y row is one constant-speed X move. A background thread reads every HL-G1 frame with a host
timestamp; the main thread polls the X encoder during the move. A robust constant-velocity fit
x(t) = a + v t of the encoder samples maps each laser frame to X; topoprobe.surface.rebin_line then
bins the frames. A row is kept only if it passes every quality gate (QUALITY), otherwise it is
rescanned up to MAX_RETRIES times; three consecutive failed rows stop the scan. Raw laser frames and
encoder samples of every attempt are saved.

Settings of the paper: X 64-110 mm, Y 69-106 mm, Y pitch 0.01 mm, 5 mm/s, 0.3 mm lead-in, 0.5 s settle,
lines from large to small X, return at 100 mm/s, scan Z 21.0 mm.
"""
import json, re, threading, time
from pathlib import Path

import numpy as np

from .hardware import PORTS, Stage

SPEED, LEAD_MM, SETTLE_S, RETURN_SPEED, ACCEL_G = 5.0, .3, .5, 100, .2
MAX_RETRIES, MAX_CONSECUTIVE_FAILURES = 2, 3
ENCODER_RES_MM = .001
QUALITY = dict(max_speed_error=.02, max_fit_rms_um=10.0, max_gap_factor=1.8, min_roi_frame_ratio=.90)


class LaserStream:
    """Continuous HL-G1 reader; timestamps are taken on receipt of each complete '\\r' frame."""

    def __init__(self, port=PORTS['laser']):
        import serial
        self.ser = serial.Serial(port, 38400, timeout=.1)
        self.t, self.d, self.fail, self.lock = [], [], 0, threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        self.ser.reset_input_buffer()
        self.ser.read_until(b'\r')                  # drop a possibly partial first frame
        while not self.stop_event.is_set():
            raw = self.ser.read_until(b'\r')
            t = time.perf_counter()
            if not raw:
                continue
            m = re.search(rb'%\d{2}\$([+-]\d{7})', raw)
            with self.lock:
                if m:
                    self.t.append(t)
                    self.d.append(int(m.group(1)) / 10000)
                else:
                    self.fail += 1

    def mark(self):
        with self.lock:
            return len(self.t), self.fail

    def since(self, mark):
        with self.lock:
            return np.array(self.t[mark[0]:]), np.array(self.d[mark[0]:]), self.fail - mark[1]

    def close(self):
        self.stop_event.set()
        self.thread.join(2)
        self.ser.close()


def fit_velocity(t, x, rtt):
    """Robust x(t) = a + v t: drop slow encoder queries, then iterate weighted MAD rejection."""
    t, x, rtt = map(lambda a: np.asarray(a, float), (t, x, rtt))
    rmed = float(np.median(rtt))
    use = rtt <= rmed + max(4 * 1.4826 * float(np.median(abs(rtt - rmed))), .25 * rmed)

    def line(idx):
        w = 1 / np.maximum(rtt[idx], .25 * rmed) ** 2
        tc = t[idx].mean()
        A = np.column_stack([t[idx] - tc, np.ones(idx.size)]) * np.sqrt(w)[:, None]
        (v, xc), *_ = np.linalg.lstsq(A, x[idx] * np.sqrt(w), rcond=None)
        return xc - v * tc, v

    for _ in range(5):
        a, v = line(np.flatnonzero(use))
        r = x - (a + v * t)
        c = np.median(r[use])
        new = use & (abs(r - c) <= max(4 * 1.4826 * float(np.median(abs(r[use] - c))), .75 * ENCODER_RES_MM))
        if np.array_equal(new, use):
            break
        use = new
    a, v = line(np.flatnonzero(use))
    rms_um = float(np.sqrt(np.mean((x[use] - (a + v * t[use])) ** 2)) * 1000)
    return a, v, rms_um, use


def scan_row(stage, laser, x_start, x_end, y, z):
    """One unidirectional X line at fixed Y and Z; returns laser frames and encoder samples."""
    p = stage.position()
    if abs(p['z'] - z) > .01:
        stage.move([p['x'], p['y'], z])
    stage.move([x_start, stage.position()['y'], z], RETURN_SPEED, ACCEL_G)
    stage.move([x_start, y, z], RETURN_SPEED, ACCEL_G)
    time.sleep(SETTLE_S)
    mark = laser.mark()
    a = format(round(ACCEL_G * 100), '04X')
    stage._send(stage.STATION + '23407' + a + a + format(round(SPEED), '04X')
                + ''.join(stage._coord(v) for v in (x_end, y, z)))
    enc, expected = [], abs(x_end - x_start) / SPEED
    t_launch = time.perf_counter()
    deadline = t_launch + expected + 5
    while time.perf_counter() < deadline:
        t0 = time.perf_counter()
        r = stage._send(stage.STATION + '21201')
        t1 = time.perf_counter()
        v = int(r[16:24], 16)
        x = (v - 2 ** 32 if v >= 2 ** 31 else v) / 1000
        busy = bool(int(r[8:10], 16) & 1)
        enc.append((.5 * (t0 + t1), x, t1 - t0))
        if not busy and abs(x - x_end) <= .01 and .5 * (t0 + t1) - t_launch >= max(.05, .5 * expected):
            break
    else:
        stage._send(stage.STATION + '2380700')
        raise RuntimeError('line move timeout')
    time.sleep(.05)
    stage.last = None                        # the scan move is not tracked by Stage.position()
    t, d, fail = laser.since(mark)
    return t, d, fail, np.array(enc)


def line_quality(t, d, fail, enc, x0, x1, direction=-1):
    """Fit encoder samples inside the ROI (plus the usable lead-in), map frames to X, apply QUALITY."""
    accel = ACCEL_G * 9806.65
    guard = max(2 * SPEED ** 2 / (2 * accel), ENCODER_RES_MM)
    lead = max(LEAD_MM - guard, 0.)
    inside = (enc[:, 1] >= x0 - lead) & (enc[:, 1] <= x1 + lead)
    a, v, rms_um, used = fit_velocity(enc[inside, 0], enc[inside, 1], enc[inside, 2])
    x = a + v * t
    dt = np.diff(t)
    rate = (len(t) - 1) / (t[-1] - t[0])
    roi = (x >= x0) & (x <= x1)
    expected = rate * (x1 - x0) / abs(v)
    ex = enc[inside, 1][used]
    q = dict(v_mm_s=v, speed_error=abs(v - direction * SPEED) / SPEED, fit_rms_um=rms_um,
             gap_count=int(np.sum(dt > QUALITY['max_gap_factor'] * np.median(dt))),
             parse_fail=int(fail), roi_frame_ratio=float(roi.sum() / expected),
             encoder_monotonic=bool(np.all(direction * np.diff(ex) >= -ENCODER_RES_MM)))
    q['passed'] = bool(q['speed_error'] <= QUALITY['max_speed_error'] and q['fit_rms_um'] <= QUALITY['max_fit_rms_um']
                       and q['gap_count'] == 0 and q['parse_fail'] == 0 and q['encoder_monotonic']
                       and q['roi_frame_ratio'] >= QUALITY['min_roi_frame_ratio'] and direction * v > 0)
    return x, q


def scan(out_dir, x_range=(64.0, 110.0), y_range=(69.0, 106.0), pitch=.01, z=21.0, stage=None, laser=None):
    """Full surface scan; writes combined_line_NNNN.npz (laser_x_mapped_mm, laser_distance_mm) per row,
    the input of topoprobe.surface.height_map. Rows already written are skipped (resume)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stage = stage or Stage()
    laser = laser or LaserStream()
    x_start, x_end = x_range[1] + LEAD_MM, x_range[0] - LEAD_MM
    ys = np.round(np.arange(y_range[0], y_range[1] + pitch / 2, pitch), 6)
    failures = 0
    with (out / 'scan_log.jsonl').open('a', encoding='utf-8') as log:
        for j, y in enumerate(ys, 1):
            if (out / f'combined_line_{j:04d}.npz').exists():
                continue
            for attempt in range(1, MAX_RETRIES + 2):
                t, d, fail, enc = scan_row(stage, laser, x_start, x_end, float(y), z)
                x, q = line_quality(t, d, fail, enc, *x_range)
                np.savez_compressed(out / f'line_{j:04d}_attempt_{attempt:02d}.npz', laser_t=t,
                                    laser_distance_mm=d, encoder=enc, laser_x_mapped_mm=x)
                log.write(json.dumps(dict(row=j, y_mm=float(y), attempt=attempt, **q)) + '\n')
                log.flush()
                if q['passed']:
                    np.savez_compressed(out / f'combined_line_{j:04d}.npz', laser_x_mapped_mm=x, laser_distance_mm=d)
                    failures = 0
                    break
            else:
                failures += 1
                if failures >= MAX_CONSECUTIVE_FAILURES:
                    raise RuntimeError(f'{failures} consecutive rows failed; scan stopped')
    return out
