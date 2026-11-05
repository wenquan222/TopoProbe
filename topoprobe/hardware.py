"""Bench hardware used for every measurement in the paper.

  Stage        IAI TT three-axis stage over IAI protocol B (38400 baud). Z is positive downward.
  Laser        Panasonic HL-G1 displacement sensor in continuous RS-422 output (38400 baud).
  Target       STM32F429 software AES-128 over UART (115200 baud): send 16 plaintext + 16 key bytes,
               receive 16 ciphertext bytes; PC6 is high during aes_cipher() and triggers the scope.
  capture()    PicoScope 3403D: A = PC6 trigger (rising, 200 mV), B = Langer probe through BT706,
               125 MS/s, 2 us before and 20 us after the trigger, one raw .npz per trace.

Port names are those of the measurement PC; edit PORTS for another machine. The safety checks of
the runs (single-axis moves, motion box, position read-back, servo/idle flags, trigger pulse,
ciphertext check, clipping) are kept; keep them whenever a populated board is mounted.
"""
import hashlib, json, re, threading, time
from pathlib import Path

import numpy as np

from .leakage import aes128_encrypt

PORTS = dict(stage='COM4', laser='COM13', target='COM14')
KEY = bytes.fromhex('00102030405060708090a0b0c0d0e0f0')
RANGE_MV = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
A_RANGE, B_RANGE = 6, 2                  # 1 V trigger channel, 50 mV probe channel
TIMEBASE, PRE_S, POST_S = 3, 2e-6, 20e-6  # 125 MS/s


# ---------------------------------------------------------------- stage
class Stage:
    STATION = '99'                       # IAI station 153
    AXES = {'x': 0x01, 'y': 0x02, 'z': 0x04}

    def __init__(self, port=PORTS['stage'], box=((0, 200), (0, 200), (0, 50)), journal=None):
        import serial
        self.ser = serial.Serial(port, 38400, bytesize=8, parity='N', stopbits=1, timeout=1)
        time.sleep(.3)
        self.ser.reset_input_buffer()
        self.box, self.journal, self.last = box, journal, None

    def _send(self, body):
        frame = '!' + body
        frame = (frame + format(sum(frame.encode()) & 255, '02X') + '\r\n').encode()
        self.ser.reset_input_buffer()
        self.ser.write(frame)
        reply = self.ser.read_until(b'\n').decode('ascii', 'replace').strip()
        if self.journal:
            self.journal.write(json.dumps(dict(t=time.time(), tx=body, rx=reply)) + '\n')
        if not reply.startswith('#') or sum(reply[:-2].encode()) & 255 != int(reply[-2:], 16):
            raise RuntimeError('stage reply invalid: ' + reply)
        return reply

    @staticmethod
    def _coord(v):
        return format(round(v * 1000) & 0xFFFFFFFF, '08X')

    def status(self):
        pos, flags = {}, {}
        for name, bit in self.AXES.items():
            r = self._send(self.STATION + '212' + format(bit, '02X'))
            if r[11:14] != '000':
                raise RuntimeError('axis error: ' + r)
            v = int(r[16:24], 16)
            pos[name] = (v - 2 ** 32 if v >= 2 ** 31 else v) / 1000
            flags[name] = int(r[8:10], 16)
        return pos, flags

    def position(self):
        pos, flags = self.status()
        if not all((f & 14) == 12 and not f & 33 for f in flags.values()):
            raise RuntimeError('stage not referenced, servo off or busy')
        if self.last and max(abs(pos[k] - self.last[k]) for k in 'xyz') > .04:
            raise RuntimeError('unexpected position change')
        self.last = pos
        return pos

    def move(self, target, speed=2.0, accel=.05):
        """Absolute move of a single axis; refuses targets outside the motion box."""
        target = [round(float(v), 3) for v in target]
        if not all(lo <= v <= hi for v, (lo, hi) in zip(target, self.box)):
            raise RuntimeError('target outside motion box')
        p = self.position()
        if sum(abs(p[k] - v) > .035 for k, v in zip('xyz', target)) > 1:
            raise RuntimeError('one axis at a time')
        if max(abs(p[k] - v) for k, v in zip('xyz', target)) < .01:
            return p
        a = format(round(accel * 100), '04X')
        motion = a + a + format(round(speed), '04X')
        try:
            self._send(self.STATION + '23407' + motion + ''.join(self._coord(v) for v in target))
            time.sleep(.15)
            t0 = time.time()
            while any(int(self._send(self.STATION + '212' + format(b, '02X'))[8:10], 16) & 1 for b in (1, 2, 4)):
                if time.time() - t0 > 30:
                    raise RuntimeError('move timeout')
                time.sleep(.05)
            time.sleep(.4)
        except BaseException:
            self._send(self.STATION + '2380700')     # software stop
            raise
        q, flags = self.status()
        if max(abs(q[k] - v) for k, v in zip('xyz', target)) > .045 or any(f != 28 for f in flags.values()):
            raise RuntimeError('move completion failed')
        self.last = q
        return self.position()

    def close(self):
        self.ser.close()


# ---------------------------------------------------------------- laser
class Laser:
    """Background reader of HL-G1 frames '%01$+dddddd' (distance in 0.1 um)."""

    def __init__(self, port=PORTS['laser']):
        import serial
        self.ser = serial.Serial(port, 38400, timeout=.1)
        self.frames, self.errors = [], []
        self.cond = threading.Condition()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while not self.stop.is_set():
                packet = self.ser.read_until(b'\r')
                if not packet:
                    continue
                m = re.search(rb'%\d{2}\$([+-]\d{7})', packet)
                with self.cond:
                    self.frames.append((time.perf_counter(), int(m.group(1)) / 10000 if m else None))
                    self.cond.notify_all()
        except Exception as e:                      # surface reader errors to the caller
            with self.cond:
                self.errors.append(repr(e))
                self.cond.notify_all()

    def height(self, stage, n=32):
        """Surface height (map frame, upward positive) under the laser at the current stage pose."""
        time.sleep(.25)
        before, stamp = stage.position(), time.perf_counter()
        with self.cond:
            end = time.monotonic() + 4
            while True:
                if self.errors:
                    raise RuntimeError(self.errors[0])
                valid = [d for t, d in self.frames if t >= stamp and d is not None]
                if len(valid) >= n:
                    break
                if time.monotonic() > end:
                    raise RuntimeError('no valid laser frames')
                self.cond.wait(.05)
        after = stage.position()
        if max(abs(before[k] - after[k]) for k in 'xyz') > .03:
            raise RuntimeError('stage moved during laser read')
        d = np.array(valid[:n])
        if np.any(abs(d) > 4):
            raise RuntimeError('laser out of range')
        return dict(height_mm=-(after['z'] + float(d.mean())), sigma_um=float(d.std() * 1000))

    def close(self):
        self.stop.set()
        self.thread.join(2)
        self.ser.close()


# ---------------------------------------------------------------- target + scope
def _handshake(ser):
    """Known-answer exchange to align the 32-byte UART frame before arming the scope."""
    pt = bytes.fromhex('00112233445566778899aabbccddeeff')
    for attempt in range(3):
        if attempt:
            time.sleep(1.2)
        if ser.in_waiting:
            ser.read(ser.in_waiting)
        ser.write(pt + KEY)
        ser.flush()
        if ser.read(16) == aes128_encrypt(pt, KEY) and not ser.in_waiting:
            return
    raise RuntimeError('UART handshake failed')


def capture(folder, n, seed, port=PORTS['target']):
    """Acquire n traces at the current probe position into `folder` (one trace_XXX.npz each).

    Plaintexts: for every 256 traces, each byte runs through an independent random permutation of
    0..255 (seeded). A trace is kept only if the ciphertext is correct, PC6 shows a >=10 us pulse at
    the trigger point, and channel B does not clip.
    """
    import ctypes
    import serial
    from picosdk.functions import assert_pico_ok
    from picosdk.ps3000a import ps3000a as ps

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    table = np.vstack([np.column_stack([rng.permutation(256) for _ in range(16)])
                       for _ in range((n + 255) // 256)]).astype(np.uint8)[:n]
    handle = ctypes.c_int16()
    status = int(ps.ps3000aOpenUnit(ctypes.byref(handle), None))
    if status in (282, 286):
        assert_pico_ok(ps.ps3000aChangePowerSource(handle, status))
    else:
        assert_pico_ok(status)
    ser, records = None, []
    try:
        assert_pico_ok(ps.ps3000aSetChannel(handle, 0, 1, 1, A_RANGE, 0.))
        assert_pico_ok(ps.ps3000aSetChannel(handle, 1, 1, 1, B_RANGE, 0.))
        for ch in (2, 3):
            assert_pico_ok(ps.ps3000aSetChannel(handle, ch, 0, 1, 8, 0.))
        maxadc = ctypes.c_int16()
        assert_pico_ok(ps.ps3000aMaximumValue(handle, ctypes.byref(maxadc)))
        interval, maxsamples = ctypes.c_float(), ctypes.c_int32()
        assert_pico_ok(ps.ps3000aGetTimebase2(handle, TIMEBASE, round((PRE_S + POST_S) * 125e6),
                                              ctypes.byref(interval), 1, ctypes.byref(maxsamples), 0))
        dt = interval.value * 1e-9
        pre, post = round(PRE_S / dt), round(POST_S / dt)
        threshold = round(200. / RANGE_MV[A_RANGE] * maxadc.value)
        assert_pico_ok(ps.ps3000aSetSimpleTrigger(handle, 1, 0, threshold, 2, 0, 0))
        A, B = np.empty(pre + post, np.int16), np.empty(pre + post, np.int16)
        assert_pico_ok(ps.ps3000aSetDataBuffer(handle, 0, A.ctypes.data, pre + post, 0, 0))
        assert_pico_ok(ps.ps3000aSetDataBuffer(handle, 1, B.ctypes.data, pre + post, 0, 0))
        ser = serial.Serial(port=None, baudrate=115200, timeout=2., write_timeout=2.)
        ser.dtr = ser.rts = False
        ser.port = port
        ser.open()
        time.sleep(1.2)
        last = None
        for i in range(n):
            if last is None or time.monotonic() - last > .5:
                assert_pico_ok(ps.ps3000aStop(handle))
                _handshake(ser)
            pt = table[i].tobytes()
            assert_pico_ok(ps.ps3000aRunBlock(handle, pre, post, TIMEBASE, 1, None, 0, None, None))
            time.sleep(.001)
            ser.write(pt + KEY)
            ser.flush()
            ct = ser.read(16)
            last = time.monotonic()
            ready, deadline = ctypes.c_int16(), time.monotonic() + 3
            while not ready.value and time.monotonic() < deadline:
                assert_pico_ok(ps.ps3000aIsReady(handle, ctypes.byref(ready)))
                if not ready.value:
                    time.sleep(.005)
            if not ready.value:
                raise RuntimeError('no hardware-triggered capture')
            got, overflow = ctypes.c_uint32(pre + post), ctypes.c_int16()
            assert_pico_ok(ps.ps3000aGetValues(handle, 0, ctypes.byref(got), 1, 0, 0, ctypes.byref(overflow)))
            a, b = A[:got.value].copy(), B[:got.value].copy()
            if ct != aes128_encrypt(pt, KEY):
                raise RuntimeError('wrong ciphertext at trace %d' % i)
            high = a.astype(float) * RANGE_MV[A_RANGE] / maxadc.value >= 200
            edges = np.flatnonzero(np.diff(np.r_[False, high, False]))
            pulses = [(s, e) for s, e in zip(edges[::2], edges[1::2]) if (e - s) * dt >= 10e-6]
            if not any(abs(s - pre) * dt <= 5e-6 for s, _ in pulses):
                raise RuntimeError('trigger pulse missing at trace %d' % i)
            if overflow.value or np.any(abs(b.astype(np.int32)) >= .98 * maxadc.value):
                raise RuntimeError('channel B clipped at trace %d' % i)
            path = folder / f'trace_{i:03d}.npz'
            with path.open('xb') as f:
                np.savez_compressed(f, A_adc=a, B_adc=b, interval_ns=interval.value, pre_samples=pre,
                                    max_adc=maxadc.value, A_range_mV=RANGE_MV[A_RANGE], B_range_mV=RANGE_MV[B_RANGE],
                                    plaintext=np.frombuffer(pt, np.uint8), key=np.frombuffer(KEY, np.uint8),
                                    ciphertext=np.frombuffer(ct, np.uint8))
            records.append(dict(index=i, raw_file=path.name, raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                ciphertext_ok=True, hardware_trigger_confirmed_in_saved_A=True))
    finally:
        if ser is not None:
            ser.close()
        ps.ps3000aStop(handle)
        ps.ps3000aCloseUnit(handle)
    (folder / 'result.json').write_text(json.dumps(dict(seed=seed, count=n, records=records), indent=1), encoding='utf-8')
    return folder


# ---------------------------------------------------------------- terrain-following executor
class Executor:
    """Moves the EM probe to a candidate at map height + gap. Coordinates are stage mm, Z downward.

    probe_z(h) = z_ref - h - gap, with z_ref the calibrated sensor-to-probe Z constant and h the
    registered map height of the candidate (session-corrected). Between candidates the probe rises to
    the bridge plane (Z = 30 mm, cleared above every component inside `em_box`), moves X then Y, and
    descends; from the parked position (Z near 0) it moves at the top plane instead.
    Values used in the paper: z_ref = 14.6698 mm, probe Z limited to 36.0-39.2 mm.
    """

    def __init__(self, stage, z_ref, em_box, bridge_z=30.0, top_z=.008, z_range=(36.0, 39.2)):
        self.stage, self.z_ref, self.em_box = stage, z_ref, em_box
        self.bridge_z, self.top_z, self.z_range = bridge_z, top_z, z_range

    def probe_z(self, height_mm, gap_mm):
        z = round(self.z_ref - height_mm - gap_mm, 3)
        if not self.z_range[0] <= z <= self.z_range[1]:
            raise RuntimeError('probe Z outside the cleared range')
        return z

    def _inside(self, xy):
        (x0, y0), (x1, y1) = self.em_box
        return x0 - .05 <= xy[0] <= x1 + .05 and y0 - .05 <= xy[1] <= y1 + .05

    def go(self, xy, height_mm, gap_mm=1.0):
        z = self.probe_z(height_mm, gap_mm)
        p = self.stage.position()
        if max(abs(p[k] - v) for k, v in zip('xyz', [*xy, z])) <= .025:
            return p
        travel = self.bridge_z if p['z'] >= self.bridge_z - .05 else self.top_z
        if travel == self.bridge_z and not (self._inside([p['x'], p['y']]) and self._inside(xy)):
            raise RuntimeError('bridge-plane move outside the cleared EM rectangle')
        self.stage.move([p['x'], p['y'], travel])                   # raise
        p = self.stage.position()
        self.stage.move([xy[0], p['y'], travel], 5)                 # X
        p = self.stage.position()
        self.stage.move([p['x'], xy[1], travel], 5)                 # Y
        self.stage.move([xy[0], xy[1], z - .8])                     # approach
        return self.stage.move([xy[0], xy[1], z], 1)                # last 0.8 mm at 1 mm/s

    def lift(self):
        """Raise to the bridge plane after a capture (or to the top plane if above it)."""
        p = self.stage.position()
        return self.stage.move([p['x'], p['y'], self.bridge_z if p['z'] >= self.bridge_z else self.top_z])
