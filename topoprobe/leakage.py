"""Leakage score and key-recovery metric used in the paper.

  score(x, pt, key)         Eq. 5: median over the 16 key bytes of the peak first-round S-box
                             Hamming-weight SNR (search score of TopoProbe and SCNIFFER)
  cpa_full_key(x, pt, key)   first-round HW CPA; True when all 16 true key bytes rank first
  recovery_traces(...)       attack metric of Section IV-A: smallest trace count from which the
                             full key ranks first at every later checkpoint, located to one trace
                             by bisection between the last failing and first succeeding checkpoint
  preprocess(...)            baseline subtraction (-1.8..-0.2 us) and analysis window (0..20 us)

Traces are traces-by-samples arrays in mV with a time axis in microseconds relative to the trigger.
Recorded scores and recovery counts were recomputed from the raw captures with this module.
"""
import numpy as np

BASELINE_US = (-1.8, -0.2)
WINDOW_US = (0.0, 20.0)
CHECKPOINTS = [16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192]


def _gf_mul(a, b):
    r = 0
    for _ in range(8):
        if b & 1:
            r ^= a
        a = ((a << 1) ^ (0x11b if a & 0x80 else 0)) & 255
        b >>= 1
    return r


def _sbox(v):
    inv = 0
    if v:
        inv, base, power = 1, v, 254
        while power:
            if power & 1:
                inv = _gf_mul(inv, base)
            base, power = _gf_mul(base, base), power >> 1
    rot = 0
    for s in (1, 2, 3, 4):
        rot ^= ((inv << s) | (inv >> (8 - s))) & 255
    return inv ^ rot ^ 0x63


SBOX = np.array([_sbox(i) for i in range(256)], dtype=np.uint8)
HW = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
MODELS = HW[SBOX[np.arange(256)[:, None] ^ np.arange(256)[None, :]]].astype(float)   # [plaintext, key guess]


def load_capture(folder, n=None):
    """Read a capture folder: result.json lists one raw .npz per trace (B_adc, plaintext, key,
    ciphertext, scaling). Every ciphertext is re-checked. Returns preprocessed float32 traces,
    plaintexts and the key. float32 matches the precision used for the paper's numbers."""
    import json
    from pathlib import Path
    folder = Path(folder)
    records = json.loads((folder / 'result.json').read_text(encoding='utf-8-sig'))['records'][:n]
    x, pts = [], []
    for r in records:
        with np.load(folder / r['raw_file'], allow_pickle=False) as z:
            b, pt, key, ct = z['B_adc'], z['plaintext'], z['key'], z['ciphertext']
            if aes128_encrypt(bytes(pt), bytes(key)) != bytes(ct):
                raise ValueError('ciphertext check failed: ' + r['raw_file'])
            x.append(b.astype(float) * float(z['B_range_mV']) / float(z['max_adc']))
            pts.append(np.array(pt, np.uint8))
            t = (np.arange(len(b)) - int(z['pre_samples'])) * float(z['interval_ns']) * .001
    y, _ = preprocess(np.array(x, np.float32), t)
    return y, np.array(pts), np.array(key, np.uint8)


def preprocess(traces, time_us, baseline_us=BASELINE_US, window_us=WINDOW_US):
    x = np.asarray(traces, float)
    t = np.asarray(time_us, float)
    base = (t >= baseline_us[0]) & (t < baseline_us[1])
    x = x - x[:, base].mean(axis=1, keepdims=True)
    keep = (t >= window_us[0]) & (t < window_us[1])
    return x[:, keep].astype(np.float32), t[keep]


def snr(traces, labels):
    """Var(E[X|C]) / E[Var(X|C)] per time sample, empirical class weights."""
    x = np.asarray(traces, float)
    classes, inv, counts = np.unique(labels, return_inverse=True, return_counts=True)
    x = x - x.mean(axis=0)
    sums = np.zeros((len(classes), x.shape[1]))
    np.add.at(sums, inv, x)
    between = np.sum(sums * sums / counts[:, None], axis=0) / len(x)
    total = np.mean(x * x, axis=0)
    within = np.maximum(total - between, 0)
    defined = within > np.maximum(total * 1e-12, 1e-30)
    return np.divide(between, within, out=np.full_like(total, np.nan), where=defined)


def score(traces, pt, key, n=1024):
    """Eq. 5 on the first n traces."""
    pt = np.asarray(pt)
    peaks = [float(np.nanmax(snr(traces[:n], HW[SBOX[pt[:n, b] ^ key[b]]]))) for b in range(16)]
    return float(np.median(peaks))


def cpa_full_key(traces, pt, key, chunk=256):
    """First-round HW CPA over the window; ranks use the pessimistic tie rule."""
    x = np.asarray(traces, float)
    pt = np.asarray(pt, np.uint8)
    n, t = x.shape
    x = x - x.mean(axis=0)
    energy = np.einsum('ij,ij->j', x, x)
    for byte in range(16):
        counts = np.bincount(pt[:, byte], minlength=256).astype(float)
        mean_h = counts @ MODELS / n
        eh = np.maximum(counts @ (MODELS * MODELS) - n * mean_h * mean_h, 0)
        best = np.zeros(256)
        for lo in range(0, t, chunk):
            hi = min(t, lo + chunk)
            groups = np.zeros((256, hi - lo))
            np.add.at(groups, pt[:, byte], x[:, lo:hi])
            cov = MODELS.T @ groups
            denom = np.sqrt(eh[:, None] * energy[None, lo:hi])
            rho = np.clip(np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 1e-20), -1, 1)
            best = np.maximum(best, np.max(abs(rho), axis=1))
        truth = best[int(key[byte])]
        if np.sum(best > truth + 1e-12) + np.sum(abs(best - truth) <= 1e-12) != 1:
            return False
    return True


def recovery_traces(traces, pt, key, checkpoints=CHECKPOINTS):
    """Attack metric. Returns None if the batch never sustains full-key recovery."""
    counts = [c for c in checkpoints if c <= len(traces)]
    ok = [cpa_full_key(traces[:c], pt[:c], key) for c in counts]
    first = next((i for i in range(len(ok)) if all(ok[i:])), None)
    if first is None:
        return None
    hi = counts[first]
    lo = counts[first - 1] if first else 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if cpa_full_key(traces[:mid], pt[:mid], key):
            hi = mid
        else:
            lo = mid
    return hi


# ------------------------------------------------ AES-128 reference (ciphertext check of captures)
def _round_keys(key):
    w = [list(key[i:i + 4]) for i in range(0, 16, 4)]
    rcon = 1
    for i in range(4, 44):
        t = w[-1].copy()
        if i % 4 == 0:
            t = [int(SBOX[q]) for q in t[1:] + t[:1]]
            t[0] ^= rcon
            rcon = _gf_mul(rcon, 2)
        w.append([a ^ b for a, b in zip(w[i - 4], t)])
    return [bytes(sum(w[i:i + 4], [])) for i in range(0, 44, 4)]


def aes128_encrypt(plaintext, key):
    keys = _round_keys(bytes(key))
    s = [a ^ b for a, b in zip(plaintext, keys[0])]
    for rnd in range(1, 11):
        s = [int(SBOX[q]) for q in s]
        s = [s[4 * ((c + r) % 4) + r] for c in range(4) for r in range(4)]
        if rnd < 10:
            m = []
            for c in range(4):
                a = s[c * 4:c * 4 + 4]
                tot = a[0] ^ a[1] ^ a[2] ^ a[3]
                m.extend(a[i] ^ tot ^ _gf_mul(a[i] ^ a[(i + 1) % 4], 2) for i in range(4))
            s = m
        s = [a ^ b for a, b in zip(s, keys[rnd])]
    return bytes(s)
