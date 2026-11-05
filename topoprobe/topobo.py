"""TopoBO: terrain-guided Bayesian-optimization search over board positions.

Implements Section III-C of the paper:
  * terrain groups and priority g(p)                      (Eqs. 2, 4)
  * terrain-stratified farthest-point initialization       (Eq. 3)
  * Gibbs-kernel GP with terrain-shortened length scale   (Eqs. 6-7)
  * noisy expected improvement over the joint posterior   (Eq. 8)
  * coverage-weighted selection                            (Eqs. 9-10)

`TopoBO(..., priority=zeros, initial=uniform)` is the terrain-free ablation of Section IV-C.
All arithmetic follows the code that produced the paper's runs; `checks/replay_all.py`
confirms that every recorded query of the ten TopoProbe and terrain-free runs is reproduced.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky, solve_triangular
from scipy.optimize import minimize
from scipy.special import log_ndtr, logsumexp, ndtri
from scipy.stats import qmc

# Parameters used in the paper (Section IV-A, "TopoProbe configuration").
EDGE_BAND_MM = 1.0      # r: interior is d_C > r, exterior edge is -r <= d_C <= 0
QUOTA = 6               # Q: interior and edge picks each; n0 = 4 + 2Q = 16
ELL_MIN_MM, ELL_MAX_MM = 0.5, 2.0
TAU = 0.0022379412504050632   # fixed score noise (jackknife median)
XI = 0.01               # improvement margin
SOBOL_POWER = 8         # M = 2**8 = 256 Sobol draws
RC_MM = 2.0             # coverage radius
A_WEIGHT = B_WEIGHT = 4.0
NEI_SEED_BASE = 2026102601


# ---------------------------------------------------------------- terrain
def terrain_groups(chip_dist, r=EDGE_BAND_MM):
    """Interior T and exterior-edge E from the signed distance to the chip plateau (positive inside)."""
    chip_dist = np.asarray(chip_dist, float)
    return chip_dist > r, (chip_dist >= -r) & (chip_dist <= 0)


def terrain_priority(chip_dist, quota=QUOTA, r=EDGE_BAND_MM):
    """Eq. 4: g = 1 on the smaller group, |E|/|T| (or |T|/|E|) on the larger, 0 elsewhere."""
    top, edge = terrain_groups(chip_dist, r)
    g = quota / top.sum() * top + quota / edge.sum() * edge
    return g / g.max()


def initial_positions(ids, xy, chip_dist, seed, quota=QUOTA, r=EDGE_BAND_MM):
    """Four seeded quadrant points, then `quota` interior and `quota` edge picks, each the
    farthest (normalized XY) from those already chosen, requested quadrant rotating in seeded order."""
    xy = np.asarray(xy, float)
    norm = (xy - xy.min(0)) / np.ptp(xy, axis=0)
    quad = (norm[:, 0] >= .5).astype(int) + 2 * (norm[:, 1] >= .5).astype(int)
    top, edge = terrain_groups(chip_dist, r)
    rng = np.random.default_rng(seed)
    order = rng.permutation(4)
    chosen = [int(rng.choice(np.flatnonzero(quad == q))) for q in order]
    for step in range(2 * quota):
        group = top if step % 2 == 0 else edge
        target = (step // 2) % 4
        available = np.ones(len(ids), bool)
        available[chosen] = False
        available &= group
        for shift in range(4):
            pool = np.flatnonzero(available & (quad == order[(target + shift) % 4]))
            if len(pool):
                break
        else:
            raise ValueError('terrain group exhausted')
        d = ((norm[pool, None] - norm[chosen][None, :]) ** 2).sum(2).min(1)
        tied = pool[abs(d - d.max()) <= 1e-14]
        chosen.append(min(tied.tolist(), key=lambda j: ids[j]))
    return chosen


def uniform_initial_positions(n_candidates, seed, n0=4 + 2 * QUOTA):
    """Initialization of the terrain-free ablation."""
    return np.random.default_rng(seed).choice(n_candidates, n0, replace=False).astype(int).tolist()


# ---------------------------------------------------------------- GP
def gibbs_xy(a, b, la, lb):
    denom = la[:, None] ** 2 + lb[None, :] ** 2
    prefactor = (2 * la[:, None] * lb[None, :] / denom) ** (a.shape[1] / 2)
    return prefactor * np.exp(-np.sum((a[:, None] - b[None, :]) ** 2, axis=2) / denom)


def _factor(matrix):
    for jitter in (1e-10, 1e-8, 1e-6):
        try:
            return cho_factor(matrix + jitter * np.eye(len(matrix)), lower=True), jitter
        except np.linalg.LinAlgError:
            pass
    raise ValueError('covariance factorization failed')


class ScaleGP:
    """GP with Gibbs covariance, local scale ell_p = ell_0 / (1 + g(p)) (Eq. 6), fixed noise TAU.

    Free hyperparameters: log ell_0 (normalized units) and log sigma_f, fitted by marginal likelihood
    from two predetermined starts. theta keeps the 12-slot layout of the original implementation so
    that the arithmetic, including the noise bookkeeping in `_cov`, is reproduced exactly.
    """
    THETA0 = np.r_[np.log([.3, .2, .25, 1, 1, 1, 1, .5, .6, .05, .1]), .25]
    ACTIVE = np.array([2, 6])

    def __init__(self, priority, max_iter=200):
        self.g = np.asarray(priority, float)
        self.max_iter = max_iter

    def _k(self, ia, ib, theta):
        ell = np.exp(theta[2]) / (1 + self.g)
        return np.exp(2 * theta[6]) * gibbs_xy(self.xy[ia], self.xy[ib], ell[ia], ell[ib])

    def _cov(self, theta):
        n = len(self.train)
        p = np.exp(theta[:11])
        matrix = self._k(self.train, self.train, theta) + p[10] ** 2 * np.eye(n)
        matrix[-n:, -n:] += np.diag((self.noise / self.scale) ** 2 - np.exp(theta[10]) ** 2)
        return matrix

    def fit(self, xy, train, y, noise=TAU):
        xy = np.asarray(xy, float)
        span = np.ptp(xy, axis=0)
        self.xy = (xy - xy.min(axis=0)) / span
        self.train = np.asarray(train, int)
        y = np.asarray(y, float)
        self.noise = np.full(len(self.train), float(noise))
        self.mean, self.scale = float(y.mean()), max(float(y.std()), 1e-6)
        self.response = (y - self.mean) / self.scale
        bounds = [(float(np.log(max(ELL_MIN_MM / span))), float(np.log(min(ELL_MAX_MM / span)))), (-5, 2)]

        def expand(v):
            theta = self.THETA0.copy()
            theta[self.ACTIVE] = v
            return theta

        def nll(v):
            try:
                chol, _ = _factor(self._cov(expand(v)))
                alpha = cho_solve(chol, self.response)
                return float(.5 * self.response @ alpha + np.log(np.diag(chol[0])).sum()
                             + .5 * len(alpha) * np.log(2 * np.pi))
            except (ValueError, np.linalg.LinAlgError):
                return 1e100

        starts = [self.THETA0.copy(), self.THETA0.copy()]
        starts[1][2] = np.log(max(1 / span))
        fits = []
        for s in starts:
            opt = minimize(nll, s[self.ACTIVE], method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': self.max_iter, 'ftol': 1e-7})
            fits.append((opt, expand(opt.x)))
        good = [f for f in fits if f[0].success and np.isfinite(f[0].fun) and f[0].fun < 1e99]
        _, self.theta = min(good or fits, key=lambda f: f[0].fun)
        self.converged = bool(good)
        self.chol, _ = _factor(self._cov(self.theta))
        self.alpha = cho_solve(self.chol, self.response)
        return self

    def predict(self, ii):
        cross = self._k(ii, self.train, self.theta)
        prior = np.exp(self.theta[6]) ** 2
        solved = solve_triangular(self.chol[0], cross.T, lower=True)
        variance = prior - np.sum(solved * solved, axis=0)
        if np.any(variance < -1e-6):
            raise ValueError('negative posterior variance')
        return self.mean + self.scale * (cross @ self.alpha), self.scale * np.sqrt(np.maximum(variance, 0))

    def posterior_cov(self, ia, ib):
        va = solve_triangular(self.chol[0], self._k(ia, self.train, self.theta).T, lower=True)
        vb = solve_triangular(self.chol[0], self._k(ib, self.train, self.theta).T, lower=True)
        return self.scale ** 2 * (self._k(ia, ib, self.theta) - va.T @ vb)


# ---------------------------------------------------------------- acquisition
def log_ei(gain, std):
    """log of the analytic Gaussian EI, stable far into the lower tail."""
    gain, std = np.broadcast_arrays(np.asarray(gain, float), np.asarray(std, float))
    out = np.full(gain.shape, -np.inf)
    positive = std > 1e-14
    z = np.zeros_like(gain)
    z[positive] = gain[positive] / std[positive]
    lp = -.5 * z * z - .5 * np.log(2 * np.pi)
    with np.errstate(divide='ignore', invalid='ignore'):
        pos = positive & (z >= 0)
        out[pos] = np.log(std[pos]) + np.logaddexp(lp[pos], np.log(z[pos]) + log_ndtr(z[pos]))
        neg = positive & (z < 0) & (z >= -10)
        v = np.log(-z[neg]) + log_ndtr(z[neg]) - lp[neg]
        out[neg] = np.log(std[neg]) + lp[neg] + np.log(-np.expm1(v))
        tail = positive & (z < -10)
        a = -z[tail]
        u = 1 / (a * a)
        series = 1 - 3*u + 15*u**2 - 105*u**3 + 945*u**4 - 10395*u**5 + 135135*u**6 - 2027025*u**7 + 34459425*u**8
        out[tail] = np.log(std[tail]) + lp[tail] - 2 * np.log(a) + np.log(series)
        det = ~positive & (gain > 0)
        out[det] = np.log(gain[det])
    return out


def noisy_ei(model, remaining, xi=XI, seed=1, sobol_power=SOBOL_POWER):
    """Eq. 8: average analytic EI over 2**sobol_power joint draws of the latent scores at measured sites."""
    base = model.train
    mb, _ = model.predict(base)
    mc, sc = model.predict(remaining)
    bb = model.posterior_cov(base, base)
    bb = (bb + bb.T) / 2
    cb = model.posterior_cov(remaining, base)
    jitter = max(1e-14, 1e-10 * max(float(np.diag(bb).max()), 1e-10))
    chol = cholesky(bb + jitter * np.eye(len(base)), lower=True)
    zs = ndtri(np.clip(qmc.Sobol(len(base), scramble=True, seed=seed).random_base2(sobol_power), 1e-12, 1 - 1e-12))
    fb = mb + zs @ chol.T
    weights = cho_solve((chol, True), cb.T).T
    conditional = mc[:, None] + weights @ (fb - mb).T
    variance = sc * sc - np.sum(weights * cb, axis=1)
    sd = np.sqrt(np.maximum(variance, 0))
    threshold = fb.max(axis=1) + xi
    logs = log_ei(conditional - threshold[None, :], sd[:, None])
    return logsumexp(logs, axis=1) - np.log(len(zs)), mc, sc


# ---------------------------------------------------------------- search policy
class TopoBO:
    """Chooses the next position from the measurement history [(candidate_id, score), ...]."""

    def __init__(self, ids, xy_mm, priority, initial, a=A_WEIGHT, b=B_WEIGHT, rc=RC_MM):
        self.ids = list(ids)
        self.index = {c: i for i, c in enumerate(self.ids)}
        self.xy = np.asarray(xy_mm, float)
        self.g = np.asarray(priority, float)
        self.initial = list(initial)
        self.a, self.b, self.rc = a, b, rc

    def choose(self, history):
        n = len(history)
        if n < len(self.initial):
            return self.ids[self.initial[n]]
        rows = np.array([self.index[c] for c, _ in history])
        remaining = np.setdiff1d(np.arange(len(self.ids)), rows)
        model = ScaleGP(self.g).fit(self.xy, rows, [s for _, s in history])
        if not model.converged:
            raise RuntimeError('GP fit did not converge')
        logs, _, _ = noisy_ei(model, remaining, seed=NEI_SEED_BASE + n)
        dist = np.sqrt(((self.xy[remaining, None] - self.xy[rows][None, :]) ** 2).sum(2)).min(1)
        coverage = 1 - np.exp(-dist ** 2 / (2 * self.rc ** 2))
        utility = logs + np.log(1 + self.g[remaining] * (self.a + self.b * coverage))   # Eq. 10
        ties = np.flatnonzero(utility.max() - utility <= 1e-10)
        return self.ids[int(remaining[min(ties.tolist(), key=lambda k: self.ids[remaining[k]])])]
