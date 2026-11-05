"""Baselines of Section IV: SCNIFFER (gradient search) and Swarm in EM Hay (particle swarm).

Both run on the same 1 mm grid of candidate positions as TopoProbe and are re-implementations of the
published methods with the settings stated in Section IV-A:
  SCNIFFER  four starts, one uniform at random per spatial quadrant; two-cell gradient step;
            four charged neighbor queries per step.
  Swarm     three uniformly random particles; score = max(0, H(trace) - mean H(corner)), with
            H the matrix-based Renyi entropy of order 1.01 computed on 256-trace blocks; four
            corner background batches of 256 traces precede the search.
`checks/replay_baselines.py` confirms every recorded query of the ten baseline runs.
"""
import numpy as np
from scipy.spatial.distance import cdist, pdist

ALPHA = 1.01
BLOCK = 256


# ---------------------------------------------------------------- grid
class Grid:
    """Integer lattice over candidate positions (x and y must lie on uniform 1 mm axes)."""

    def __init__(self, ids, xy):
        xy = np.asarray(xy, float)
        self.ids = list(ids)
        self.x, self.y = np.unique(xy[:, 0]), np.unique(xy[:, 1])
        self.lookup, self.pos = {}, {}
        for i, (px, py) in enumerate(xy):
            q = (int(np.searchsorted(self.x, px)), int(np.searchsorted(self.y, py)))
            self.lookup[q] = i
            self.pos[i] = np.array(q, float)
        self.upper = np.array([len(self.x) - 1, len(self.y) - 1], float)

    def at(self, p):
        q = tuple(np.clip(np.floor(np.asarray(p) + .5), 0, self.upper).astype(int))
        return self.lookup.get(q)

    def corners(self):
        return [self.at(p) for p in ([0, 0], [self.upper[0], 0], [0, self.upper[1]], self.upper)]


def quadrant_starts(ids, xy, seed):
    """SCNIFFER initialization: one uniformly random candidate in each spatial quadrant."""
    xy = np.asarray(xy, float)
    upper = (xy >= (xy.min(0) + xy.max(0)) / 2).astype(int)
    rng = np.random.default_rng(seed)
    out = []
    for yy in range(2):
        for xx in range(2):
            pool = [i for i in range(len(ids)) if tuple(upper[i]) == (xx, yy)]
            out.append(ids[int(rng.choice(pool))])
    return out


# ---------------------------------------------------------------- SCNIFFER
class SCNIFFER:
    """ask() -> (candidate_id, purpose) or None when the method stops; tell(score) after each query."""

    def __init__(self, ids, xy, initial_ids, step_cells=2.0, max_iterations=64, stall_iterations=64):
        self.grid = Grid(ids, xy)
        index = {c: i for i, c in enumerate(ids)}
        self.queue = [('coarse', index[c], None) for c in initial_ids]
        self.step, self.max_iter, self.max_stall = step_cells, max_iterations, stall_iterations
        self.pending = self.current = self.best = None
        self.best_score, self.neighbors, self.iterations, self.stall = -np.inf, [], 0, 0
        self.done, self.stop_reason, self.boundary_exit = False, None, False

    def _stop(self, reason):
        self.done, self.stop_reason = True, reason
        return None

    def ask(self):
        if self.done:
            return None
        if not self.queue:
            if self.current is None:
                self.current = self.best
            if self.iterations >= self.max_iter or self.stall >= self.max_stall:
                return self._stop('iteration_or_stall_limit')
            point = self.grid.pos[self.current]
            for d in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                c = point + d
                if np.any(c < 0) or np.any(c > self.grid.upper):
                    return self._stop('edge_reached')
                i = self.grid.at(c)
                if i is None:
                    return self._stop('ineligible_neighbor')
                self.queue.append(('neighbor', i, list(d)))
            self.neighbors, self.cycle_best = [], self.best_score
        self.pending = self.queue.pop(0)
        return self.grid.ids[self.pending[1]], self.pending[0]

    def tell(self, score):
        purpose, i, direction = self.pending
        self.pending = None
        if score > self.best_score:
            self.best_score, self.best = score, i
        if purpose == 'neighbor':
            self.neighbors.append((direction, score))
            if self.queue:
                return
            g = np.mean([np.array(d) * s for d, s in self.neighbors], axis=0)
            if np.linalg.norm(g) < 1e-14:
                self._stop('zero_gradient'); return
            proposed = self.grid.pos[self.current] + self.step * g / np.linalg.norm(g)
            self.boundary_exit = bool(np.any(proposed < 0) or np.any(proposed > self.grid.upper))
            target = self.grid.at(proposed)
            if target is None:
                self._stop('ineligible_target'); return
            self.queue = [('advance', target, None)]
        elif purpose == 'advance':
            self.current = i
            self.iterations += 1
            self.stall = 0 if self.best_score > self.cycle_best + 1e-12 else self.stall + 1
            if self.boundary_exit:
                self._stop('clipped_to_edge_then_stopped')


# ---------------------------------------------------------------- Swarm entropy score
def renyi_entropy(x, width, alpha=ALPHA, ridge=1e-5):
    """Matrix-based Renyi entropy of a traces-by-samples block with Gaussian kernel exp(-d^2/width)."""
    x = np.asarray(x, float)
    e = np.maximum(np.linalg.eigvalsh(np.exp(-cdist(x, x, 'sqeuclidean') / width)) + ridge, 0)
    p = e / e.sum()
    p = p[p > 0]
    if abs(alpha - 1) < 1e-8:
        return float(-(p * np.log2(p)).sum())
    return float(np.log2(np.sum(p ** alpha)) / (1 - alpha))


def median_width(x):
    d = pdist(np.asarray(x, float), 'sqeuclidean')
    d = d[d > 1e-20]
    return float(np.median(d)) if len(d) else 1.


class EntropyScore:
    """Swarm fitness. Kernel width and baseline come from the four corner batches; each search batch is
    split into 256-trace blocks and the contrasts are averaged."""

    def fit_background(self, batches):
        self.width = float(np.median([median_width(b) for b in batches]))
        self.background = [renyi_entropy(b, self.width) for b in batches]
        self.baseline = float(np.mean(self.background))
        return self

    def __call__(self, x):
        x = np.asarray(x, float)
        blocks = [renyi_entropy(x[i:i + BLOCK], self.width) - self.baseline for i in range(0, len(x), BLOCK)]
        return max(0., float(np.mean(blocks)))


# ---------------------------------------------------------------- Swarm
class Swarm:
    """Particle swarm over the grid. ask() first returns the four corners ('background'),
    then particle positions ('search'); tell(score) is called after each query (score ignored for
    background, whose entropy enters through EntropyScore.fit_background)."""

    def __init__(self, ids, xy, seed, swarm_size=3, c1=1.5, c2=1.5, inertia=.7, max_iterations=64,
                 stall_iterations=64, tolerance=1e-3):
        self.grid = Grid(ids, xy)
        self.rng = np.random.default_rng(seed)
        self.n, self.c1, self.c2, self.w = swarm_size, c1, c2, inertia
        self.max_iter, self.max_stall, self.tol = max_iterations, stall_iterations, tolerance
        self.corners = self.grid.corners()
        self.n_background = 0
        self.particles = None
        self.gbest, self.gscore, self.generation, self.stall, self.cursor = None, -np.inf, 0, 0, 0
        self.done, self.stop_reason, self.pending = False, None, None

    def ask(self):
        if self.done:
            return None
        if self.n_background < 4:
            self.pending = ('background', self.corners[self.n_background])
            return self.grid.ids[self.pending[1]], 'background'
        if self.particles is None:
            starts = self.rng.choice(len(self.grid.ids), size=self.n, replace=False).tolist()
            self.particles = [dict(x=self.grid.pos[i].copy(), v=np.zeros(2), best=self.grid.pos[i].copy(),
                                   score=-np.inf) for i in starts]
            self.start_best = -np.inf
        if self.cursor == len(self.particles):
            self.generation += 1
            self.stall = self.stall + 1 if self.gscore - self.start_best <= self.tol else 0
            if self.gbest is None or self.generation >= self.max_iter or self.stall >= self.max_stall:
                self.done, self.stop_reason = True, 'generation_stall_or_no_success'
                return None
            self.start_best = self.gscore
            for p in self.particles:
                p['v'] = (self.w * p['v'] + self.c1 * self.rng.random(2) * (p['best'] - p['x'])
                          + self.c2 * self.rng.random(2) * (self.gbest - p['x']))
                p['x'] = np.clip(p['x'] + p['v'], 0, self.grid.upper)
            self.cursor = 0
        index = self.grid.at(self.particles[self.cursor]['x'])
        self.pending = ('search', index)
        return self.grid.ids[index], 'entropy_fitness'

    def tell(self, score):
        phase, index = self.pending
        self.pending = None
        if phase == 'background':
            self.n_background += 1
            return
        p, position = self.particles[self.cursor], self.grid.pos[index]
        if score > p['score']:
            p['score'], p['best'] = score, position.copy()
        if score > self.gscore:
            self.gscore, self.gbest = score, position.copy()
        self.cursor += 1
