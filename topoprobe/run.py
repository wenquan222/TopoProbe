"""TopoProbe main program: one search run on the bench, followed by CPA at the recommendations.

    python -m topoprobe.run --method topoprobe --seed 2026110304 --out runs/g4_topo
    python -m topoprobe.run --method plainbo   --seed 2026110504 --out runs/b4
    python -m topoprobe.run --method scniffer  --seed 2026110304 --out runs/g4_sc
    python -m topoprobe.run --method swarm     --seed 2026110304 --out runs/g4_sw

Inputs (paths in --setup JSON): the registered candidate table (results/surface/candidates.csv),
the signed distance of each candidate to the chip plateau (same table), the probe Z constant z_ref
and the cleared EM rectangle. Every query: move at map height + 1 mm, acquire 1024 traces, score.
Recommendations at 32/48/64 queries are validated with fresh traces; the recovery count is
leakage.recovery_traces. All motion stops on any check failure, leaving the probe raised.
"""
import argparse, csv, json, time
from pathlib import Path

import numpy as np

from . import baselines, leakage, topobo
from .hardware import Executor, Stage, capture

BUDGETS = (32, 48, 64)


def load_candidates(path):
    with open(path, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    ids = [r['id'] for r in rows]
    em_xy = np.array([[float(r['em_x_mm']), float(r['em_y_mm'])] for r in rows])
    map_xy = np.array([[float(r['map_x_mm']), float(r['map_y_mm'])] for r in rows])
    h = np.array([float(r['height_mm']) for r in rows])
    chip = np.array([float(r['signed_chip_dist_mm']) for r in rows])
    return ids, em_xy, map_xy, h, chip


def make_policy(method, ids, em_xy, map_xy, chip, seed):
    if method == 'topoprobe':
        return topobo.TopoBO(ids, em_xy, topobo.terrain_priority(chip), topobo.initial_positions(ids, em_xy, chip, seed))
    if method == 'plainbo':
        return topobo.TopoBO(ids, em_xy, np.zeros(len(ids)), topobo.uniform_initial_positions(len(ids), seed))
    if method == 'scniffer':
        return baselines.SCNIFFER(ids, map_xy, baselines.quadrant_starts(ids, map_xy, seed))
    if method == 'swarm':
        return baselines.Swarm(ids, map_xy, seed)
    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', choices=['topoprobe', 'plainbo', 'scniffer', 'swarm'], required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--setup', default='setup.json')
    ap.add_argument('--out', required=True)
    ap.add_argument('--gap', type=float, default=1.0)
    a = ap.parse_args()
    setup = json.loads(Path(a.setup).read_text(encoding='utf-8'))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    ids, em_xy, map_xy, h, chip = load_candidates(setup['candidates'])
    where = {c: i for i, c in enumerate(ids)}
    policy = make_policy(a.method, ids, em_xy, map_xy, chip, a.seed)
    stage = Stage(box=setup['motion_box'])
    ex = Executor(stage, setup['z_ref_mm'], setup['em_box'])
    entropy = baselines.EntropyScore() if a.method == 'swarm' else None
    corner_batches, history, log = [], [], (out / 'queries.jsonl').open('w', encoding='utf-8')
    rec, serial = {}, 0

    def measure(cand, n, label):
        nonlocal serial
        serial += 1
        i = where[cand]
        ex.go(em_xy[i], h[i] + setup.get('height_shift_mm', 0.), a.gap)
        folder = capture(out / 'captures' / f'{serial:03d}_{label}_{cand}', n, a.seed * 1000 + serial)
        ex.lift()
        return leakage.load_capture(folder)

    try:
        while len(history) < BUDGETS[-1]:
            if a.method in ('topoprobe', 'plainbo'):
                cand, purpose = policy.choose(history), 'search'
            else:
                step = policy.ask()
                if step is None:
                    break
                cand, purpose = step
            if purpose == 'background':
                corner_batches.append(measure(cand, 256, 'background')[0])
                if len(corner_batches) == 4:
                    entropy.fit_background(corner_batches)
                policy.tell(None)
                continue
            x, pt, key = measure(cand, 1024, 'search')
            s = entropy(x) if entropy else leakage.score(x, pt, key)
            if a.method not in ('topoprobe', 'plainbo'):
                policy.tell(s)
            history.append((cand, s))
            log.write(json.dumps(dict(query=len(history), candidate=cand, route=purpose, score=s, t=time.time())) + '\n')
            log.flush()
            if len(history) in BUDGETS:
                rec[len(history)] = max(history, key=lambda r: r[1])[0]
        for b in BUDGETS:                                     # a method that stopped early keeps its last best
            rec.setdefault(b, max(history, key=lambda r: r[1])[0])
        cpa = {}
        for cand in sorted(set(rec.values())):
            x, pt, key = measure(cand, setup.get('validation_traces', {}).get(a.method, 1024), 'validation')
            cpa[cand] = leakage.recovery_traces(x, pt, key)
    finally:
        log.close()
        try:
            ex.lift()
        finally:
            stage.close()
    result = dict(method=a.method, seed=a.seed, recommendations={b: rec[b] for b in BUDGETS},
                  recovery_traces={b: cpa[rec[b]] for b in BUDGETS})
    (out / 'result.json').write_text(json.dumps(result, indent=1), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
