"""Reproduce the paper's search results from the files in this repository (no hardware, no waveforms).

For every one of the 20 runs in data/runs/, the search code in topoprobe/ is replayed with the run's
seed and recorded scores; each query and each 32/48/64 recommendation must equal the record. The
five-run means of Tables II and IV are then recomputed from the recorded recovery counts.

    python checks/replay_all.py
"""
import csv, json, sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from topoprobe import baselines, terrain, topobo  # noqa: E402

DATA = ROOT / 'data'


def load_candidates():
    with (DATA / 'candidates.csv').open(encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    ids = [r['id'] for r in rows]
    em = np.array([[float(r['em_x_mm']), float(r['em_y_mm'])] for r in rows])
    mp = np.array([[float(r['map_x_mm']), float(r['map_y_mm'])] for r in rows])
    classes = terrain.classes_from_map(DATA / 'height_map.npz', mp)
    return ids, em, mp, classes


def replay(run, ids, em, mp, classes):
    q = run['queries']
    seed = run['seed']
    got = []
    if run['method'] in ('topoprobe', 'terrain_free_bo'):
        if run['method'] == 'topoprobe':
            p = topobo.TopoBO(ids, em, topobo.terrain_priority(classes), topobo.initial_positions(ids, em, classes, seed),
                              classes=classes)
        else:
            p = topobo.TopoBO(ids, em, np.zeros(len(ids)), topobo.uniform_initial_positions(len(ids), seed))
        hist = [(r['candidate'], r['score']) for r in q]
        got = [p.choose(hist[:k]) for k in range(len(q))]
    else:
        p = (baselines.SCNIFFER(ids, mp, baselines.quadrant_starts(ids, mp, seed)) if run['method'] == 'scniffer'
             else baselines.Swarm(ids, mp, seed))
        k = 0
        while k < len(q):
            step = p.ask()
            if step is None:
                break
            cand, purpose = step
            if purpose == 'background':
                p.tell(None)
                continue
            got.append(cand)
            p.tell(q[k]['score'])
            k += 1
    queries_ok = got == [r['candidate'] for r in q]
    rec = {b: max(q[:int(b)], key=lambda r: r['score'])['candidate'] for b in ('32', '48', '64')}
    return queries_ok, rec == run['recommendations']


def main():
    ids, em, mp, classes = load_candidates()
    runs = sorted((DATA / 'runs').glob('*_run*.json'))
    ok_all, means = True, {}
    with threadpool_limits(limits=1):
        for f in runs:
            run = json.loads(f.read_text())
            qok, rok = replay(run, ids, em, mp, classes)
            ok_all &= qok and rok
            print(f'{f.stem:24s} queries {"ok" if qok else "MISMATCH"}  recommendations {"ok" if rok else "MISMATCH"}', flush=True)
            means.setdefault(run['method'], []).append(run['recovery_traces'])
    print('\nmean traces to full-key recovery (5 runs)')
    for m in ('scniffer', 'swarm', 'terrain_free_bo', 'topoprobe'):
        print(f'  {m:16s}', '  '.join(f'{b}q {np.mean([r[b] for r in means[m]]):7.1f}' for b in ('32', '48', '64')))
    print('\nALL PASS' if ok_all and len(runs) == 20 else '\nFAIL')


if __name__ == '__main__':
    main()
