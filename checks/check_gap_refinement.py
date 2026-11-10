"""Replay the height-study metric on recorded CPA decisions and verify published means."""
from pathlib import Path
from unittest.mock import patch
import json, sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from topoprobe import leakage


def main():
    data = json.loads((ROOT / 'data/gap_ablation_refined.json').read_text())
    legacy = json.loads((ROOT / 'data/gap_ablation.json').read_text())
    groups = {name: [] for name in data['recovery_traces']}
    censored = 0
    for batch in data['batches']:
        decisions = batch['evaluations']
        limit = batch['acquisition_limit']
        traces = np.empty((limit, 0))
        with patch.object(leakage, 'cpa_full_key', side_effect=lambda x, pt, key: decisions[str(len(x))]):
            endpoint = leakage.recovery_traces(traces, traces, None)
        assert endpoint == batch['exact_recovery_traces'], batch['site']
        if endpoint is None:
            assert not decisions[str(limit)]
            censored += 1
        else:
            assert decisions[str(endpoint)] and not decisions[str(endpoint - 1)]
        groups[batch['site'] + '_' + batch['mode']].append((batch['repeat'], endpoint))
    assert len(data['batches']) == 18 and censored == 1
    for name, values in groups.items():
        counts = [value for _, value in sorted(values)]
        mean = float(np.mean([data['acquisition_limit'] if value is None else value for value in counts]))
        assert counts == data['recovery_traces'][name] == legacy[name]['recovery_traces']
        assert mean == data['capped_means'][name] == legacy[name]['mean_recovery_traces_capped']
        print(f'{name:24s} {counts}  mean {mean:.1f}')
    reduction = 100 * (1 - data['capped_means']['B_low_board_follow'] / data['capped_means']['B_low_board_fixed'])
    print(f'Low-board reduction: {reduction:.1f}%')
    print('PASS: 18 recorded-decision metric replays, 6 means, 1 censored batch.')


if __name__ == '__main__':
    main()
