# TopoProbe evaluation data — 2026-11-10

This guide describes the refined recovery metric used by the current manuscript. Earlier guides and the historical checkpoint fields preserve the original discrete-checkpoint observations.

## Recovery metric

For each independent validation batch, find the first tested checkpoint that recovers all key bytes at that checkpoint and every later tested checkpoint. Bisect the bracket ending at that checkpoint until its width is one trace, then report the successful upper endpoint. The same rule applies to the search comparison, search ablation, and paired height study.

The implementation is recovery_traces in topoprobe/leakage.py. The height-study record includes CPA decisions at the evaluated trace counts and the successful/failed endpoints of each refined bracket.

## Height study

Three paired repetitions compare fixed-plane acquisition against surface following at the same sites. The nominal gap is 0.5 mm relative to the IC top for the fixed plane and relative to each site's surface for surface following.

| Site and mode | Repetition 1 | Repetition 2 | Repetition 3 | Mean |
|---|---:|---:|---:|---:|
| Capacitor A, fixed | >8192 | 7884 | 7534 | 7870.0 capped |
| Capacitor A, following | 1877 | 2729 | 2097 | 2234.3 |
| Low board, fixed | 3009 | 4193 | 3445 | 3549.0 |
| Low board, following | 206 | 293 | 235 | 244.7 |
| IC top, fixed | 1872 | 1352 | 3133 | 2119.0 |
| IC top, following | 1387 | 1774 | 2024 | 1728.3 |

The low-board mean decreases by 93.1%. The low-board and IC-top recovery-cost ordering reverses in all three pairs. The capacitor fixed-plane mean substitutes the 8192-trace acquisition limit for the censored repetition; its uncapped mean exceeds 7870.0.

The third fixed-plane IC-top batch succeeds at 2048 traces, fails at 3072, and has a refined endpoint of 3133 under the uniform rule above. These checkpoint decisions remain in the record.

- data/gap_ablation.json: historical first_success_traces, current recovery_traces, paired gaps, and capped means.
- data/gap_ablation_refined.json: metric definition, all 18 batches, recorded CPA decisions, refined endpoints, and provenance hashes.

Run:

    python checks/check_gap_refinement.py

The check replays the metric using recorded CPA outcomes and verifies all six group means. The source refinement verified 147456 waveform hashes across the 18 batches. The exported data preserve the source manifest and analysis hashes.

## Search comparison

Each of five runs uses a 64-query budget with 1024 traces per query. Initialization and repeated visits count toward this budget. Swarm separately acquires four 256-trace corner-background batches. Independent recovery counts measure the quality of recommended sites, while query counts measure search cost.

| Method | 32 queries | 48 queries | 64 queries |
|---|---:|---:|---:|
| SCNIFFER | 363.4 | 363.4 | 363.4 |
| Swarm | 1180.2 | 1841.2 | 1868.8 |
| Without topographic guidance | 105.2 | 102.8 | 97.0 |
| TopoProbe | 127.6 | 44.8 | 35.6 |

checks/replay_all.py reproduces query selection and recommendations. Read per-run values in numeric run order from data/runs/.

## Preparation and reuse

The recorded high-resolution surface scan contains 3701 accepted rows at a 0.01 mm row pitch. Two active scan intervals total 11.4 h; the elapsed span is 11.8 h including a 25.0-minute pause. Registration fitting takes 4.1 s after reference measurements. data/preparation_cost.json records these components and their provenance. Reference-point acquisition and operator checks remain separate preparation activities.

At 44–50 seconds per query, 64 search queries require approximately 46.9–53.3 minutes. Acquisition at all 1748 candidates would require an estimated 21.4–24.3 hours at those same rates. Stable mounting and probe geometry allow reuse of the registered map across searches.
