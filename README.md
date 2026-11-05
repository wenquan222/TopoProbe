# TopoProbe

Code and data for **TopoProbe: Topography-Aware Probe Positioning for Electromagnetic Side-Channel Analysis on Circuit Boards**.

TopoProbe scans the surface of an assembled board with a laser displacement sensor, places an EM probe
at a prescribed local gap above every candidate position, and searches for measurement sites with a
terrain-guided Bayesian optimization (TopoBO).

## Results

STM32F429 software AES, 1748 candidate positions on a 1 mm grid, all searches at a 1 mm local gap with
the same map and executor, five runs per method. Mean number of traces to full-key recovery at the
recommended position (lower is better):

| Method | 32 queries | 48 queries | 64 queries |
|---|---:|---:|---:|
| SCNIFFER | 363.4 | 363.4 | 363.4 |
| Swarm in EM Hay | 1180.2 | 1841.2 | 1868.8 |
| Terrain-free BO (ablation) | 105.2 | 102.8 | 97.0 |
| **TopoProbe** | 172.6 | **80.2** | **50.6** |

Gap ablation (fixed probe plane vs. surface following, 0.5 mm gap, first checkpoint with all 16 key
bytes ranked first, three repetitions):

| Site | Fixed plane | Surface following |
|---|---|---|
| Capacitor A | >8192, 8192, 8192 | 2048, 3072, 3072 |
| Low-board site | 3072, 6144, 4096 | 256, 512, 256 |
| Chip top | 2048, 1536, 2048 | 1536, 2048, 2048 |

## Reproduce the search results

```
pip install -r requirements.txt
python checks/replay_all.py
```

This replays all 20 runs in `data/runs/` with the code in `topoprobe/`, the run seeds and the recorded
scores, checks that every query and every recommendation equals the record, and prints the table above.
It needs no hardware and no waveforms.

## Layout

```
topoprobe/
  scan.py          on-the-fly laser surface scan (one constant-speed X line per Y row)
  surface.py       height map from raw laser frames, local-median field, registration to the stage (Eq. 1),
                   1 mm candidate grid
  topobo.py        TopoBO: terrain groups and priority, initialization, Gibbs-kernel GP, noisy EI,
                   coverage-weighted selection (Eqs. 2-4, 6-10); terrain-free ablation = priority 0, uniform start
  baselines.py     SCNIFFER and Swarm in EM Hay on the same grid
  leakage.py       SNR search score (Eq. 5), first-round CPA, traces to full-key recovery, capture loader
  hardware.py      stage, laser, AES target with PicoScope capture, terrain-following executor
  run.py           one search run on the bench: python -m topoprobe.run --method topoprobe --seed ... --out ...
  gap_ablation.py  fixed-plane vs. surface-following measurement at three sites
checks/
  replay_all.py    reproduces the search results from data/
data/
  runs/            20 run records: every query (position, score), recommendations at 32/48/64, recovery traces
  summary.json     per-run recovery traces and five-run means
  candidates.csv   1748 candidates: map XY, probe stage XY, registered height, signed distance to the chip plateau
  height_map.npz   registered positioning field (local-median height map, mm)
  setup.json       registration parameters, probe offset, Z constant, motion limits
  gap_ablation.json  table of the gap ablation
```

Run records are named `<method>_run<k>.json`. The TopoProbe and Swarm runs with the same k ran in the same
session, as did SCNIFFER run 4. The SCNIFFER score and the TopoProbe score are the same (median over the 16
key bytes of the peak Hamming-weight SNR); Swarm uses its own entropy contrast score.

Recovery traces come from one fresh validation batch at each recommended position: 8192 traces for the
TopoProbe and Swarm recommendations of runs 1-3, 4096 for the other Swarm runs, and 1024 otherwise
(`validation_traces` in each record).

Candidate heights in the run records include a common session correction of +0.00308 mm relative to
`candidates.csv`, measured at the reference sites before the runs.

## Not included

The raw data are too large for this repository: raw laser frames of the surface scan (about 1.9 GB), the
reconstructed height grid before smoothing (55 MB) and the raw EM waveforms (about 14 GB). The positioning
field derived from the scan (`data/height_map.npz`) and the per-query scores are included.

## Hardware

IAI TT three-axis stage, Panasonic HL-G1 laser displacement sensor, Langer EM probe with BT706, PicoScope
3403D, STM32F429 running software AES-128 (16 plaintext + 16 key bytes over UART at 115200 baud; PC6 high
during encryption as the scope trigger). Serial port names in `topoprobe/hardware.py` are those of the
measurement PC. The motion code moves the probe close to a populated board and keeps the checks used in
the experiments; keep them when adapting it.

## License

MIT, see `LICENSE`.
