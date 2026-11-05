"""Gap ablation of Section II-A: fixed probe plane versus surface following at the same three sites.

For each site and each of three repetitions, both modes are measured in a predetermined order:
  fixed    probe Z set by the chip-top height + gap (one plane for all sites)
  follow   probe Z set by the site's freshly measured height + gap
Gap 0.5 mm. Each condition acquires 1024 search traces and 8192 independent validation traces; the
table reports the first checkpoint at which all 16 key bytes rank first (checkpoints 256 ... 8192).
Sites (stage XY of the EM probe): A capacitor (94.553, 26.19), B low board (94.538, 27.344),
C chip top (96.378, 42.634).

    python -m topoprobe.gap_ablation --setup setup.json --out runs/gap
"""
import argparse, json
from pathlib import Path

from . import leakage
from .hardware import Executor, Stage, capture

CHECKPOINTS = [256, 512, 1024, 1536, 2048, 3072, 4096, 6144, 8192]
ORDERS = {1: [('A_capacitor', 'fixed'), ('B_low_board', 'follow'), ('C_chip_top', 'fixed'), ('A_capacitor', 'follow'),
              ('B_low_board', 'fixed'), ('C_chip_top', 'follow')],
          2: [('C_chip_top', 'follow'), ('B_low_board', 'fixed'), ('A_capacitor', 'follow'), ('C_chip_top', 'fixed'),
              ('B_low_board', 'follow'), ('A_capacitor', 'fixed')],
          3: [('B_low_board', 'fixed'), ('C_chip_top', 'follow'), ('A_capacitor', 'fixed'), ('B_low_board', 'follow'),
              ('C_chip_top', 'fixed'), ('A_capacitor', 'follow')]}


def first_success(x, pt, key):
    for n in CHECKPOINTS:
        if n <= len(x) and leakage.cpa_full_key(x[:n], pt[:n], key):
            return n
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--setup', default='setup.json')
    ap.add_argument('--out', required=True)
    ap.add_argument('--gap', type=float, default=.5)
    a = ap.parse_args()
    setup = json.loads(Path(a.setup).read_text(encoding='utf-8'))
    sites = setup['gap_sites']            # {name: {"em_xy_mm": [x, y], "height_mm": h}}
    chip_h = sites['C_chip_top']['height_mm']
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    stage = Stage(box=setup['motion_box'])
    ex = Executor(stage, setup['z_ref_mm'], setup['em_box'])
    table = {}
    try:
        for rep, order in ORDERS.items():
            for k, (site, mode) in enumerate(order):
                s = sites[site]
                h = s['height_mm'] if mode == 'follow' else chip_h       # fixed plane: chip top + gap
                ex.go(s['em_xy_mm'], h, a.gap)
                capture(out / f'rep{rep}_{site}_{mode}_search', 1024, 1000000 * rep + 1000 * k)
                folder = capture(out / f'rep{rep}_{site}_{mode}_validation', 8192, 1000000 * rep + 1000 * k + 500)
                ex.lift()
                x, pt, key = leakage.load_capture(folder)
                table.setdefault(site, {}).setdefault(mode, []).append(first_success(x, pt, key))
                (out / 'table.json').write_text(json.dumps(table, indent=1), encoding='utf-8')
    finally:
        try:
            ex.lift()
        finally:
            stage.close()
    print(json.dumps(table))


if __name__ == '__main__':
    main()
