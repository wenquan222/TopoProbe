"""Check paired plaintext streams and the height-ablation execution loop without hardware."""
from pathlib import Path
import sys,types,tempfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from topoprobe import gap_ablation as gap

def plaintexts(seed,n=8192):
    rng=np.random.default_rng(seed)
    return np.vstack([np.column_stack([rng.permutation(256) for _ in range(16)])
                      for _ in range((n+255)//256)]).astype(np.uint8)[:n]

seen={}
for rep,order in gap.ORDERS.items():
    for site,mode in order:
        for batch in ['search','validation']:
            seed=gap.paired_seed(rep,site,batch)
            seen[(rep,site,mode,batch)]=seed
    for site in gap.SITE_INDEX:
        for batch in ['search','validation']:
            assert seen[rep,site,'fixed',batch]==seen[rep,site,'follow',batch]
            assert np.array_equal(plaintexts(seen[rep,site,'fixed',batch]),plaintexts(seen[rep,site,'follow',batch]))
streams={gap.paired_seed(rep,site,batch) for rep in gap.ORDERS for site in gap.SITE_INDEX for batch in ['search','validation']}
assert len(streams)==18
assert len({plaintexts(s,256).tobytes() for s in streams})==18

# Exercise main with a fake hardware module, so the check covers the seeds actually passed to capture.
captures=[]
class Stage:
    def __init__(self,**kw):pass
    def close(self):pass
class Executor:
    def __init__(self,*a,**kw):pass
    def go(self,*a,**kw):pass
    def lift(self):pass
def capture(folder,n,seed):
    captures.append((Path(folder).name,n,seed))
    return folder
fake=types.ModuleType('topoprobe.hardware')
fake.Stage,fake.Executor,fake.capture=Stage,Executor,capture
original=sys.modules.get('topoprobe.hardware')
argv=sys.argv[:]
load,first=gap.leakage.load_capture,gap.first_success
try:
    sys.modules['topoprobe.hardware']=fake
    gap.leakage.load_capture=lambda folder:(None,None,None)
    gap.first_success=lambda *a:256
    with tempfile.TemporaryDirectory() as temp:
        import json
        setup=Path(temp)/'setup.json'
        setup.write_text(json.dumps(dict(motion_box={},em_box={},z_ref_mm=0,
            gap_sites={site:dict(em_xy_mm=[0,0],height_mm=0) for site in gap.SITE_INDEX})))
        sys.argv=['gap_ablation','--setup',str(setup),'--out',str(Path(temp)/'out')]
        gap.main()
    assert len(captures)==36
    for rep,order in gap.ORDERS.items():
        for site in gap.SITE_INDEX:
            for batch,n in [('search',1024),('validation',8192)]:
                paired=[(count,seed) for name,count,seed in captures if name in
                        [f'rep{rep}_{site}_fixed_{batch}',f'rep{rep}_{site}_follow_{batch}']]
                assert paired==[(n,gap.paired_seed(rep,site,batch))]*2
finally:
    sys.argv=argv
    gap.leakage.load_capture,gap.first_success=load,first
    if original is None:sys.modules.pop('topoprobe.hardware',None)
    else:sys.modules['topoprobe.hardware']=original
print('PASS: 9 site/repetition pairs, 18 independent streams, 36 mocked captures, zero hardware access.')
