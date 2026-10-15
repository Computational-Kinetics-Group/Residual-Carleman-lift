"""Compare newly computed main runs with the immutable archived evidence."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent
checks={}
def csvcheck(label,new,old,rtol=1e-7,atol=1e-9):
 a=pd.read_csv(new);b=pd.read_csv(old)
 assert a.shape==b.shape,(label,a.shape,b.shape)
 cols=list(a.select_dtypes(include='number').columns)
 assert list(a.columns)==list(b.columns),label
 x=a[cols].to_numpy();y=b[cols].to_numpy()
 np.testing.assert_allclose(x,y,rtol=rtol,atol=atol,equal_nan=True,err_msg=label)
 checks[label]={'max_abs_difference':float(np.nanmax(abs(x-y))),'rtol':rtol,'atol':atol}
csvcheck('pilot',P/'outputs/pilot/N32_Nc16_scores.csv',P/'spectral_preparation/N32_Nc16_scores.csv')
tag='N64_Nc32_Re100000_K28'
csvcheck('RC2_history',P/f'outputs/rc2/{tag}_allsteps.csv',P/f'spectral_preparation/C2_K28/{tag}_allsteps.csv')
a=np.load(P/f'outputs/rc2/{tag}_snapshots.npz');b=np.load(P/f'spectral_preparation/C2_K28/{tag}_snapshots.npz')
assert set(a.files)==set(b.files)
mx=0
for k in a.files:
 np.testing.assert_allclose(a[k],b[k],rtol=1e-7,atol=1e-10,err_msg=k)
 mx=max(mx,float(np.max(abs(a[k]-b[k]))))
checks['snapshots']={'max_abs_difference':mx,'rtol':1e-7,'atol':1e-10}
csvcheck('conventional_baseline',P/'baseline/baseline_allsteps.csv',P/'data/baseline_allsteps.csv')
(P/'validation/reproduction.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
