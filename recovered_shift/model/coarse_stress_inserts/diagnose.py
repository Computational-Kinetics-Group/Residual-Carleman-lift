import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'dependencies'))
import models as m
from hybrid import collide,collision_jvp
import run as old

def rem(s,v,r):
 q=v[...,1:3]-s[...,1:3]*v[...,0,None]/s[...,0,None];den=s[...,0]+v[...,0];a=np.zeros_like(v)
 a[...,3]=(1-r)*q[...,0]**2/(np.sqrt(2)*den);a[...,4]=(1-r)*q[...,0]*q[...,1]/den;a[...,5]=(1-r)*q[...,1]**2/(np.sqrt(2)*den)
 return a

def models(s,v,h,r):
 rh=rem(s,h,r);lhv=collision_jvp(s+h,v,r)-collision_jvp(s,v,r);lhh=collision_jvp(s+h,h,r)-collision_jvp(s,h,r)
 return {'forcing':rh,'operator':lhv,'paired':lhv+rh-lhh}

def run():
 rows=[];tensor=[];gammas=[-2,-1,-.5,-.25,0,.125,.25,.5,1,2,4];scores={typ:{g:0. for g in gammas} for typ in ['forcing','operator','paired']};denom=0.
 for Re in [1198.08,100000]:
  old.rate=lambda n:1-1/(.5+3*.08*n/Re)
  for N in [32,64]:
   z=np.load(P.parent/'fixed_background_refinement'/f'N{N}_Nc16_Re{Re:g}_snapshots.npz');c8=old.Coarse(N,8,0);r=old.rate(N)
   times={round(t*N/.08):t for t in [0,.5,1,1.5,2]}
   for n in range(round(2*N/.08)+1):
    b8=c8.at(n,'raw')
    if n not in times:continue
    t=times[n];b=z[f'background_{n}'];a=z[f'fine_{n}'];s=m.stream6(b);v=m.stream6(a-b)
    # Richardson-shaped extrapolation, calibrated rather than assumed valid.
    hbase=m.stream6((1-(16/N)**2)/3*(b-b8));target=rem(s,v,r);tt=np.sum(target**2)
    if N==32 and Re==1198.08 and t<=1:denom+=tt
    for g in gammas:
     h=g*hbase
     if (s[...,0]+h[...,0]).min()<=0:continue
     for typ,pred in models(s,v,h,r).items():
      err=np.sum((pred-target)**2);pp=np.sum(pred**2);dot=np.sum(pred*target)
      rows.append(dict(N=N,Re=Re,time=t,gamma=g,kind=typ,target_power=tt,error_power=err,prediction_power=pp,cosine=dot/np.sqrt(tt*pp) if tt*pp>1e-50 else np.nan))
      if N==32 and Re==1198.08 and t<=1:scores[typ][g]+=err
    q=v[...,1:3]-s[...,1:3]*v[...,0,None]/s[...,0,None];pr=hbase[...,1:3]-s[...,1:3]*hbase[...,0,None]/s[...,0,None]
    tensor.append(dict(N=N,Re=Re,time=t,q_alignment=np.sum(q*pr)/np.sqrt(np.sum(q*q)*np.sum(pr*pr)) if np.sum(q*q)*np.sum(pr*pr)>1e-50 else np.nan,target_norm=np.sqrt(tt),proxy_norm=np.linalg.norm(rem(s,hbase,r))))
 best={typ:min(vals,key=vals.get) for typ,vals in scores.items()}
 out=dict(training='N32/Nc16 Re1198.08 at t=0,.5,1 only; actual residual used only for calibration',best_gamma=best,scores={typ:{str(g):v/denom for g,v in vals.items()} for typ,vals in scores.items()})
 (P/'calibration.json').write_text(json.dumps(out,indent=2));pd.DataFrame(rows).to_csv(P/'source_diagnostics.csv',index=False);pd.DataFrame(tensor).to_csv(P/'proxy_diagnostics.csv',index=False)
 print(json.dumps(out,indent=2))
if __name__=='__main__':run()
