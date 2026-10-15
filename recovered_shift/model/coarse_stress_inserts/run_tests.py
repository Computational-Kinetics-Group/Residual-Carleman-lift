import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'dependencies'))
import models as m
from active import initial
from hybrid import collide,collision_jvp as J
from diagnose import rem
import run as old

def run(N,Nc,Re):
 old.rate=lambda n:1-1/(.5+3*.08*n/Re);r=old.rate(N);rc=old.rate(Nc);steps=round(3*N/.08);ratio=N//Nc;gamma=(1-(Nc/N)**2)/3
 # Exact nonlinear discrepancy and its equilibrium-stress remainder on Nc.
 B=initial(Nc);aux=old.Coarse(Nc,Nc//2,0);Ts=[]
 for k in range(steps//ratio+2):
  aa=aux.at(k,'raw');Ts.append(rem(m.stream6(aa),m.stream6(B-aa),rc)/(1-rc));B=m.step6(B,rc)
 c=old.Coarse(N,Nc,0);c2=old.Coarse(N,Nc//2,0);b=c.at(0,'raw');b2=c2.at(0,'raw');f=initial(N);D=np.linalg.norm(f-np.array([1,0,0,0,0,0]));names=['baseline','stress_raw','stress_scaled','shifted','old_source'];etas={name:f-b for name in names};failed={};rows=[];diag=[];cache={}
 for n in range(steps+1):
  if n%max(1,N//8)==0:
   for name,v in etas.items():rows.append(dict(N=N,Nc=Nc,Re=Re,time=n*.08/N,model=name,error=np.linalg.norm(b+v-f)/D,min_density=np.min((b+v)[...,0])))
  if n==steps:break
  bn=c.at(n+1,'raw');bn2=c2.at(n+1,'raw');s=m.stream6(b);h=gamma*m.stream6(b-b2);d=collide(s,r)-bn;k=n//ratio;frac=(n%ratio)/ratio
  for kk in [k,k+1]:
   if kk not in cache:cache[kk]=old.prolong(Ts[kk],N)
  if k-1 in cache:del cache[k-1]
  raw=(1-r)*((1-frac)*cache[k]+frac*cache[k+1])*(N!=Nc);scaled=gamma**2*raw;oldsrc=rem(s,h,r);pairedsrc=oldsrc-J(s+h,h,r)+J(s,h,r)
  target=rem(s,m.stream6(f-b),r)
  if n%max(1,N//8)==0:
   for name,src in [('stress_raw',raw),('stress_scaled',scaled),('old_source',oldsrc)]:
    xx=np.sum(src*src);yy=np.sum(target*target);dot=np.sum(src*target);Txx=np.sqrt(2)*src[...,3];Tyy=np.sqrt(2)*src[...,5];Txy=src[...,4];mineig=.5*(Txx+Tyy-np.sqrt((Txx-Tyy)**2+4*Txy*Txy))
    diag.append(dict(N=N,Nc=Nc,Re=Re,time=n*.08/N,model=name,target_power=yy,prediction_power=xx,cross=dot,error_power=np.sum((src-target)**2),negative_eigenvalue_fraction=np.mean(mineig < -1e-12),minimum_eigenvalue=mineig.min(),conserved_max=np.max(abs(src[...,:3]))))
  for name,v in list(etas.items()):
   if name in failed:continue
   sv=m.stream6(v)
   vn=d+J(s,sv,r)
   if name=='stress_raw':vn+=raw
   elif name=='stress_scaled':vn+=scaled
   elif name=='old_source':vn+=oldsrc
   elif name=='shifted':vn=d+J(s+h,sv,r)+pairedsrc
   if not np.isfinite(vn).all() or (bn+vn)[...,0].min()<=0 or np.max(abs(vn))>1e6:
    failed[name]=dict(step=n+1,time=(n+1)*.08/N,min_density=float((bn+vn)[...,0].min()));etas[name]=np.full_like(vn,np.nan)
   else:etas[name]=vn
  f=m.step6(f,r);b=bn;b2=bn2
 tag=f'N{N}_Nc{Nc}_Re{Re:g}';pd.DataFrame(rows).to_csv(P/(tag+'.csv'),index=False);pd.DataFrame(diag).to_csv(P/(tag+'_source.csv'),index=False);(P/(tag+'.json')).write_text(json.dumps(failed,indent=2));print(tag,failed,flush=True)
if __name__=='__main__':
 for N,Nc in [(32,16),(64,16),(64,32),(16,16)]:
  for Re in [1198.08,100000]:run(N,Nc,Re)
