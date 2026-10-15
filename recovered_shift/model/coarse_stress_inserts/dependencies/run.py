"""One-way coarse-grid backgrounds for the six-field degree-one residual lift."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np,pandas as pd,json,time,argparse
from pathlib import Path
from scipy.signal import resample
from hybrid import Hybrid,collide,collision_jvp
from active import initial,velocity,vort
import models as m
P=Path(__file__).resolve().parent
NU64=(1/3)*(1/1.95-.5)
def rate(N):return 1-1/(.5+3*NU64*N/64)
def prolong(q,N):return resample(resample(q,N,axis=0),N,axis=1)
class Coarse:
 def __init__(self,N,Nc,steps):
  assert N%Nc==0;self.N=N;self.Nc=Nc;self.r=rate(Nc);self.ratio=N//Nc;self.q0=initial(Nc);self.q1=m.step6(self.q0,self.r);self.k=0;self.valid=True
  tf=1/(1-rate(N));tc=1/(1-self.r);self.factor=(tf-1)/(tc-1)/self.ratio
  self.ends={kind:[self.reconstruct(self.q0,kind),self.reconstruct(self.q1,kind)] for kind in ['raw','stress_scaled']}
 def reconstruct(self,q,kind):
  if kind=='raw':return prolong(q,self.N)
  hydro=prolong(q[...,:3],self.N);neq=q[...,3:]-m.eqkin(q[...,:3])[...,:3]
  return np.concatenate([hydro,m.eqkin(hydro)[...,:3]+self.factor*prolong(neq,self.N)],axis=-1)
 def at(self,n,kind):
  target=n//self.ratio
  if target>self.k:
   assert target==self.k+1;self.k=target;self.q0=self.q1;self.q1=m.step6(self.q0,self.r)
   if not np.isfinite(self.q1).all() or np.max(np.abs(self.q1))>1e8 or np.min(self.q1[...,0])<=0:raise RuntimeError(f'Coarse background invalid: N={self.Nc}, step={target}')
   for typ in self.ends:self.ends[typ]=[self.ends[typ][1],self.reconstruct(self.q1,typ)]
  alpha=(n%self.ratio)/self.ratio;aa,bb=self.ends[kind];return (1-alpha)*aa+alpha*bb
class Corrector:
 def __init__(self,b,a,r):self.b=b;self.eta=a-b;self.r=r;self.error_budget=np.zeros_like(a);self.defect=np.zeros_like(a);self.omission=np.zeros_like(a)
 def state(self):return self.b+self.eta
 def step(self,nb,ref,refnext):
  s=m.stream6(self.b);v=m.stream6(self.eta);fb=collide(s,self.r);self.defect=fb-nb
  # Offline reference remainder and forced error equation; never supplied to eta.
  vv=m.stream6(ref-self.b);rho=s[...,0];rx=vv[...,1]-s[...,1]*vv[...,0]/rho;ry=vv[...,2]-s[...,2]*vv[...,0]/rho
  R=np.zeros_like(ref);den=rho+vv[...,0];R[...,3]=(1-self.r)*rx*rx/(np.sqrt(2)*den);R[...,4]=(1-self.r)*rx*ry/den;R[...,5]=(1-self.r)*ry*ry/(np.sqrt(2)*den)
  self.omission=R;self.error_budget=collision_jvp(s,m.stream6(self.error_budget),self.r)-R
  self.eta=self.defect+collision_jvp(s,v,self.r);self.b=nb

def run(N,base_steps=1500):
 scale=N//64;steps=base_steps*scale;r=rate(N);a=initial(N);q=a.copy();q[...,0]-=1;norm=np.linalg.norm(q);unorm=np.linalg.norm(velocity(a));energy=unorm**2
 cores={Nc:Coarse(N,Nc,steps) for Nc in [N//4,N//2]};corr={}
 for Nc,c in cores.items():
  for kind in ['raw','stress_scaled']:corr[(Nc,kind)]=Corrector(c.at(0,kind),a,r)
 modes={K:Hybrid(a,K,r) for K in [2,4]};rows=[];snaps={};clock=time.time();checks=[]
 for n in range(steps+1):
  if n%(5*scale)==0:
   for (Nc,kind),h in corr.items():
    z=h.state();err=z-a
    rows.append(dict(N=N,step=n/scale,model=f'grid{Nc}_{kind}',kind=kind,Nc=Nc,K=-1,nonlinear_dof=6*Nc*Nc,core_fraction=(Nc/N)**2,background_error=np.linalg.norm(h.b-a)/norm,corrected_error=np.linalg.norm(err)/norm,velocity_error=np.linalg.norm(velocity(z)-velocity(a))/unorm,residual_norm=np.linalg.norm(h.eta)/norm,reference_residual_norm=np.linalg.norm(a-h.b)/norm,defect_norm=np.linalg.norm(h.defect)/norm,omitted_remainder=np.linalg.norm(h.omission)/norm,budget_mismatch=np.linalg.norm(err-h.error_budget)/norm,min_density=z[...,0].min(),background_min_density=h.b[...,0].min(),energy_fraction=np.sum(velocity(a)**2)/energy))
   for K,h in modes.items():
    z=h.state();rows.append(dict(N=N,step=n/scale,model=f'modes{K}',kind='modes',Nc=-1,K=K,nonlinear_dof=6*N*(2*K+1),core_fraction=(2*K+1)/N,background_error=np.linalg.norm(h.b-a)/norm,corrected_error=np.linalg.norm(z-a)/norm,velocity_error=np.linalg.norm(velocity(z)-velocity(a))/unorm,residual_norm=np.linalg.norm(h.eta)/norm,reference_residual_norm=np.linalg.norm(a-h.b)/norm,defect_norm=np.nan,omitted_remainder=np.linalg.norm(h.defect())/norm,budget_mismatch=np.nan,min_density=z[...,0].min(),background_min_density=h.b[...,0].min(),energy_fraction=np.sum(velocity(a)**2)/energy))
  if n//scale in [0,500,1000,1500] and n%scale==0:
   snaps[f'reference_{n//scale}']=a.copy()
   for (Nc,kind),h in corr.items():snaps[f'background_grid{Nc}_{kind}_{n//scale}']=h.b.copy();snaps[f'corrected_grid{Nc}_{kind}_{n//scale}']=h.state().copy()
   for K,h in modes.items():snaps[f'background_modes{K}_{n//scale}']=h.b.copy();snaps[f'corrected_modes{K}_{n//scale}']=h.state().copy()
  if n==steps:break
  an=m.step6(a,r)
  for (Nc,kind),h in corr.items():h.step(cores[Nc].at(n+1,kind),a,an)
  for h in modes.values():h.step()
  a=an
  if (n+1)%(250*scale)==0:print('N',N,'step',(n+1)/scale,'seconds',round(time.time()-clock,1),flush=True)
 d=pd.DataFrame(rows);d.to_csv(P/f'results_N{N}.csv',index=False);np.savez_compressed(P/f'snapshots_N{N}.npz',**snaps)
 (P/f'settings_N{N}.json').write_text(json.dumps(dict(N=N,steps=steps,base_steps=base_steps,r=r,nu=NU64*N/64,coarse=[dict(Nc=Nc,r=c.r,ratio=c.ratio,stress_factor=c.factor) for Nc,c in cores.items()],core_independent=True,degree=1,restarts=False,reference_used_only_for_scoring_and_offline_error_budget=True,seconds=time.time()-clock),indent=2))
 print(d.groupby('model')[['background_error','corrected_error','budget_mismatch']].max().to_string(),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--N',type=int,default=64);args=ap.parse_args();run(args.N)
