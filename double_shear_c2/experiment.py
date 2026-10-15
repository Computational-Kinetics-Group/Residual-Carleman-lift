from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"carleman_c2"))
from engine import *
from active import initial
P=Path(__file__).resolve().parent
U=.08
import time,json,argparse,pandas as pd
def velocity(a):return a[...,1:3]/(np.sqrt(3)*a[...,0,None])
def vorticity(a):
 v=velocity(a);N=len(a);k=2*np.pi*np.fft.fftfreq(N)*N;h=np.fft.fft2(v,axes=(0,1));return np.fft.ifft2(1j*k[:,None]*h[...,1]-1j*k[None,:]*h[...,0]).real
def run(N,Nc,Re,K=12,threads=4):
 start=time.time();old.initial=initial;old.rate=lambda n:1-1/(.5+3*U*n/Re);rate=old.rate(N)
 core=old.Coarse(N,Nc,0);b=core.at(0,'raw');ref=initial(N);steps=round(3*N/U)
 freq=np.fft.fftfreq(N)*N;mask=np.hypot(freq[:,None],freq[None,:])<=K
 def proj(a):return np.fft.ifft2(np.fft.fft2(a,axes=(0,1))*mask[...,None],axes=(0,1)).real
 c=b+proj(ref-b);r0=ref-c;lift=Lift(r0,steps,threads);r1=r0.copy();rq=r0.copy()
 D=np.linalg.norm(ref-np.array([1,0,0,0,0,0]));rows=[];diag=[];failed=None;history=[];snap={};morph=[]
 for n in range(steps+1):
  er2=100*np.linalg.norm(c+lift.r-ref)/D;errq=100*np.linalg.norm(c+rq-ref)/D;er1=100*np.linalg.norm(c+r1-ref)/D
  if n%max(1,N//4)==0 or n==steps or not np.isfinite(er2) or er2>1e6:
   for name,r,err in [('C1',r1,er1),('C2',lift.r,er2),('quadratic_direct',rq,errq)]:
    rows.append(dict(N=N,Nc=Nc,Re=Re,K=K,time=n*U/N,model=name,error_percent=err,error_to_quadratic_percent=100*np.linalg.norm(r-rq)/D,min_density=float((c+r)[...,0].min())))
  history.append(dict(time=n*U/N,C1=er1,C2=er2,quadratic_direct=errq))
  if n%round(.5*N/U)==0 or n==steps:
   for name,a in [('reference',ref),('C1',c+r1),('C2',c+lift.r),('quadratic',c+rq),('center',c)]:snap[f'{name}_{n}']=a.copy()
  if n%round(.05*N/U)==0 or n==steps:
   vr=vorticity(ref);ur=velocity(ref)
   for name,a in [('C1',c+r1),('C2',c+lift.r),('quadratic',c+rq)]:
    vv=vorticity(a);uv=velocity(a)
    morph.append(dict(time=n*U/N,model=name,velocity_current_percent=100*np.linalg.norm(uv-ur)/np.linalg.norm(ur),vorticity_current_percent=100*np.linalg.norm(vv-vr)/np.linalg.norm(vr),vorticity_correlation=np.sum(vv*vr)/np.linalg.norm(vv)/np.linalg.norm(vr),enstrophy_ratio=np.sum(vv*vv)/np.sum(vr*vr)))

  if n and n%200==0:
   print(f'N{N}/{Nc} Re{Re:g} step{n}/{steps} t={n*U/N:.3f} C1={er1:.5g}% C2={er2:.5g}% Q={errq:.5g}% elapsed={time.time()-start:.1f}s',flush=True)
   pd.DataFrame(history).to_csv(P/f'partial_N{N}_Nc{Nc}_Re{Re:g}.csv',index=False)
  if not np.isfinite(er2) or er2>1e6:
   failed=dict(step=n,time=n*U/N,reason='nonfinite or error above 1e6 percent',C2_error_percent=er2);break
  if n==steps:break
  sc=m.stream6(c);fc=collide(sc,rate);bn=core.at(n+1,'raw');cn=bn+proj(fc-bn);d=fc-cn
  qrankone=quadratic(sc,m.stream6(lift.r),rate)
  qlift=lift.step(sc,d,rate)
  if n%max(1,N//4)==0:
   diag.append(dict(time=n*U/N,lifted_Q_D0=np.linalg.norm(qlift)/D,rankone_Q_D0=np.linalg.norm(qrankone)/D,closure_force_defect_D0=np.linalg.norm(qlift-qrankone)/D,forcing_D0=np.linalg.norm(d)/D,factor_columns=lift.active))
  srq=m.stream6(rq);rq=d+J(sc,srq,rate)+quadratic(sc,srq,rate);r1=d+J(sc,m.stream6(r1),rate)
  ref=m.step6(ref,rate);c=cn;b=bn
 tag=f'N{N}_Nc{Nc}_Re{Re:g}_K{K}'
 pd.DataFrame(rows).to_csv(P/f'{tag}_flow.csv',index=False);pd.DataFrame(history).to_csv(P/f'{tag}_allsteps.csv',index=False);pd.DataFrame(diag).to_csv(P/f'{tag}_diagnostics.csv',index=False)
 pd.DataFrame(morph).to_csv(P/f'{tag}_morphology.csv',index=False)
 np.savez_compressed(P/f'{tag}_snapshots.npz',**snap)
 np.savez_compressed(P/f'{tag}_final.npz',center=c,reference=ref,C1=c+r1,C2=c+lift.r,quadratic=c+rq)
 info=dict(N=N,Nc=Nc,Re=Re,K=K,steps_completed=n,time=n*U/N,failed=failed,seconds=time.time()-start,factor_columns=lift.active,factor_buffer_bytes=lift.f.nbytes+lift.work.nbytes,population_workspace_bytes=lift.pop.nbytes if lift.fast else 0)
 (P/f'{tag}_checks.json').write_text(json.dumps(info,indent=2));print(info,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--N',type=int,default=32);ap.add_argument('--Nc',type=int,default=16);ap.add_argument('--Re',type=float,required=True);ap.add_argument('--threads',type=int,default=4);args=ap.parse_args();run(args.N,args.Nc,args.Re,threads=args.threads)
