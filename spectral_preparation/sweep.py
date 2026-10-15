from pathlib import Path
import sys,time,json,argparse
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'double_shear_c2'))
from experiment import *
P=Path(__file__).resolve().parent

def bilinear(sc,x,y,rate):
 rho=sc[...,0];ux=sc[...,1]/rho;uy=sc[...,2]/rho
 xx=x[...,1]-ux*x[...,0];xy=x[...,2]-uy*x[...,0]
 yx=y[...,1]-ux*y[...,0];yy=y[...,2]-uy*y[...,0]
 z=np.zeros_like(x);fac=(1-rate)/rho
 z[...,3]=fac*xx*yx/np.sqrt(2);z[...,4]=fac*(xx*yy+xy*yx)/2;z[...,5]=fac*xy*yy/np.sqrt(2)
 return z

def run(N,Nc,ks):
 start=time.time();Re=100000;old.initial=initial;old.rate=lambda n:1-1/(.5+3*U*n/Re);rate=old.rate(N)
 core=old.Coarse(N,Nc,0);b=core.at(0,'raw');ref=initial(N);steps=round(3*N/U)
 k=np.fft.fftfreq(N)*N;rad=np.hypot(k[:,None],k[None,:]);mask=np.stack([rad<=K for K in ks],axis=2)
 proj=lambda a:np.fft.ifft2(np.fft.fft2(a,axes=(0,1))*mask[...,None],axes=(0,1)).real
 c=b[:,:,None,:]+proj(np.repeat((ref-b)[:,:,None,:],len(ks),axis=2));r1=ref[:,:,None,:]-c;rq=r1.copy()
 D=np.linalg.norm(ref-np.array([1,0,0,0,0,0]));rows=[];source=[];peaks=np.zeros((3,len(ks)));sum2=np.zeros_like(peaks);prev=np.zeros_like(peaks);minrho=np.ones(len(ks))*1e30
 norm=lambda a:np.sqrt(np.sum(a*a,axis=(0,1,3)))
 for n in range(steps+1):
  err=np.stack([100*norm(c-ref[:,:,None,:])/D,100*norm(c+r1-ref[:,:,None,:])/D,100*norm(c+rq-ref[:,:,None,:])/D])
  peaks=np.maximum(peaks,err)
  if n:sum2+=(prev**2+err**2)/2*U/N
  prev=err
  minrho=np.minimum(minrho,c[...,0].min(axis=(0,1)))
  if n%max(1,N//4)==0 or n==steps:
   rn=norm(rq)/D
   for i,K in enumerate(ks):rows.append(dict(N=N,Nc=Nc,K=K,time=n*U/N,background_error_percent=err[0,i],C1_error_percent=err[1,i],quadratic_error_percent=err[2,i],quadratic_residual_D0=rn[i]))
  if n==steps:break
  sc=m.stream6(c);fc=collide(sc,rate);bn=core.at(n+1,'raw');cn=bn[:,:,None,:]+proj(fc-bn[:,:,None,:]);d=fc-cn
  sq=m.stream6(rq);ar=J(sc,sq,rate);q=quadratic(sc,sq,rate)
  if n%max(1,N//4)==0:
   sn=m.stream6(cn);sa=m.stream6(ar);sqq=m.stream6(q)
   cubic=2*bilinear(sn,sa,sqq,rate);quartic=quadratic(sn,sqq,rate)
   cnorm=norm(cubic)/D;qnorm=norm(quartic)/D;dnorm=norm(d)/D
   for i,K in enumerate(ks):source.append(dict(N=N,Nc=Nc,K=K,time=n*U/N,cubic_return_D0=cnorm[i],quartic_return_D0=qnorm[i],defect_D0=dnorm[i]))
  rq=d+ar+q;r1=d+J(sc,m.stream6(r1),rate);c=cn;ref=m.step6(ref,rate)
  if (n+1)%400==0:print(f'N{N}/{Nc}: {n+1}/{steps}, elapsed {time.time()-start:.1f}s',flush=True)
 data=pd.DataFrame(rows);src=pd.DataFrame(source);data.to_csv(P/f'N{N}_Nc{Nc}_sweep.csv',index=False);src.to_csv(P/f'N{N}_Nc{Nc}_sources.csv',index=False)
 summary=[]
 for i,K in enumerate(ks):
  ss=src[src.K==K];summary.append(dict(N=N,Nc=Nc,K=K,mode_count=int(mask[:,:,i].sum()),nonlinear_dof=int(6*mask[:,:,i].sum()),nonlinear_fraction=float(mask[:,:,i].mean()),background_peak=peaks[0,i],background_final=err[0,i],C1_peak=peaks[1,i],C1_final=err[1,i],quadratic_peak=peaks[2,i],quadratic_final=err[2,i],C1_rms=np.sqrt(sum2[1,i]/3),cubic_return_rms_D0=np.sqrt(np.mean(ss.cubic_return_D0**2)),quartic_return_rms_D0=np.sqrt(np.mean(ss.quartic_return_D0**2)),min_background_density=minrho[i]))
 scores=pd.DataFrame(summary);scores.to_csv(P/f'N{N}_Nc{Nc}_scores.csv',index=False)
 (P/f'N{N}_Nc{Nc}_checks.json').write_text(json.dumps(dict(seconds=time.time()-start,steps=steps,ks=ks,finite=bool(np.isfinite(err).all())),indent=2))
 print(scores.to_string(index=False),flush=True)
 if N==32:
  selections=[]
  for tol in [5.,1.]:
   valid=scores[scores.C1_peak<=tol].sort_values('nonlinear_dof')
   K=int(valid.iloc[0].K) if len(valid) else None
   selections.append(dict(tolerance_percent=tol,coarse_K=K,same_physical_K=K,same_fraction_K=2*K if K is not None else None))
  (P/'Coarse_Selections.json').write_text(json.dumps(dict(protocol='Minimum nonlinear mode count among tested K passing peak C1 error tolerance; DSL pilot32/16 only. Both transfers are hypotheses.',selections=selections),indent=2));print(selections,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--pilot',action='store_true');a=ap.parse_args()
 if a.pilot:run(32,16,list(range(15)))
 else:run(64,32,[0,8,11,12,14,16,20,22,24,26,28,46])
