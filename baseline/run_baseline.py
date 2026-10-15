"""Conventional fixed-rest C1/C2, same regularized D2Q9 shear-layer reference.

The pair tensor has the exact factor Z_n=u_n u_n^T, u_{n+1}=A_*u_n,
because the fixed equilibrium has zero forcing. No product replacement from
the C2 first sector is performed. Nonlinear Q is a separate comparator.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,json,time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent.parent
import baseline_engine as m
from baseline_engine import initial,collision_jvp

def run(N=64,Re=100000.,U=.08):
    p=Path(__file__).resolve().parent;t0=time.time();steps=round(3*N/U)
    rate=1-1/(.5+3*U*N/Re)
    rest=np.zeros((N,N,6));rest[...,0]=1
    sr=m.stream6(rest)
    A=lambda r:collision_jvp(sr,m.stream6(r),rate)
    def B(r):
        v=m.stream6(r);rho=sr[...,0]
        x=v[...,1]-sr[...,1]*v[...,0]/rho
        y=v[...,2]-sr[...,2]*v[...,0]/rho
        q=np.zeros_like(v);fac=(1-rate)/rho
        q[...,3]=fac*x*x/np.sqrt(2);q[...,4]=fac*x*y;q[...,5]=fac*y*y/np.sqrt(2)
        return q
    ref=initial(N);x0=ref-rest;D=np.linalg.norm(x0)
    u=x0.copy();x=x0.copy();rq=x0.copy();rows=[];snaps={};q_failed=None
    refpath=ROOT/'data/rollup_K28.npz'
    if not refpath.exists():refpath=ROOT/'spectral_preparation/C2_K28/N64_Nc32_Re100000_K28_snapshots.npz'
    archived=np.load(refpath)
    maxref=0.
    for n in range(steps+1):
        rows.append(dict(time=n*U/N,C1=100*np.linalg.norm(rest+u-ref)/D,
            C2=100*np.linalg.norm(rest+x-ref)/D,
            quadratic_direct=100*np.linalg.norm(rest+rq-ref)/D,
            C2_min_density=float((rest+x)[...,0].min())))
        if n%400==0:
            maxref=max(maxref,float(np.max(abs(ref-archived[f'reference_{n}']))))
            for name,a in [('reference',ref),('C1',rest+u),('C2',rest+x),('quadratic',rest+rq)]:snaps[f'{name}_{n}']=a.copy()
            print(n,rows[-1],flush=True)
        if n==steps:break
        # q is contracted from the independently propagated pair factor u,
        # never from x (the C2 first sector).
        xn=A(x)+B(u);un=A(u)
        if q_failed is None:
            qn=A(rq)+B(rq)
            if not np.isfinite(qn).all() or np.max(abs(qn))>1e8:
                q_failed=n+1;rq=np.full_like(rq,np.nan)
            else:rq=qn
        x=xn;u=un;ref=m.step6(ref,rate)
    d=pd.DataFrame(rows);d.to_csv(p/'baseline_allsteps.csv',index=False)
    np.savez_compressed(p/'baseline_snapshots.npz',**snaps)
    results=dict(N=N,Re=Re,U=U,steps=steps,final_time=steps*U/N,
        fixed_background='uniform unit-density rest equilibrium',pair_factorization='exact rank one; u=A_*^n x0',
        forcing=0,restarts=False,reference_snapshot_max_abs_difference=maxref,
        rest_fixed_point_max_abs=float(np.max(abs(m.step6(rest,rate)-rest))),
        final_errors_percent={k:float(d[k].iloc[-1]) for k in ['C1','C2','quadratic_direct']},
        peak_errors_percent={k:float(d[k].max()) for k in ['C1','C2','quadratic_direct']},
        first_C2_crossings={str(tol):float(d.loc[d.C2>tol,'time'].iloc[0]) if (d.C2>tol).any() else None for tol in [.1,1,5,10]},
        quadratic_failed_step=q_failed,seconds=time.time()-t0)
    (p/'baseline_checks.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2),flush=True)
if __name__=='__main__':run()
