import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np,json
from pathlib import Path
import baseline_engine as m
N=4;D=6*N*N;rate=1-1/(.5+3*.08*N/100000)
rest=np.zeros((N,N,6));rest[...,0]=1;sr=m.stream6(rest)
eye=np.eye(D)
S=np.column_stack([m.stream6(eye[:,j].reshape(N,N,6)).ravel() for j in range(D)])
A=np.column_stack([m.collision_jvp(sr,S[:,j].reshape(N,N,6),rate).ravel() for j in range(D)])
rho=sr[...,0].ravel();sx=sr[...,1].ravel()/rho;sy=sr[...,2].ravel()/rho
Lx=S[1::6]-sx[:,None]*S[0::6];Ly=S[2::6]-sy[:,None]*S[0::6]
def denseB(Z):
 q=np.zeros((N*N,6));fac=(1-rate)/rho
 q[:,3]=fac*np.einsum('ij,ij->i',Lx@Z,Lx)/np.sqrt(2)
 q[:,4]=fac*np.einsum('ij,ij->i',Lx@Z,Ly)
 q[:,5]=fac*np.einsum('ij,ij->i',Ly@Z,Ly)/np.sqrt(2)
 return q.ravel()
rng=np.random.default_rng(789);x=rng.normal(scale=.004,size=D);u=x.copy();Z=np.outer(x,x);xd=x.copy();errs=[]
for n in range(20):
 # Factor contraction and evolution used in the operational baseline.
 v=m.stream6(u.reshape(N,N,6));xx=v[...,1]-sr[...,1]*v[...,0]/sr[...,0];yy=v[...,2]-sr[...,2]*v[...,0]/sr[...,0]
 q=np.zeros_like(v);fac=(1-rate)/sr[...,0]
 q[...,3]=fac*xx*xx/np.sqrt(2);q[...,4]=fac*xx*yy;q[...,5]=fac*yy*yy/np.sqrt(2)
 x=m.collision_jvp(sr,m.stream6(x.reshape(N,N,6)),rate).ravel()+q.ravel()
 u=m.collision_jvp(sr,v,rate).ravel()
 xd=A@xd+denseB(Z);Z=A@Z@A.T
 errs.append(dict(step=n+1,x_max_abs=float(np.max(abs(x-xd))),Z_max_abs=float(np.max(abs(np.outer(u,u)-Z)))))
out=dict(N=N,steps=20,max_x_error=max(e['x_max_abs'] for e in errs),max_Z_error=max(e['Z_max_abs'] for e in errs))
assert out['max_x_error']<1e-14 and out['max_Z_error']<1e-14
Path(__file__).with_name('baseline_dense_validation.json').write_text(json.dumps(out,indent=2));print(out)
