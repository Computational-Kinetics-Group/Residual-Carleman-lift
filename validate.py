"""Independent dense pair check, Fourier transfer check and frozen-file integrity."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,hashlib
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'carleman_c2'))
from engine import Lift,m,J,old
rng=np.random.default_rng(20261015);N=2;D=6*N*N;rate=-.93
r=rng.normal(0,.0001,(N,N,6));Z=np.outer(r.ravel(),r.ravel())
lifts=[Lift(r,5,1,fast=False),Lift(r,5,1,fast=True)]
errors=[]
for n in range(5):
 sc=rng.normal(0,.01,(N,N,6));sc[...,0]+=1
 d=rng.normal(0,.00001,(N,N,6))
 eye=np.eye(D)
 S=np.column_stack([m.stream6(v.reshape(N,N,6)).ravel() for v in eye])
 A=np.column_stack([J(sc,S[:,j].reshape(N,N,6),rate).ravel() for j in range(D)])
 rho=sc[...,0].ravel();lx=S[1::6]-sc[...,1].ravel()[:,None]/rho[:,None]*S[0::6];ly=S[2::6]-sc[...,2].ravel()[:,None]/rho[:,None]*S[0::6]
 q=np.zeros((N*N,6));fac=(1-rate)/rho
 q[:,3]=fac*np.einsum('ij,ij->i',lx@Z,lx)/np.sqrt(2)
 q[:,4]=fac*np.einsum('ij,ij->i',lx@Z,ly)
 q[:,5]=fac*np.einsum('ij,ij->i',ly@Z,ly)/np.sqrt(2)
 s=A@r.ravel()+q.ravel();dd=d.ravel()
 Z=A@Z@A.T+np.outer(dd,s)+np.outer(s,dd)+np.outer(dd,dd)
 r=(dd+s).reshape(N,N,6)
 for lift in lifts:
  lift.step(sc,d,rate)
  f=lift.f[:,:lift.active];zz=np.outer(f[:,0],f[:,0])
  for j in range(1,lift.active,2):zz+=np.outer(f[:,j],f[:,j+1])+np.outer(f[:,j+1],f[:,j])
  errors.append(max(np.max(abs(lift.r-r)),np.max(abs(zz-Z))))
x=rng.normal(size=(8,8,6));transfer=float(np.max(abs(old.prolong(x,16)[::2,::2]-x)))
assert max(errors)<1e-12 and transfer<1e-12
manifest=P/'SHA256.json'
if manifest.exists():
 for name,digest in json.loads(manifest.read_text()).items():
  assert hashlib.sha256((P/name).read_bytes()).hexdigest()==digest,name
print(json.dumps({'dense_pair_max_abs_error':float(max(errors)),'coincident_transfer_max_abs_error':transfer,'kernels':['standard','fast'],'steps':5},indent=2))
