import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,ctypes
import numpy as np
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'recovered_shift/model/coarse_stress_inserts'))
sys.path.insert(0,str(P.parent/'perturbed_tg'))
from run_tests import m,old,rem,collide,J
from initial_condition import initial,U
lib=ctypes.CDLL(str(P/'factors.so'));ptr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
lib.evolve.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_double,ptr,ptr,ptr,ptr,ptr,ptr,ctypes.c_int]
lib.evolve.restype=None
fastlib=ctypes.CDLL(str(P/'factors_fast.so'))
fastlib.evolve_fast.argtypes=lib.evolve.argtypes[:-1]+[ptr,ctypes.c_int]
fastlib.evolve_fast.restype=None
basis=np.ascontiguousarray(m.B6,dtype=float);weights=np.ascontiguousarray(m.v.w,dtype=float)
def quadratic(s,v,rate):
 rho=s[...,0];x=v[...,1]-s[...,1]*v[...,0]/rho;y=v[...,2]-s[...,2]*v[...,0]/rho
 out=np.zeros_like(v);fac=(1-rate)/rho
 out[...,3]=fac*x*x/np.sqrt(2);out[...,4]=fac*x*y;out[...,5]=fac*y*y/np.sqrt(2)
 return out
class Lift:
 def __init__(self,residual,steps,threads=4,fast=None):
  self.N=residual.shape[0];self.D=residual.size;self.cap=2*steps+3;self.active=1;self.threads=threads
  self.f=np.empty((self.D,self.cap));self.work=np.empty_like(self.f);self.f[:,0]=residual.ravel();self.r=residual.copy()
  self.fast=(self.N>=64) if fast is None else fast
  self.pop=np.empty((self.N*self.N*9,self.cap)) if self.fast else None
 def step(self,sc,d,rate):
  q=np.empty_like(self.r)
  args=(self.N,self.cap,self.active,rate,basis,weights,np.ascontiguousarray(sc),self.f,self.work,q)
  if self.fast:fastlib.evolve_fast(*args,self.pop,self.threads)
  else:lib.evolve(*args,self.threads)
  ar=J(sc,m.stream6(self.r),rate);rn=d+ar+q
  self.work[:,self.active]=d.ravel();self.work[:,self.active+1]=(rn-d/2).ravel()
  self.f,self.work=self.work,self.f;self.active+=2;self.r=rn
  return q
