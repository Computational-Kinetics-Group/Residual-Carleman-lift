"""Independent Fourier-Galerkin nonlinear core plus degree-one affine lift.
The core does not receive the linear residual or nonlinear-reference data.
"""
from active import *

def coefficients(a,K):
 ft=np.fft.fft(a,axis=0)/len(a)
 return ft[np.arange(-K,K+1)%len(a)].copy()

def reconstruct(c,M):
 K=(len(c)-1)//2
 out=np.zeros((M,)+c.shape[1:],complex);out[np.arange(-K,K+1)%M]=c
 return np.fft.ifft(out,axis=0).real*M

def stream_modes(c,N):
 K=(len(c)-1)//2;k=np.arange(-K,K+1);out=np.zeros_like(c)
 for i in range(9):out+=np.exp(-2j*np.pi*k*m.v.c[i,0]/N)[:,None,None]*(np.roll(c,int(m.v.c[i,1]),axis=1)@m.K6[i])
 return out

def collide(s,r):
 out=s.copy();out[...,3:]=r*s[...,3:]+(1-r)*m.eqkin(s[...,:3])[...,:3]
 return out

def collision_jvp(s,v,r):
 rho=s[...,0];px=s[...,1];py=s[...,2];dr=v[...,0];dx=v[...,1];dy=v[...,2]
 out=v.copy()
 out[...,3]=r*v[...,3]+(1-r)/np.sqrt(2)*(2*px*dx/rho-px*px*dr/rho**2)
 out[...,4]=r*v[...,4]+(1-r)*((py*dx+px*dy)/rho-px*py*dr/rho**2)
 out[...,5]=r*v[...,5]+(1-r)/np.sqrt(2)*(2*py*dy/rho-py*py*dr/rho**2)
 return out

class Hybrid:
 def __init__(self,a,K,r=-.95,quad_factor=1):
  self.N=len(a);self.K=K;self.r=r;self.M=max(1,4*K+1)*quad_factor
  self.c=coefficients(a,K);self.b=reconstruct(self.c,self.N);self.eta=a-self.b
 def core_advance(self):
  sc=stream_modes(self.c,self.N)
  next_c=coefficients(collide(reconstruct(sc,self.M),self.r),self.K)
  return sc,next_c
 def step(self):
  sc,nc=self.core_advance();s=reconstruct(sc,self.N);fb=collide(s,self.r)
  nb=reconstruct(nc,self.N);v=m.stream6(self.eta)
  self.eta=fb-nb+collision_jvp(s,v,self.r)
  self.c=nc;self.b=nb
 def state(self):return self.b+self.eta
 def defect(self):
  s=reconstruct(stream_modes(self.c,self.N),self.N);v=m.stream6(self.eta)
  return collide(s+v,self.r)-collide(s,self.r)-collision_jvp(s,v,self.r)
