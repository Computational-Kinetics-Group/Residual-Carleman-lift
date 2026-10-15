"""Moment filtering of standard D2Q9 LBGK, exact density normalization.
All variants retain the same quadratic equilibrium. r is shear relaxation.
"""
import numpy as np
import reference_model as v
from functools import lru_cache
NAMES=['LBGK9','drop_G22_8','drop_G21_G12_7','regularized6','bulk_equilibrated5']

@lru_cache(None)
def rate(name,r=-.95):
    R=r*np.eye(6)
    if name in ['drop_G22_8','regularized6','bulk_equilibrated5']:R[5,5]=0
    if name in ['drop_G21_G12_7','regularized6','bulk_equilibrated5']:R[3,3]=R[4,4]=0
    if name=='bulk_equilibrated5':
        bulk=np.array([1,0,1])/np.sqrt(2);R[:3,:3]-=r*np.outer(bulk,bulk)
    return R

def eqkin(h):
    rho=h[...,0];x=h[...,1];y=h[...,2]
    out=np.zeros(h.shape[:-1]+(6,),dtype=h.dtype)
    out[...,0]=x*x/(np.sqrt(2)*rho);out[...,1]=x*y/rho;out[...,2]=y*y/(np.sqrt(2)*rho)
    return out

def moment_collision(m,name,r=-.95):
    R=rate(name,r);e=eqkin(m[...,:3]);out=m.copy()
    out[...,3:]=m[...,3:]@R.T+e@(np.eye(6)-R).T
    return out

def collision(f,name,r=-.95):return (moment_collision(f@v.H,name,r)@v.H.T)*v.w
def step(f,name,r=-.95):return v.stream(collision(f,name,r))
def equilibrium(u,rho=1):return rho*v.eq(np.asarray(u))
def jacobian(name,u,r=-.95):
    f=equilibrium(u)
    return np.column_stack([np.imag(collision(f.astype(complex)+1j*1e-25*e,name,r))/1e-25 for e in np.eye(9)])

def mul(a,b):
    out=np.zeros(np.broadcast_shapes(a.shape,b.shape))
    for n in range(len(out)):
        for j in range(n+1):out[n]+=a[j]*b[n-j]
    return out

def jetstep(f,name,r=-.95):
    m=f@v.H;rho=m[...,0];inv=np.zeros_like(rho);inv[0]=1
    for n in range(1,len(f)):
        for j in range(1,n+1):inv[n]-=rho[j]*inv[n-j]
    x=m[...,1];y=m[...,2];eq=np.zeros(m.shape[:-1]+(6,))
    eq[...,0]=mul(mul(x,x),inv)/np.sqrt(2);eq[...,1]=mul(mul(x,y),inv);eq[...,2]=mul(mul(y,y),inv)/np.sqrt(2)
    R=rate(name,r);m[...,3:]=m[...,3:]@R.T+eq@(np.eye(6)-R).T
    return v.stream((m@v.H.T)*v.w)

# Exact six-field post-collision realization, with no evolving ghost fields.
B6=v.H[:,:6];K6=v.w[:,None,None]*B6[:,:,None]*B6[:,None,:]
def stream6(a):
    out=np.zeros_like(a)
    for i in range(9):out+=np.roll(a,tuple(v.c[i]),axis=(0,1))@K6[i]
    return out
def step6(a,r=-.95):
    m=stream6(a);m[...,3:]=r*m[...,3:]+(1-r)*eqkin(m[...,:3])[...,:3]
    return m

def reconstruct5(a):
    m=np.zeros(a.shape[:-1]+(9,));m[...,:3]=a[...,:3]
    bulk=(a[...,1]**2+a[...,2]**2)/(2*a[...,0])
    m[...,3]=(bulk+a[...,3])/np.sqrt(2);m[...,4]=a[...,4];m[...,5]=(bulk-a[...,3])/np.sqrt(2)
    return m@v.H.T*v.w
def extract5(f):
    m=f@v.H;return np.stack([m[...,0],m[...,1],m[...,2],(m[...,3]-m[...,5])/np.sqrt(2),m[...,4]],axis=-1)
def step5(a,r=-.95):return extract5(collision(v.stream(reconstruct5(a)),'bulk_equilibrated5',r))
