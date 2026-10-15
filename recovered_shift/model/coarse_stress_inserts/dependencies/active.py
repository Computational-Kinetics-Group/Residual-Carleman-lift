"""Exact factor evolution of power-initialized finite Carleman lifts.
No reinitialization, locality mask, or nonlinear readout feedback.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import models as m

def initial(N,U=.08,width=.04,pert=.05):
    x,y=np.meshgrid(np.arange(N)/N,np.arange(N)/N,indexing='ij')
    ux=U*np.where(y<=.5,np.tanh((y-.25)/width),np.tanh((.75-y)/width))
    uy=pert*U*np.sin(2*np.pi*x)
    a=np.zeros((N,N,6));a[...,0]=1;a[...,1]=np.sqrt(3)*ux;a[...,2]=np.sqrt(3)*uy
    a[...,3:]=m.eqkin(a[...,:3])[...,:3]
    return a

def stream_batch(j):
    out=np.zeros_like(j)
    for i in range(9):out+=np.roll(j,tuple(m.v.c[i]),axis=(1,2))@m.K6[i]
    return out

def jetstep(j,r=-.95):
    s=stream_batch(j);P=len(j)-1
    inv=np.zeros(s.shape[:-1],dtype=s.dtype);inv[0]=1
    for n in range(1,P+1):
        for k in range(1,n+1):inv[n]-=s[k,...,0]*inv[n-k]
    products=np.zeros(s.shape[:-1]+(3,),dtype=s.dtype)
    for n in range(2,P+1):
        for k in range(1,n):
            products[n,...,0]+=s[k,...,1]*s[n-k,...,1]/np.sqrt(2)
            products[n,...,1]+=s[k,...,1]*s[n-k,...,2]
            products[n,...,2]+=s[k,...,2]*s[n-k,...,2]/np.sqrt(2)
    s[...,3:]*=r
    for n in range(2,P+1):
        for k in range(2,n+1):s[n,...,3:]+=(1-r)*products[k]*inv[n-k,...,None]
    return s

def velocity(a):return a[...,1:3]/(np.sqrt(3)*a[...,:1])
def vort(a):
    u=velocity(a)
    return (np.roll(u[...,1],-1,0)-np.roll(u[...,1],1,0)-np.roll(u[...,0],-1,1)+np.roll(u[...,0],1,1))/2
