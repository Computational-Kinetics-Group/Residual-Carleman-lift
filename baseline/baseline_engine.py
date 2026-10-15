"""Six-field postcollision regularized D2Q9 map used in the baseline.

Same operations and ordering as the original model; standalone for reproducibility.
"""
import numpy as np
c=np.array([(x,y) for x in (-1,0,1) for y in (-1,0,1)])
w=np.prod(np.where(c==0,2/3,1/6),axis=1)
def h(j,x):return (np.ones_like(x),np.sqrt(3)*x,(3*x*x-1)/np.sqrt(2))[j]
labels=[(0,0),(1,0),(0,1),(2,0),(1,1),(0,2)]
B6=np.array([h(i,c[:,0])*h(j,c[:,1]) for i,j in labels]).T
K6=w[:,None,None]*B6[:,:,None]*B6[:,None,:]
def stream6(a):
    out=np.zeros_like(a)
    for i in range(9):out+=np.roll(a,tuple(c[i]),axis=(0,1))@K6[i]
    return out
def eqkin(a):
    rho=a[...,0];x=a[...,1];y=a[...,2]
    out=np.zeros(a.shape[:-1]+(3,),dtype=a.dtype)
    out[...,0]=x*x/(np.sqrt(2)*rho);out[...,1]=x*y/rho;out[...,2]=y*y/(np.sqrt(2)*rho)
    return out
def step6(a,r):
    s=stream6(a);s[...,3:]=r*s[...,3:]+(1-r)*eqkin(s[...,:3]);return s
def collision_jvp(s,v,r):
    rho=s[...,0];px=s[...,1];py=s[...,2];dr=v[...,0];dx=v[...,1];dy=v[...,2]
    out=v.copy()
    out[...,3]=r*v[...,3]+(1-r)/np.sqrt(2)*(2*px*dx/rho-px*px*dr/rho**2)
    out[...,4]=r*v[...,4]+(1-r)*((py*dx+px*dy)/rho-px*py*dr/rho**2)
    out[...,5]=r*v[...,5]+(1-r)/np.sqrt(2)*(2*py*dy/rho-py*py*dr/rho**2)
    return out
def initial(N,U=.08,width=.04,pert=.05):
    x,y=np.meshgrid(np.arange(N)/N,np.arange(N)/N,indexing='ij')
    ux=U*np.where(y<=.5,np.tanh((y-.25)/width),np.tanh((.75-y)/width))
    uy=pert*U*np.sin(2*np.pi*x)
    a=np.zeros((N,N,6));a[...,0]=1;a[...,1]=np.sqrt(3)*ux;a[...,2]=np.sqrt(3)*uy
    a[...,3:]=eqkin(a[...,:3]);return a
