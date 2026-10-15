import numpy as np
U=.04
EPS=.2/np.sqrt(2)

def fields(N):
    x,y=np.meshgrid(2*np.pi*np.arange(N)/N,2*np.pi*np.arange(N)/N,indexing='ij')
    a=2*x+y+.37;b=x-3*y+.21
    ux=np.sin(x)*np.cos(y)-EPS/np.sqrt(5)*np.sin(a)-3*EPS/np.sqrt(10)*np.cos(b)
    uy=-np.cos(x)*np.sin(y)+2*EPS/np.sqrt(5)*np.sin(a)-EPS/np.sqrt(10)*np.cos(b)
    u=U/np.sqrt(1+.2**2)*np.stack([ux,uy],axis=-1)
    k=2*np.pi*np.fft.fftfreq(N)*N;kx=k[:,None];ky=k[None,:];k2=kx*kx+ky*ky
    def dx(f):return np.fft.ifft2(1j*kx*np.fft.fft2(f)).real
    def dy(f):return np.fft.ifft2(1j*ky*np.fft.fft2(f)).real
    adv=np.stack([u[...,0]*dx(u[...,i])+u[...,1]*dy(u[...,i]) for i in range(2)],axis=-1)
    divadv=dx(adv[...,0])+dy(adv[...,1]);ph=np.zeros((N,N),complex)
    np.divide(np.fft.fft2(divadv),k2,out=ph,where=k2>0)
    pressure=np.fft.ifft2(ph).real
    return u,pressure,adv,dx(u[...,0])+dy(u[...,1])

def initial(N):
    from run_tests import m
    u,p,_,_=fields(N);rho=1+3*p
    a=np.zeros((N,N,6));a[...,0]=rho;a[...,1:3]=np.sqrt(3)*rho[...,None]*u
    a[...,3:]=m.eqkin(a[...,:3])[...,:3]
    return a

def solenoidal(v):
    N=v.shape[0];k=np.fft.fftfreq(N)*N;kx=k[:,None];ky=k[None,:];k2=kx*kx+ky*ky
    h=np.fft.fft2(v,axes=(0,1));dot=kx*h[...,0]+ky*h[...,1];fac=np.zeros_like(dot)
    np.divide(dot,k2,out=fac,where=k2>0)
    h[...,0]-=kx*fac;h[...,1]-=ky*fac
    return h

def audit(N):
    u,p,adv,div=fields(N);s=solenoidal(adv)
    return dict(N=N,velocity_rms=float(np.sqrt(np.mean(np.sum(u*u,axis=-1)))),
        target_velocity_rms=U/np.sqrt(2),divergence_max=float(np.max(abs(div))),
        pressure_mean=float(p.mean()),min_density=float((1+3*p).min()),
        solenoidal_advection_fraction=float(np.linalg.norm(s)/np.linalg.norm(np.fft.fft2(adv,axes=(0,1)))))
