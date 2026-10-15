"""Independent recovery of the September K0 collide-stream Schur model.
No fit, trajectory reset, or projection of the doubled reference onto compatibility.
All population arrays have last axis nine; velocities use x,y array axes.
"""
from pathlib import Path
import argparse,json,csv,time
import numpy as np
OUT=Path(__file__).resolve().parent/'results'
c=np.array([(x,y) for x in (-1,0,1) for y in (-1,0,1)])
w=np.prod(np.where(c==0,2/3,1/6),axis=1)
h=lambda j,x:(np.ones_like(x),np.sqrt(3)*x,(3*x*x-1)/np.sqrt(2))[j]
labels=[(0,0),(1,0),(0,1),(2,0),(1,1),(0,2),(2,1),(1,2),(2,2)]
H=np.array([h(i,c[:,0])*h(j,c[:,1]) for i,j in labels]).T
B=H[:,3:];WB=w[:,None]*B
P=WB@B.T # row vector v @ P = kinetic projection in velocity-value coordinates
PH=(w[:,None]*H[:,1:3])@H[:,1:3].T

def stream(v):
    if v.ndim<3:return v.copy()
    return np.stack([np.roll(v[...,i],tuple(c[i]),axis=(-2,-1)) for i in range(9)],axis=-1)

def source(x,kind):
    # Numerically stable small-increment defect: avoid cancellation with log.
    start=3 if kind=='pc1' else 5
    if np.max(abs(x))<.5:
        out=np.zeros_like(x)
        for p in range(start,55,2):out-=2*x**p/p
        return out
    inc=2*x if kind=='pc1' else 2*(x+x**3/3)
    return inc-2*np.arctanh(x)

def parent(f,d,gamma,kind='pc3',tol=2e-14):
    """Full local doubled trapezoidal K0 collision. d=mu-log(f/w).
    exact: logarithmic compatibility (d is transported unchanged).
    mem3/mem5: keep leading PC1/PC3 defect source, retain nonlinear logarithms.
    Returns collision only; stream BOTH f and d outside.
    """
    rho=f.sum(-1);s=f/(rho[...,None]*w)-1
    k=s@WB;M=s@PH;eta=np.log(f/w)+d
    r=(1-gamma/2)/(1+gamma/2)
    y=r*k
    maxres=0.
    for it in range(45):
        sp=M+y@B.T;den=2+sp+s;x=(sp-s)/den
        dx=2*(1+s)/den**2
        if kind=='pc1':inc=2*x;der=2*dx
        elif kind=='pc3':inc=2*(x+x**3/3);der=2*(1+x*x)*dx
        else:
            inc=np.log1p(sp)-np.log1p(s);der=1/(1+sp)
            if kind in ('mem3','mem5'):
                p=3 if kind=='mem3' else 5
                inc-=2*x**p/p;der-=2*x**(p-1)*dx
        res=y-k+gamma/2*((2*eta+inc)@WB)
        err=np.max(abs(res));maxres=max(maxres,float(err))
        if err<tol:break
        jac=np.eye(6)+gamma/2*np.einsum('ia,...i,ib->...ab',WB,der,B)
        step=np.linalg.solve(jac,res[...,None])[...,0]
        # Per-node line search protects positivity of f+.
        rate=np.ones_like(rho)
        for ls in range(35):
            trial=M+(y-rate[...,None]*step)@B.T
            bad=np.min(1+trial,axis=-1)<=1e-8
            if not np.any(bad):break
            rate=np.where(bad,rate/2,rate)
        y-=rate[...,None]*step
    else:raise RuntimeError(f'Newton failed {kind}: {err}')
    fn=rho[...,None]*w*(1+sp)
    if kind in ('pc1','pc3'):dn=d+source(x,kind)
    elif kind in ('mem3','mem5'):dn=d-2*x**(3 if kind=='mem3' else 5)/(3 if kind=='mem3' else 5)
    else:dn=d.copy()
    if fn.min()<=0:raise RuntimeError('positivity')
    return fn,dn,float(err)

def reduced(f,gamma,order=3,d=None):
    """Recovered density-normalized quadratic/cubic map, all nine fields.
    Optional leading PC1 memory correction is cubic-order consistent.
    """
    rho=f.sum(-1);s=f/(rho[...,None]*w)-1
    M=s@PH;K=s@P;r=(1-gamma/2)/(1+gamma/2)
    A=M+K;Ar=M+r*K
    q2=(1-r)/4*((A*A+Ar*Ar)@P)
    q3=(1-r)/2*((Ar*q2)@P)-(1-r)/6*((A**3+Ar**3)@P)
    kp=r*K+q2+(q3 if order>=3 else 0)
    if d is not None:
        src=(1-r)**3*K**3/12
        kp-=(1-r)*((d+src/2)@P)
        dn=d+src
    else:dn=None
    return rho[...,None]*w*(1+M+kp),dn

def norm(v):return np.sqrt(np.sum(v*v/w,axis=tuple(range(v.ndim-1))+(-1,)))
def velocity(f):return (f@c)/f.sum(-1)[...,None]
def eq(u):
    cu=u@c.T
    return w*(1+3*cu+4.5*cu**2-1.5*np.sum(u*u,axis=-1)[...,None])
def initial(N,U,case):
    x,y=np.meshgrid(np.arange(N)/N,np.arange(N)/N,indexing='ij')
    if case=='shear':
        ux=np.where(y<=.5,np.tanh(30*(y-.25)),np.tanh(30*(.75-y)))
        uy=.05*np.sin(2*np.pi*(x+.25))
    elif case=='tgv':ux=np.sin(2*np.pi*x)*np.cos(2*np.pi*y);uy=-np.cos(2*np.pi*x)*np.sin(2*np.pi*y)
    else:raise ValueError(case)
    u=U*np.stack([ux,uy],-1)
    return eq(u)

def csvsave(name,rows):
    with (OUT/name).open('w') as f:
        wr=csv.DictWriter(f,fieldnames=rows[0]);wr.writeheader();wr.writerows(rows)

def local_tests():
    rng=np.random.default_rng(72461);v=rng.normal(size=(24,9))@H.T
    v/=np.max(abs(v),axis=-1)[:,None]
    rows=[];checks={'orthogonality_error':float(np.max(abs(H.T@(w[:,None]*H)-np.eye(9))))}
    for gamma in (2/3,4.,78.):
        for a in (.08,.04,.02,.01,.005):
            f=w*(1+a*v);d=np.zeros_like(f)
            f1,d1,_=parent(f,d,gamma,'pc1');f3,d3,_=parent(f,d,gamma,'pc3');fe,de,_=parent(f,d,gamma,'exact')
            f2,_=reduced(f,gamma,2);fc,_=reduced(f,gamma,3);fm,dm=reduced(f,gamma,3,d)
            mem,md,_=parent(f,d,gamma,'mem3')
            row=dict(gamma=gamma,amplitude=a,quadratic_vs_pc1=norm(f2-f1),cubic_vs_pc3=norm(fc-f3),cubic_vs_exact=norm(fc-fe),
                     cubic_vs_pc1=norm(fc-f1),cubic_memory_vs_pc1=norm(fm-f1),log_memory_vs_pc1=norm(mem-f1),
                     pc1_vs_exact=norm(f1-fe),pc3_vs_exact=norm(f3-fe),pc1_defect=norm(w*d1),pc3_defect=norm(w*d3))
            rows.append(row)
    csvsave('local_order_tests.csv',rows)
    # History-dependent neutral defect: homogeneous kinetic initial state.
    K=B[:,0];r=.5;g=2/3;a=.02
    f=w*(1+a*K);d=np.zeros(9);history=[]
    for n in range(81):
        pred=(1-r)**3/(12*(1-r**3))*(1-r**(3*n))*a**3*K**3
        history.append(dict(step=n,defect=norm(w*d),leading_prediction=norm(w*pred),prediction_error=norm(w*(d-pred)),kinetic_norm=norm(w*((f/w-1)@P))))
        if n<80:f,d,_=parent(f,d,g,'pc1')
    csvsave('homogeneous_memory.csv',history)
    (OUT/'local_checks.json').write_text(json.dumps(checks,indent=2))
    print('local checks',checks,flush=True)

# Directional initial-amplitude jets: exact finite-degree Carleman action on
# the recovered nonlinear map along this initial family. Never recentered.
def mul(a,b):
    out=np.zeros(np.broadcast_shapes(a.shape,b.shape))
    for n in range(len(out)):
        for k in range(n+1):out[n]+=a[k]*b[n-k]
    return out

def jet_step(z,gamma):
    rho=np.sum(z*w,axis=-1,keepdims=True);rho[0]+=1
    inv=np.zeros_like(rho);inv[0]=1
    for n in range(1,len(z)):
        for j in range(1,n+1):inv[n]-=rho[j]*inv[n-j]
    s=mul(z+np.eye(len(z),1).reshape((len(z),)+(1,)*(z.ndim-1)),inv);s[0]-=1
    M=s@PH;K=s@P;r=(1-gamma/2)/(1+gamma/2)
    A=M+K;Ar=M+r*K
    q2=(1-r)/4*((mul(A,A)+mul(Ar,Ar))@P)
    q3=(1-r)/2*(mul(Ar,q2)@P)-(1-r)/6*((mul(mul(A,A),A)+mul(mul(Ar,Ar),Ar))@P)
    out=M+r*K+q2+q3;out[0]+=1
    out=mul(rho,out);out[0]-=1
    return stream(out)

def run(N=32,U=.08,gamma=78.,steps=300,case='shear',jet=True):
    f0=initial(N,U,case);scale=norm(f0-w);u0=velocity(f0)
    names=['pc1','pc3','exact','mem3','mem5','quadratic','cubic','cubic_mem3']
    states={k:(f0.copy(),np.zeros_like(f0)) for k in names}
    ref_cons=np.r_[f0.sum(),np.sum(f0@c,axis=(0,1))]
    diag={k:dict(max_mass_drift=0.,max_momentum_drift=0.,min_population=float(f0.min()),max_residual=0.) for k in names}
    if jet:
        z=np.zeros((5,N,N,9));uu=velocity(initial(N,1.,case)) # only polynomial evaluation; may be nonpositive
        cu=uu@c.T
        z[1]=3*cu;z[2]=4.5*cu**2-1.5*np.sum(uu*uu,axis=-1)[...,None]
        powers=U**np.arange(5)
    rows=[];t0=time.time();snap={}
    for n in range(steps+1):
        row={'step':n}
        for name,(f,d) in states.items():
            ref=states['pc1' if name in ('mem3','cubic_mem3') else 'pc3'][0]
            row[name+'_pop_vs_parent']=norm(f-ref)/scale
            row[name+'_velocity_vs_parent']=np.sqrt(np.mean(np.sum((velocity(f)-velocity(ref))**2,axis=-1)))/U
            row[name+'_defect']=norm(w*d)/scale
            row[name+'_signal']=norm(f-w)/scale
            con=np.r_[f.sum(),np.sum(f@c,axis=(0,1))]
            diag[name]['max_mass_drift']=max(diag[name]['max_mass_drift'],float(abs(con[0]-ref_cons[0])))
            diag[name]['max_momentum_drift']=max(diag[name]['max_momentum_drift'],float(np.max(abs(con[1:]-ref_cons[1:]))))
            diag[name]['min_population']=min(diag[name]['min_population'],float(f.min()))
        row['pc1_vs_exact']=norm(states['pc1'][0]-states['exact'][0])/scale
        row['pc3_vs_exact']=norm(states['pc3'][0]-states['exact'][0])/scale
        row['cubic_vs_pc1']=norm(states['cubic'][0]-states['pc1'][0])/scale
        row['cubic_vs_exact']=norm(states['cubic'][0]-states['exact'][0])/scale
        if jet:
            for grade in (2,3,4):
                fj=w*(1+np.einsum('p,p...->...',powers[:grade+1],z[:grade+1]))
                row[f'lift{grade}_vs_cubic']=norm(fj-states['cubic'][0])/scale
                row[f'lift{grade}_vs_pc3']=norm(fj-states['pc3'][0])/scale
        rows.append(row)
        if n in (0,steps//2,steps):
            for name in ('pc1','pc3','cubic','mem3'):snap[f'{name}_f_{n}']=states[name][0].copy();snap[f'{name}_d_{n}']=states[name][1].copy()
        if n==steps:break
        for name,(f,d) in states.items():
            if name in ('quadratic','cubic'):fn,dn=reduced(f,gamma,2 if name=='quadratic' else 3);dn=d
            elif name=='cubic_mem3':fn,dn=reduced(f,gamma,3,d)
            else:
                fn,dn,res=parent(f,d,gamma,name);diag[name]['max_residual']=max(diag[name]['max_residual'],res)
            if not np.all(np.isfinite(fn)) or fn.min()<=0:raise RuntimeError(f'Invalid {name} at {n+1}')
            states[name]=(stream(fn),stream(dn))
        if jet:z=jet_step(z,gamma)
        if (n+1)%100==0:print(case,N,U,gamma,n+1,'seconds',round(time.time()-t0,1),flush=True)
    tag=f'{case}_N{N}_U{U:g}_g{gamma:g}'
    csvsave(tag+'.csv',rows)
    np.savez_compressed(OUT/(tag+'_snapshots.npz'),**snap)
    (OUT/(tag+'_checks.json')).write_text(json.dumps(dict(N=N,U=U,gamma=gamma,steps=steps,case=case,diagnostics=diag,seconds=time.time()-t0),indent=2))
    print('FINISHED',tag,rows[-1],flush=True)
    return rows

if __name__=='__main__':
    pa=argparse.ArgumentParser();pa.add_argument('--local',action='store_true');pa.add_argument('--N',type=int,default=32);pa.add_argument('--U',type=float,default=.08);pa.add_argument('--gamma',type=float,default=78.);pa.add_argument('--steps',type=int,default=300);pa.add_argument('--case',default='shear');pa.add_argument('--no-jet',action='store_true');a=pa.parse_args();OUT.mkdir(exist_ok=True)
    if a.local:local_tests()
    else:run(a.N,a.U,a.gamma,a.steps,a.case,not a.no_jet)
