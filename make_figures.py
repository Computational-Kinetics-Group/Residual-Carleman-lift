from pathlib import Path
import shutil, io, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def save_pdf(fig,path,**kwargs):
    buf=io.BytesIO()
    fig.savefig(buf,format='pdf',**kwargs)
    raw=buf.getvalue()
    assert raw.rstrip().endswith(b'%%EOF')
    tmp=path.with_suffix('.tmp')
    tmp.write_bytes(raw)
    os.replace(tmp,path)

P=Path(__file__).resolve().parent
R=P.parent
plt.rcParams.update({'font.family':'serif','font.size':8,'axes.labelsize':8,'legend.fontsize':7,'axes.titlesize':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'lines.linewidth':1.25})
colors={'background':'#777777','C1':'#d5791f','C2':'#225ea8','quadratic_direct':'#20945c'}
def history(k):
    return pd.read_csv(P/f'data/history_K{k}.csv')
fine=pd.read_csv(P/'data/N64_Nc32_scores.csv')
pilot=pd.read_csv(P/'data/N32_Nc16_scores.csv')
sweep=pd.read_csv(P/'data/N64_Nc32_sweep.csv')
c2=pd.read_csv(P/'data/C2_Comparison.csv')
base=pd.read_csv(P/'data/baseline_allsteps.csv')
fig,ax=plt.subplots(figsize=(3.35,2.8),layout='constrained')
h=history(28);ss=sweep[sweep.K==28]
ax.semilogy(base.time,np.maximum(base.C2,1e-12),color='#b2182b',label='Conventional C2')
ax.semilogy(ss.time,np.maximum(ss.background_error_percent,1e-12),color=colors['background'],ls=':',label=r'Background, $K=28$')
ax.semilogy(h.time,np.maximum(h.C2,1e-12),color=colors['C2'],label=r'RC2, $K=28$')
ax.semilogy(h.time,np.maximum(h.quadratic_direct,1e-12),color=colors['quadratic_direct'],ls='-.',label='Nonlinear quadratic\nresidual, $K=28$')
ax.set(xlim=(0,3),ylim=(1e-7,150),xlabel=r'Convective time $t^*$',ylabel='Population error (%)',xticks=[0,1,2,3])
ax.legend(loc='lower left',bbox_to_anchor=(0,1.01,1,0.1),mode='expand',ncol=2,fontsize=6.5,frameon=False,borderaxespad=0)
ax.grid(alpha=.16)
save_pdf(fig,P/'figures/fig1_errors.pdf');plt.close(fig)

fig,ax=plt.subplots(figsize=(3.35,2.45),layout='constrained')
ax.semilogy(pilot.K,pilot.C1_peak,'o-',ms=3,color='#333333')
ax.axhline(1,color='#20945c',ls='--',lw=.9)
a=pilot[pilot.K==14].iloc[0];ax.plot(14,a.C1_peak,'o',mfc='white',mec='#225ea8',ms=6)
ax.text(.6,1.15,'1% tolerance',color='#187c4d',fontsize=7)
ax.annotate(r'Selected $K^\prime=14$',xy=(14,a.C1_peak),xytext=(6.8,.73),fontsize=7,arrowprops=dict(arrowstyle='-',lw=.6,color='#555555'))
ax.set(xlabel=r'Pilot cutoff $K^\prime$',ylabel='Peak reconstruction error (%)',xticks=[0,4,8,12,14],xlim=(-.5,14.5),ylim=(.65,240))
ax.grid(alpha=.16)
save_pdf(fig,P/'figures/fig2_preparation.pdf');plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(6.3,2.3),layout='constrained')
axs[0].semilogy(100*c2.nonlinear_fraction,c2.C2_peak,'s-',ms=3.5,color=colors['C2'])
axs[0].set(title='(a)  Target cutoff sweep',xlabel='Nonlinear modes (%)',ylabel='Peak RC2 error (%)')
for _,row in c2.iterrows():
    axs[0].annotate(f'K={int(row.K)}',(100*row.nonlinear_fraction,row.C2_peak),xytext=(4,4),textcoords='offset points',fontsize=6.5)
axs[0].set_xlim(6,72)
axs[1].semilogy(c2.K,c2.cubic_return_rms_D0,'o-',color='#76549a',ms=3.5)
axs[1].set(title='(b)  Cubic first return',xlabel=r'Target cutoff $K$',ylabel=r'RMS norm / $D_0$',xticks=[12,14,22,28])
for ax in axs:ax.grid(alpha=.16)
save_pdf(fig,P/'figures/figS1_diagnostics.pdf');plt.close(fig)

z=np.load(P/'data/rollup_K28.npz')
def vort(a):
    n=len(a);u=a[...,1:3]/(np.sqrt(3)*a[...,0,None]);k=2*np.pi*np.fft.fftfreq(n)*n
    h=np.fft.fft2(u,axes=(0,1))
    return np.fft.ifft2(1j*k[:,None]*h[...,1]-1j*k[None,:]*h[...,0]).real
ns=[800,1600,2400]
ref=[vort(z[f'reference_{n}']) for n in ns];approx=[vort(z[f'C2_{n}']) for n in ns]
lim=max(abs(a).max() for a in ref)
fig,axs=plt.subplots(2,3,figsize=(6.8,4.0),layout='constrained')
for row,fields in enumerate([ref,approx]):
    for col,a in enumerate(fields):
        ax=axs[row,col];im=ax.imshow(a.T,origin='lower',extent=(0,1,0,1),interpolation='nearest',cmap='RdBu_r',vmin=-lim,vmax=lim)
        ax.set_xticks([0,.5,1]);ax.set_yticks([0,.5,1])
        if row==0:ax.set_title(r'$t^* = '+str(col+1)+'$')
        if col==0:ax.set_ylabel('Direct LB' if row==0 else 'RC2, K = 28')
fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.85,pad=.02,label=r'Vorticity $\partial_x u_y-\partial_y u_x$')
save_pdf(fig,P/'figures/fig3_rollup.pdf',dpi=250);plt.close(fig)
