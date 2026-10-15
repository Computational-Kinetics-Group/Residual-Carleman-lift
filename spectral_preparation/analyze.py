from pathlib import Path
import json,zipfile
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;root=P.parent
pilot=pd.read_csv(P/'N32_Nc16_scores.csv');fine=pd.read_csv(P/'N64_Nc32_scores.csv');sweep=pd.read_csv(P/'N64_Nc32_sweep.csv');sources=pd.read_csv(P/'N64_Nc32_sources.csv')
selection=json.loads((P/'Coarse_Selections.json').read_text());summary=[];hist={};snaps={};checks={};horizons=[]
colors={12:'tab:red',14:'tab:orange',22:'tab:blue',28:'tab:green'}
for K in [12,14,22,28]:
 folder=root/'double_shear_c2' if K==12 else P/f'C2_K{K}'
 tag=f'N64_Nc32_Re100000_K{K}'
 x=pd.read_csv(folder/f'{tag}_allsteps.csv');hist[K]=x
 z=np.load(folder/f'{tag}_snapshots.npz');snaps[K]=z
 mm=pd.read_csv(folder/f'{tag}_morphology.csv');mf=mm[(mm.time==3)&(mm.model=='C2')].iloc[0]
 sc=fine[fine.K==K].iloc[0]
 j=sweep[sweep.K==K].merge(x,on='time')
 checks[f'K{K}_C1_agreement_pp']=float(np.max(abs(j.C1_error_percent-j.C1)))
 checks[f'K{K}_quadratic_agreement_pp']=float(np.max(abs(j.quadratic_error_percent-j.quadratic_direct)))
 assert checks[f'K{K}_C1_agreement_pp']<1e-7 and checks[f'K{K}_quadratic_agreement_pp']<1e-7
 D=np.linalg.norm(z['reference_0']-np.array([1,0,0,0,0,0]))
 summary.append(dict(K=K,nonlinear_dof=int(sc.nonlinear_dof),nonlinear_fraction=sc.nonlinear_fraction,background_peak=sc.background_peak,C1_peak=float(x.C1.max()),C1_final=x.C1.iloc[-1],C2_peak=float(x.C2.max()),C2_final=x.C2.iloc[-1],quadratic_final=x.quadratic_direct.iloc[-1],C2_to_quadratic_final=100*np.linalg.norm(z['C2_2400']-z['quadratic_2400'])/D,C2_velocity_error=mf.velocity_current_percent,C2_vorticity_error=mf.vorticity_current_percent,C2_vorticity_correlation=mf.vorticity_correlation,C2_enstrophy_ratio=mf.enstrophy_ratio,cubic_return_rms_D0=sc.cubic_return_rms_D0))
 for model in ['C1','C2']:
  for tol in [.1,1,5,10]:
   hit=x[x[model]>tol];horizons.append(dict(K=K,model=model,tolerance_percent=tol,first_crossing=hit.time.iloc[0] if len(hit) else np.nan))
 checks[f'K{K}_reference_difference']=max(float(np.max(abs(z[f'reference_{n}']-snaps[12][f'reference_{n}']))) for n in [0,400,800,1200,1600,2000,2400])
 assert checks[f'K{K}_reference_difference']==0
# Pilot K12 reproduces the prior DSL run; this calibrates flow choice and units.
a=pd.read_csv(P/'N32_Nc16_sweep.csv');a=a[a.K==12];b=pd.read_csv(root/'double_shear_c2/N32_Nc16_Re100000_K12_allsteps.csv');j=a.merge(b,on='time')
checks['pilot_K12_C1_agreement_pp']=float(np.max(abs(j.C1_error_percent-j.C1)))
checks['pilot_K12_quadratic_agreement_pp']=float(np.max(abs(j.quadratic_error_percent-j.quadratic_direct)))
assert checks['pilot_K12_C1_agreement_pp']<1e-7 and checks['pilot_K12_quadratic_agreement_pp']<1e-7
checks['full_projector_max_background_error_percent']=float(fine[fine.K==46].background_peak.iloc[0])
s=pd.DataFrame(summary);s.to_csv(P/'C2_Comparison.csv',index=False);pd.DataFrame(horizons).to_csv(P/'Error_Horizons.csv',index=False);(P/'Reproduction_Checks.json').write_text(json.dumps(checks,indent=2))
# Explicit validation of both coarse-only transfer hypotheses.
trans=[]
for choice in selection['selections']:
 for rule,key in [('same physical K','same_physical_K'),('same spectral fraction','same_fraction_K')]:
  k=choice[key];a=fine[fine.K==k].iloc[0]
  trans.append(dict(tolerance_percent=choice['tolerance_percent'],coarse_K=choice['coarse_K'],rule=rule,fine_K=k,fine_C1_peak=a.C1_peak,passes_C1_tolerance=bool(a.C1_peak<=choice['tolerance_percent'])))
pd.DataFrame(trans).to_csv(P/'Coarse_Transfer_Validation.csv',index=False)
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,3,figsize=(15,4.8),constrained_layout=True)
axs[0].semilogy(pilot.K,pilot.C1_peak,'o-',label='32/16 pilot C1 peak')
for t,c in [(5,'tab:orange'),(1,'tab:green')]:axs[0].axhline(t,color=c,ls='--',label=f'{t}% tolerance')
axs[0].set(title='Coarse-only selection',xlabel='Pilot cutoff K',ylabel='Peak population error (%)')
f=fine[fine.K<=28]
axs[1].semilogy(f.K,f.C1_peak,'o-',color='.5',label='Fine C1 peak')
axs[1].semilogy(s.K,s.C2_peak,'o-',color='tab:blue',label='Fine C2 peak (tested)')
axs[1].set(title='Validation on fixed 64² lattice',xlabel='Fine cutoff K',ylabel='Peak population error (%)')
axs[2].semilogy(100*f.nonlinear_fraction,f.cubic_return_rms_D0,'o-',color='tab:purple')
axs[2].set(title='Omitted cubic source: first return',xlabel='Retained nonlinear Fourier modes (%)',ylabel='RMS return norm / D0')
for ax in axs:ax.grid(alpha=.2)
axs[0].legend(fontsize=8);axs[1].legend(fontsize=8)
fig.suptitle('Spectral preparation audit: double shear layer, Re=100,000, t* ≤ 3',fontsize=15)
fig.savefig(P/'Spectral_Preparation_Selection.png',dpi=170);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
for K in [12,14,22,28]:
 x=hist[K];axs[0].semilogy(x.time,x.C2,label=f'K={K}',color=colors[K])
 a=sweep[sweep.K==K];axs[1].plot(a.time,a.background_error_percent,label=f'K={K}',color=colors[K])
axs[0].axhline(1,color='.5',ls=':',lw=.8);axs[0].set(title='Genuine C2 (unchanged lift)',xlabel='Convective time t*',ylabel='Population error (% of initial norm)',ylim=(1e-8,100))
axs[1].set(title='Prepared background alone',xlabel='Convective time t*',ylabel='Population error (% of initial norm)')
for ax in axs:ax.grid(alpha=.2);ax.legend()
fig.suptitle('Fine/coarse 64/32, Re=100,000: changing only the spectral cutoff',fontsize=14)
fig.savefig(P/'Spectral_Preparation_C2_Errors.png',dpi=170);plt.close(fig)
def vort(a):
 N=len(a);u=a[...,1:3]/(np.sqrt(3)*a[...,0,None]);k=2*np.pi*np.fft.fftfreq(N)*N;f=np.fft.fft2(u,axes=(0,1));return np.fft.ifft2(1j*k[:,None]*f[...,1]-1j*k[None,:]*f[...,0]).real
fig,axs=plt.subplots(4,3,figsize=(10,11),constrained_layout=True)
lim=max(abs(vort(snaps[12][f'reference_{n}'])).max() for n in [800,1600,2400])
for j,n in enumerate([800,1600,2400]):
 for i,(name,a) in enumerate([('Direct fine LB',snaps[12][f'reference_{n}']),('C2, K=12',snaps[12][f'C2_{n}']),('C2, K=22',snaps[22][f'C2_{n}']),('C2, K=28',snaps[28][f'C2_{n}'])]):
  ax=axs[i,j];im=ax.imshow(vort(a).T,origin='lower',extent=(0,1,0,1),cmap='RdBu_r',vmin=-lim,vmax=lim,interpolation='nearest');ax.set_xticks([0,.5,1]);ax.set_yticks([0,.5,1])
  if i==0:ax.set_title(f't*={n*.08/64:g}')
  if j==0:ax.set_ylabel(name)
fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.75,extend='both',label='Vorticity: common reference scale');fig.suptitle('Double-shear roll-up: fixed 64² fine reference, Re=100,000',fontsize=14)
fig.savefig(P/'Spectral_Preparation_Rollup.png',dpi=170);plt.close(fig)
report='''SPECTRAL PREPARATION OF THE DOUBLE-SHEAR RESIDUAL
Fixed fine grid64², Re=100,000, U=.08, final t*=3

QUESTION AND DESIGN
Does the background need a larger spectral cutoff before judging C2?
The fine LB reference, initial state, relaxation rate, time horizon, spatial
Carleman pairs, forcing couplings and C2 truncation are held fixed. Only K in
 c_next=P_K F(c)+(I-P_K)b_next
changes. The original six-field regularized D2Q9 formulation is retained.
Coarse b is Fourier-prolonged and linearly interpolated from32². Relaxation
is scaled with lattice size: tau_N=.5+3 U N/Re. All runs have2400 fine steps.
These are original C2 runs, without the subsequently explored C1-centering
closure. No lift resets, rank compression, nonlinear fine-reference feedback,
measurements or quantum execution are used.

COARSE-ONLY PILOT AND PREDECLARED TRANSFERS
The previous cutoff calibration concerned perturbed Taylor-Green. We repeated
it for DOUBLE SHEAR on32/16 before examining the new64² sweep. An initial
seven-cutoff pilot was expanded to every integer K=0,...,14 to avoid missing
nonmonotone candidates. The authoritative selection is from that expanded set.
Score: peak C1 population error over all1200 pilot time steps, relative to the
independent32² nonlinear reference. Choose the fewest nonlinear Fourier modes
passing5% or1%. The32² reference is a coarse diagnostic for the64² deployment.
Selections were written before starting fine-grid validation. No fine-grid
results enter the selections. These tolerances calibrate C1, not Carleman error.
Two distinct, explicitly hypothesized transfers are tested:
 (a) preserve physical wavenumber K;
 (b) preserve K/N, so double K when going from32 to64.
The second rule preserves roughly the nonlinear mode fraction, not the same
physical wavelength. It is a tested heuristic, not a derived transfer theorem.
The pilot requires two nonlinear base trajectories PLUS candidate projected
predictors and corrector diagnostics; it is not literally just two simulations.
'''+json.dumps(selection,indent=2)
report+='\n\nTRANSFER VALIDATION (C1 PEAK SCORE)\n'+pd.DataFrame(trans).to_string(index=False)
report+='''

FINE-GRID SWEEP AND C2 TESTS
Fine classical sweep: K=0,8,11,12,14,16,20,22,24,26,28,46.
K46 contains the full64² Fourier lattice and is a trivial exact-background
sanity check, not a proposed useful hybrid. C2 is evaluated at K14,K22,K28,
with prior K12 retained as the baseline. K22 andK28 come from the coarse-only
fraction-preserving selections; K14 is the physical-cutoff1% selection.
No C2 result is claimed for the other sweep cutoffs, and no global optimum is
claimed. C1 and the nonlinear quadratic comparator in these C2 runs reproduce
the independently vectorized sweep; fine-reference snapshots match exactly.

C2 RESULTS (PERCENT ERRORS)
'''+s.to_string(index=False)
report+='''

ERROR DEFINITIONS AND SOURCE DIAGNOSTIC
D0=||a_initial-rest|| in six orthonormal population moments. Population errors
are100||a-a_fine||/D0, not percentages of total populations including rest density.
Velocity and vorticity errors use their CURRENT fine-reference norms.
The quadratic residual comparator is evolved nonlinearly and independently.
Its negligible error versus full fine LB distinguishes physical cubic-and-higher
Taylor terms from truncation of the Carleman hierarchy.
Along its trajectory, let a=A r and q=B[r,r]. The omitted degree-two-sector
sources are S3=a tensor q+q tensor a and S4=q tensor q. We measure B_next[S3]
and B_next[S4], their FIRST return to populations, using streamed bilinear
contractions without dense tensors. This return enters r at n+2 from a source
formed at n. RMS uses uniform diagnostic samples, not an accumulated-error
identity or an adjoint sensitivity ranking. No diagnostic source feeds C2.

MODE COUNTS, AUTONOMY AND COST
P_K is a radial disk in integer Fourier wavenumber, applied to all six fields.
Nonlinear spectral DOF=6 times the number of retained Fourier indices; this is
a real-DOF count using conjugate symmetry, not an actual optimized runtime cost.
The full fine dimension is6*64²=24576. A larger K transfers more dynamics into
the nonlinear predictor. The experimental implementation evaluates F on the
full fine lattice then projects; it is not a reduced-cost spectral solver.
For K28 the coarse32 support is entirely inside P_K, so its signal cancels and
the predictor is autonomous P_K F(c), with initial P_K f0. At K22 only coarse
corner modes beyond22 remain; at K12 and14 a larger coarse tail survives.
No quantum-cost or classical-speedup claim is inferred from these accuracy tests.

FINE CLASSICAL SCORES
'''+fine.to_string(index=False)
report+='\n\nPILOT CLASSICAL SCORES\n'+pilot.to_string(index=False)
report+='\n\nERROR HORIZONS\n'+pd.DataFrame(horizons).to_string(index=False)
report+='\n\nNUMERICAL CHECKS\n'+json.dumps(checks,indent=2)
report+='''

LIMITS
One double-shear initial condition, Re=100,000,64² fine grid and finite horizon.
Same-lattice LB fidelity, not resolved continuum DNS. Passing a coarse C1
threshold does not certify fine C1 or C2 error. Fine-grid source diagnostics
validate the candidate choice but are not operational coarse-only inputs.
The comparison does not establish that the retained nonlinear fraction remains
small or favorable with further grid refinement. No adaptive cutoff switching
is used. Images share a reference vorticity scale with saturation outside it;
no smoothing of computed fields is applied.

REPRODUCE
From archive root compile the two kernels:
 g++ -O3 -march=native -fopenmp -shared -fPIC carleman_c2/factors.cpp -o carleman_c2/factors.so
 g++ -O3 -march=native -fopenmp -shared -fPIC carleman_c2/factors_fast.cpp -o carleman_c2/factors_fast.so
Then run:
 python3 spectral_preparation/sweep.py --pilot
 python3 spectral_preparation/sweep.py
 python3 spectral_preparation/run_c2.py --K 14
 python3 spectral_preparation/run_c2.py --K 22
 python3 spectral_preparation/run_c2.py --K 28
 python3 spectral_preparation/analyze.py
Large C2 cases should run sequentially on an8GB worker. Python,numpy,scipy,
pandas,matplotlib,C++/OpenMP required. Prior K12 baseline data and all model
sources are included. The imported TG initialization is overridden by the
DSL experiment's active.initial and U=.08.
'''
findings='\nMAIN NUMERICAL FINDINGS\n'
for _,row in s.iterrows():findings+=f"K={int(row.K)} retains{100*row.nonlinear_fraction:.2f}% of fine Fourier modes: C2 final error{row.C2_final:.6g}%, velocity{row.C2_velocity_error:.6g}%, vorticity{row.C2_vorticity_error:.6g}%.\n"
findings+='These measured changes isolate spectral preparation, not a change to the lift.\n'
findings+='Both spectral-fraction transfers pass their fine C1 tolerances; both same-physical-cutoff transfers fail. This is one-flow empirical validation, not a bound or a universal scaling law.\n'
findings+='At K28 the final C2 vorticity error is below1%, while its final population error is about0.136%; the original K12 failure is therefore not a decisive limitation of C2 with better preparation. The nonlinear predictor now retains about60% of fine Fourier modes.\n'
base=s[s.K==12].iloc[0]
for kk in [22,28]:
 row=s[s.K==kk].iloc[0]
 findings+=f"K{kk}: omitted cubic first-return RMS is reduced by{base.cubic_return_rms_D0/row.cubic_return_rms_D0:.3g} times relative to K12; final C2 error is reduced by{base.C2_final/row.C2_final:.3g} times. These ratios are diagnostic observations, not a fitted error law.\n"

report=report.replace('\nQUESTION AND DESIGN',findings+'\nQUESTION AND DESIGN',1)
(P/'Spectral_Preparation_Report.txt').write_text(report)
with zipfile.ZipFile(P/'Spectral_Preparation_Code_and_Data.zip','w',zipfile.ZIP_DEFLATED) as z:
 for f in P.rglob('*'):
  if f.is_file() and f.suffix in ['.py','.csv','.json','.npz','.txt','.png'] and '__pycache__' not in str(f) and not f.name.startswith('partial_') and f.name!='upload_request.json':z.write(f,f.relative_to(root))
 for name in ['engine.py','factors.cpp','factors_fast.cpp','Algebra_Validation.json','Algebra_Validation_Fast.json']:
  f=root/'carleman_c2'/name;z.write(f,f.relative_to(root))
 f=root/'double_shear_c2/experiment.py';z.write(f,f.relative_to(root))
 for name in ['N64_Nc32_Re100000_K12_allsteps.csv','N64_Nc32_Re100000_K12_snapshots.npz','N64_Nc32_Re100000_K12_morphology.csv','N32_Nc16_Re100000_K12_allsteps.csv']:
  f=root/'double_shear_c2'/name;z.write(f,f.relative_to(root))
 for f in (root/'recovered_shift/model/coarse_stress_inserts').rglob('*.py'):z.write(f,f.relative_to(root))
 f=root/'perturbed_tg/initial_condition.py';z.write(f,f.relative_to(root))
print(s.to_string(index=False));print(pd.DataFrame(trans).to_string(index=False));print(checks)
