# Residual Carleman lift for nonlinear lattice Boltzmann simulations

## Installation

Tested on Linux with Python 3, NumPy, SciPy, pandas, Matplotlib and GNU C++ with
OpenMP. Windows users can use WSL. The build command assumes GCC-compatible
flags; macOS requires an OpenMP-enabled compiler. No compiled binaries are shipped.

```sh
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python reproduce.py build
python validate.py
```

`CXX` can select another GCC-compatible compiler. The scripts use one BLAS thread;
`--threads` selects the C++ OpenMP thread count. `validation/environment.json`
records the versions used in the package verification; requirements.txt gives
minimum dependencies. requirements-tested.txt pins the versions used here;
these pins do not guarantee bitwise equality across platforms.

## Reproduce the results

Run from this directory:

```sh
# Regenerate all three main figures and the supplemental diagnostic figure
# from frozen data; does not run the flow simulations.
python reproduce.py figures

# Full pilot: target 32^2, coarse 16^2, K'=0,...,14; selects K'=14 at 1%.
python reproduce.py pilot

# Main RC2 experiment: target 64^2, coarse 32^2, K=28, Re=100000, t*=3.
python reproduce.py rc2 --N 64 --Nc 32 --K 28 --Re 100000 --threads 4

# Conventional fixed-rest C2 comparison on the same target lattice.
python reproduce.py baseline

# Compare the three main reruns against archived histories and snapshots.
python compare_reproduction.py

# Additional cutoff diagnostics used in the supplement.
python reproduce.py target-sweep
python reproduce.py rc2 --K 14
python reproduce.py rc2 --K 22

# Small execution check (8/4); not a scientific resolution test.
python reproduce.py smoke
```

Pilot, target sweep and RC2 outputs go to `outputs/<task>/`. Each RC2 case has a
parameter-specific filename. Repeating a case replaces only its generated
outputs. Baseline scripts write generated files beside their code in `baseline/`;
they do not alter `data/`. Figures are written to `figures/` from frozen `data/`.
To plot newly generated results, first compare them with the archived evidence;
figure generation deliberately does not silently substitute new runs.

The full target run uses 2400 steps, with 1200 coarse time intervals. It allocates
1.889 GB of factor buffers and 1.416 GB of population workspace, plus other arrays.
Allow at least 4–5 GB available RAM. Cost grows with both lattice size and horizon;
this implementation stores the exact growing factor representation of the
truncated pair sector, without rank compression. Runtime depends on hardware.

## Numerical construction

D2Q9, six postcollision Hermite fields, periodic unit square. U=0.08,
Re=100000, shear width=0.04, transverse perturbation amplitude=0.05 U.
Each grid uses nu=U N/Re and tau=1/2+3 nu. Code variable `rate` is 1-1/tau,
not the collision frequency omega=1/tau. Time t*=n U/N.

The independent coarse trajectory is prolonged using scipy.signal.resample in
each spatial direction (Fourier interpolation including Nyquist splitting),
and linearly interpolated in time. A circular Fourier mask |k|<=K prepares the
background on the target grid. Collision is evaluated pointwise on that grid,
then projected; there is no extra de-aliasing prescription.

The pilot uses a linear residual correction to select the least retained mode
count passing the tolerance over the prescribed interval. It is a screening
criterion, not a guaranteed RC2 bound. RC2 is subsequently evolved independently.
The nonlinear quadratic residual and direct LB trajectories are diagnostics;
there is no reference feedback into the prepared background or residual lift.
The pair sector is initialized once and never reset to the current residual product.

Stored states have shape (N,N,6), with coordinates
(rho, sqrt(3) j_x, sqrt(3) j_y, a20, a11, a02).
Population reconstruction is f_i=w_i sum_a H_ia a_a using B6 in models.py.
Velocity is (a1,a2)/(sqrt(3) rho). Population errors use the equivalent weighted
population/orthonormal-moment norm, normalized by the initial departure from rest:
100 ||a-a_ref||_2 / ||a_initial-(1,0,0,0,0,0)||_2.
Velocity and vorticity diagnostics use their own current-reference norms.

Historical source/data names `C1` and `C2` in the residual experiments mean linear
residual correction and RC2; the same names in `baseline/` mean conventional
fixed-rest C1/C2. `quadratic_direct` is the nonlinear quadratic diagnostic, not a
lift. `center` means the prepared background. These names are retained to preserve
provenance. Main figures use the manuscript terminology.

## Code and data map

- `carleman_c2/engine.py`, `factors.cpp`, `factors_fast.cpp`: operational RC2
  factor evolution and quadratic contraction.
- `double_shear_c2/experiment.py`: target flow, prepared background, RC2 and
  independent nonlinear diagnostics; snapshots and error histories.
- `spectral_preparation/sweep.py`: pilot and target cutoff sweeps.
- `recovered_shift/model/coarse_stress_inserts/dependencies/`: original dependency
  modules defining D2Q9, six-field collision, Jacobian, coarse interpolation and
  the shear-layer initial condition. Historical unused routines remain unchanged.
- `perturbed_tg/initial_condition.py`: retained import dependency; the DSL driver
  explicitly replaces the imported initial condition with the shear layer.
- `baseline/`: self-contained conventional fixed-rest comparison.
- `data/`: frozen numerical inputs to the manuscript plots.
- `spectral_preparation/C2_K{14,22,28}/`: original full histories and snapshots;
  `double_shear_c2/` also holds the earlier K12 comparison data.
- `docs/Carleman_LBM_Supplement.pdf`: derivation and indexed implementation,
  including explicit Fourier conventions in S6.
- `validation/`: checks performed when preparing this package.
- `SHA256.json`: integrity hashes for distributed source and frozen evidence.

Fig. 1 uses data/baseline_allsteps.csv, history_K28.csv and N64_Nc32_sweep.csv.
Fig. 2 uses N32_Nc16_scores.csv. Fig. 3 uses rollup_K28.npz.
Fig. S1 uses the target sweep and C2_Comparison.csv. Snapshot keys encode the
model and integer target step; the roll-up panels use steps 800,1600,2400.

## Scope and release

Results establish finite-time fidelity to the selected regularized LB model on
its target lattice, not continuum convergence at Re=100000. No runtime advantage
or quantum speedup is claimed. The manuscript and its numerical interpretation
remain the scientific reference.
