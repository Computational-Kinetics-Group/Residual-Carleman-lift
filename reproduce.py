"""Portable entry points; generated results never replace archived evidence."""
import argparse, os, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ['OPENBLAS_NUM_THREADS']='1'
p=argparse.ArgumentParser()
p.add_argument('task',choices=['build','figures','pilot','target-sweep','rc2','smoke','baseline'])
p.add_argument('--N',type=int,default=64);p.add_argument('--Nc',type=int,default=32)
p.add_argument('--K',type=int,default=28);p.add_argument('--Re',type=float,default=100000)
p.add_argument('--threads',type=int,default=4)
a=p.parse_args()
if a.task=='build':
 for name in ['factors','factors_fast']:
  subprocess.run([os.environ.get('CXX','g++'),'-O3','-std=c++17','-shared','-fPIC','-fopenmp',str(ROOT/'carleman_c2'/f'{name}.cpp'),'-o',str(ROOT/'carleman_c2'/f'{name}.so')],check=True)
elif a.task=='figures':
 (ROOT/'figures').mkdir(exist_ok=True)
 subprocess.run([sys.executable,str(ROOT/'make_figures.py')],check=True)
elif a.task=='baseline':
 subprocess.run([sys.executable,str(ROOT/'baseline/validate_baseline.py')],check=True)
 subprocess.run([sys.executable,str(ROOT/'baseline/run_baseline.py')],check=True)
else:
 out=ROOT/'outputs'/a.task;out.mkdir(parents=True,exist_ok=True)
 sys.path.insert(0,str(ROOT/'double_shear_c2'))
 import experiment
 if a.task in ['pilot','target-sweep']:
  sys.path.insert(0,str(ROOT/'spectral_preparation'))
  import sweep
  sweep.P=out
  sweep.run(32,16,list(range(15))) if a.task=='pilot' else sweep.run(64,32,[0,8,11,12,14,16,20,22,24,26,28,46])
 else:
  experiment.P=out
  if a.task=='smoke':experiment.run(8,4,100000,K=2,threads=a.threads)
  else:
   if a.N%a.Nc:raise ValueError('N must be an integer multiple of Nc')
   experiment.run(a.N,a.Nc,a.Re,K=a.K,threads=a.threads)
