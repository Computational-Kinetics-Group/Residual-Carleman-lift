from pathlib import Path
import sys,argparse
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'double_shear_c2'))
import experiment
p=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--K',type=int,required=True);args=ap.parse_args()
# Separate run directories avoid collisions of the old progress filename.
out=p/f'C2_K{args.K}';out.mkdir(exist_ok=True);experiment.P=out
experiment.run(64,32,100000,K=args.K)
