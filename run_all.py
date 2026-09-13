"""Run experiments in separate processes, avoiding framework/DLL collisions."""
import argparse,subprocess,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--backend',nargs='+',default=['numpy','pytorch','tensorflow'],choices=['numpy','pytorch','tensorflow'])
    p.add_argument('--force',action='store_true')
    args=p.parse_args()
    root=Path(__file__).resolve().parent
    for backend in args.backend:
        for dataset in ['mnist','cifar10','cifar100']:
            for variant in ['baseline','improved']:
                cmd=[sys.executable,'-m','src.train','--backend',backend,'--dataset',dataset,'--variant',variant]
                if args.force:cmd.append('--force')
                subprocess.run(cmd,cwd=root,check=True)

if __name__=='__main__':main()
