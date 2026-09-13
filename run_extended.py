"""Run the fixed 72-run protocol, reusing the 18 completed seed-42 runs.

Each subprocess owns one framework. Independent CPU seed queues may run
alongside the sequential GPU queue; timings are not isolated benchmarks.
"""
import argparse
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SEEDS = [42, 7, 2026]

def jobs():
    main = [dict(dataset=d, backend=b, variant=v, seed=s, ablation=None)
            for s, d, b, v in itertools.product(SEEDS, ['mnist','cifar10','cifar100'],
                                              ['numpy','pytorch','tensorflow'], ['baseline','improved'])]
    extra = [dict(dataset=d, backend='pytorch', variant='improved', seed=s, ablation=a)
             for d, a, s in itertools.product(['cifar10','cifar100'],
                                              ['no_bn','no_skip','no_dropout'], SEEDS)]
    return main + extra

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--queue', choices=['all','gpu','numpy_7','numpy_2026'], default='all')
    p.add_argument('--plan-only', action='store_true')
    args=p.parse_args()
    if args.plan_only:
        plan=dict(seeds=SEEDS, fixed_split_seed=42, main_runs=54, ablation_runs=18,
                  reused_seed42_runs=18, new_runs=54, epochs=dict(mnist=5,cifar10=10,cifar100=12),
                  selection='minimum validation cross-entropy within fixed epoch budget',
                  interpretation='exploratory extension; original test results have already been observed',
                  ablation_scope='PyTorch on both CIFAR datasets; full improved is the paired reference',
                  ablations=dict(no_bn='Remove all four BN layers, retain all convolutions and Dense layers',
                                 no_skip='Disable identity addition, retain Conv and BN in the branch',
                                 no_dropout='Set dropout probability to zero'),
                  jobs=jobs())
        (ROOT/'results/extended_protocol.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
        print('Protocol saved:',len(plan['jobs']),'runs')
        return
    selected=jobs()
    if args.queue=='gpu':selected=[j for j in selected if j['backend']!='numpy']
    elif args.queue.startswith('numpy_'):
        seed=int(args.queue.split('_')[1]);selected=[j for j in selected if j['backend']=='numpy' and j['seed']==seed]
    env=os.environ.copy();env['PYTHONUNBUFFERED']='1';env['PYTHONUTF8']='1'
    for i,j in enumerate(selected,1):
        cmd=[sys.executable,'-m','src.train']
        for key,value in j.items():
            if value is not None:cmd.extend(['--'+key,str(value)])
        print(f'QUEUE {args.queue} {i}/{len(selected)} {j}',flush=True)
        subprocess.run(cmd,cwd=ROOT,env=env,check=True)
    print('QUEUE COMPLETE',args.queue,flush=True)

if __name__=='__main__':main()
