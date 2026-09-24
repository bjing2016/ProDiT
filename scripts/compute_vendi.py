import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', required=True)
parser.add_argument('--count', type=int, default=100)
parser.add_argument('--glob', action='store_true')
parser.add_argument('--seq', action='store_true')
parser.add_argument('--no_tqdm', action='store_true')
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--exc', default='TMscore', choices=['TMscore', 'TMalign'])
parser.add_argument('--out', default=None)
args = parser.parse_args()

import numpy as np
import tqdm, os
from multiprocessing import Pool
from openprot.utils.prot_utils import compute_tmscore

def do(job):
    i, j = job
    path1 = f"{args.dir}/{i}"
    path2 = f"{args.dir}/{j}"
    return compute_tmscore(path1=path1, path2=path2, seq=args.seq, exc=args.exc)['tm']
    

jobs = []

if args.glob:
    files = os.listdir(args.dir)
    files = sorted([f for f in files if f[-4:] == '.pdb'])
    for f in files:
        for g in files:
            jobs.append((f, g))
            
else:
    for i in range(args.count):
        for j in range(args.count):
            jobs.append((f"sample{i}.pdb", f"sample{j}.pdb"))

if args.num_workers > 1:
    p = Pool(args.num_workers)
    p.__enter__()
    __map__ = p.imap
else:
    __map__ = map
if args.no_tqdm:
    tm_arr = list(__map__(do, jobs))
else:
    tm_arr = list(tqdm.tqdm(__map__(do, jobs), total=len(jobs)))
if args.num_workers > 1:
    p.__exit__(None, None, None)

tm_arr = np.array(tm_arr).reshape(int(len(jobs)**0.5), int(len(jobs)**0.5))

if args.out:
    np.save(f"{args.dir}/{args.out}", tm_arr)
tm_arr = tm_arr/2 + tm_arr.T/2
eigvals = np.linalg.eigvals(tm_arr / len(tm_arr))
vendi = np.e**np.nansum(-eigvals * np.log(eigvals))
print(vendi)