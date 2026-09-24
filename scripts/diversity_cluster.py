import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', required=True)
parser.add_argument('--key', type=str, default='pmpnn_scrmsd', choices=['pmpnn_scrmsd', 'plddt', 'scrmsd'])
args = parser.parse_args()

import pandas as pd
import os, subprocess

subprocess.run(['rm', '-r', f"{args.dir}/designable"])
os.makedirs(f"{args.dir}/designable", exist_ok=True)


df = pd.read_csv(f"{args.dir}/info.csv", index_col=0)
kept = 0
for name, row in df.iterrows():
    keep = False
    if args.key == 'pmpnn_scrmsd' and row.pmpnn_scrmsd <= 2:
        keep = True
    if args.key == 'plddt' and row.plddt >= 70:
        keep = True
    if args.key == 'scrmsd' and row.scrmsd <= 2:
        keep = True
    
    if keep:
        kept += 1
        subprocess.run([
            'cp',
            f"{args.dir}/{name}.pdb",
            f"{args.dir}/designable",
        ])

    

cmd = [
    'foldseek',
    'easy-cluster',
    f"{args.dir}/designable",
    f"{args.dir}/designable/clu",
    f"/tmp",
    "--alignment-type", "1",
    "--cov-mode", "0",
    "--min-seq-id", "0",
    "--tmscore-threshold", "0.5",
]
print(' '.join(cmd))
subprocess.run(cmd, stdout=subprocess.DEVNULL)
print(kept, 'designable')
print(
    len(list(open(
        f"{args.dir}/designable/clu_rep_seq.fasta"
    ))) // 2, 'designable clusters'
)
