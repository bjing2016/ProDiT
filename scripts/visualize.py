import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str)
parser.add_argument('--out', type=str)
parser.add_argument('--width', type=int, default=5)
parser.add_argument('--height', type=int, default=5)
parser.add_argument('--annotate', type=str, default='pmpnn_scrmsd')
parser.add_argument('--ss', action='store_true')

args = parser.parse_args()

import matplotlib.pyplot as plt
from pymol import cmd
import tqdm, os
import numpy as np
import pandas as pd
from PIL import Image
from biopandas.pdb import PandasPdb
from openprot.utils.secondary import assign_secondary_structures
from openprot.utils import protein
import torch
fig, axs = plt.subplots(args.width, args.height, figsize=(args.width, args.height), dpi=300)

if args.annotate:
    df = pd.read_csv(f'{args.dir}/info.csv', index_col=0)

def render(path):

    cmd.reinitialize()
    cmd.load(path, 'tmp')
    cmd.set('depth_cue', 0)
    cmd.set('ray_shadows', 0)
    cmd.spectrum('count', 'rainbow')

    if args.ss:
        prot = protein.from_pdb_string(open(path).read())
        ss = assign_secondary_structures(torch.from_numpy(prot.atom_positions[:,1][None]), return_encodings=False, full=False)[0].replace('-', 'l').upper()
        for i, s in enumerate(ss):
            cmd.alter(f"resi {i+1}", f'ss="{s}"')
        cmd.show('cartoon')
        cmd.cartoon('automatic')
    
    cmd.png(f'{path}.png', 640, 640)
    im = np.array(Image.open(f'{path}.png'))
    os.remove(f'{path}.png')
    return im

for i, ax in tqdm.tqdm(enumerate(axs.flatten())):
    path = f"{args.dir}/sample{i}.pdb"
    im = render(path)
    
    if args.annotate:
        row = df.loc[f"sample{i}"]
        ax.text(0, 0, 
            f"{i} {getattr(row, args.annotate, -1):.1f}A \n",
            #f"{int(row.helix*100)}h:{int(row.sheet*100)}s:{int(row.loop*100)}l",
        size=4)
    else:
        plddt = PandasPdb().read_pdb(path).df['ATOM']['b_factor'].mean()
        ax.text(0, 0, f"{i} plddt={round(plddt)}", size=4)
    ax.imshow(im)
    ax.set_axis_off()
args.out = args.out or f"{args.dir}/out.png"
fig.savefig(args.out, bbox_inches='tight', pad_inches=0)
    


