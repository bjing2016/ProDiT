import tempfile
import subprocess
from . import protein
from . import residue_constants as rc
import numpy as np


def seqres_to_aatype(seq):
    return [rc.restype_order.get(c, rc.unk_restype_index) for c in seq]

def aatype_to_seqres(aatype):
    return "".join([rc.restypes_with_x[c] for c in aatype])

def write_ca_traj(prot, traj):
    strs = []
    for coords in traj:
        prot.atom_positions[..., 1, :] = coords
        str_ = protein.to_pdb(prot)
        str_ = "\n".join(str_.split("\n")[1:-3])
        strs.append(str_)
    return "\nENDMDL\nMODEL\n".join(strs)


def make_ca_prot(coords, aatype, mask=None):
    L = len(coords)
    if aatype is None:
        aatype = 'A'*L
    if type(aatype) is str:
        aatype = np.array(seqres_to_aatype(aatype))
    prot = protein.Protein(
        atom_positions=np.zeros((L, 37, 3)),
        aatype=aatype,
        atom_mask=np.zeros((L, 37)),
        residue_index=np.arange(L) + 1,
        b_factors=np.zeros((L, 37)),
        chain_index=np.zeros(L, dtype=int),
    )
    if mask is not None:
        prot.atom_mask[..., 1] = mask
    else:
        prot.atom_mask[..., 1] = 1.0
    prot.atom_positions[..., 1, :] = coords
    return prot

#import ihm
#import modelcif
#from modelcif import Assembly, AsymUnit, Entity, System, dumper
#from modelcif.model import AbInitioModel, Atom, ModelGroup
#from rdkit import Chem
#periodic_table = Chem.GetPeriodicTable()

def get_entity(seqres, mol_type, lig_name):
    if mol_type == 0:
        alphabet = ihm.LPeptideAlphabet()
        chem_comp = lambda x: ihm.LPeptideChemComp(id=x, code=x, code_canonical="X")  # noqa: E731
    elif mol_type == 1:
        alphabet = ihm.DNAAlphabet()
        chem_comp = lambda x: ihm.DNAChemComp(id=x, code=x, code_canonical="N")  # noqa: E731
    elif mol_type == 2:
        alphabet = ihm.RNAAlphabet()
        chem_comp = lambda x: ihm.RNAChemComp(id=x, code=x, code_canonical="N")  # noqa: E731
    else:
        alphabet = {}
        chem_comp = lambda x: ihm.NonPolymerChemComp(id=x)  # noqa: E731

    if mol_type < 3:
        seq = [
            alphabet[c] if c in alphabet else chem_comp(item)
            for c in seqres
        ]
    else:
        seq = [chem_comp(lig_name)]
    return Entity(seq)

def get_atoms(asym, seqres, coords, atom_num, mol_type, atom_names=None):
    atoms = []
    for i in range(len(seqres)):
        symbol = periodic_table.GetElementSymbol(
            int(atom_num[i].item())
        ).upper()
        atoms.append(Atom(
            asym_unit=asym, # the asym unit itself
            type_symbol='C' if mol_type < 3 else symbol,
            seq_id=i+1 if mol_type < 3 else 1,
            atom_id='CA' if mol_type < 3 else atom_names[i],
            x=f"{coords[i,0]:.5f}",
            y=f"{coords[i,1]:.5f}",
            z=f"{coords[i,2]:.5f}",
            het=(mol_type==3),       # True/False
            biso=100,     # bfactor
            occupancy=1,
        ))
    return atoms
                     

import io
def write_mmcif(data): # 
    asyms = []
    atoms = []
    for idx in data['chain'].unique():
        mask = data['chain'] == idx
        mol_type = data['mol_type'][mask][0]
        asym = AsymUnit(
            get_entity(
                [data['seqres'][i] for i, c in enumerate(mask) if c],
                data['mol_type'][mask][0],
                lig_name=data['name'].split('_')[-1] if mol_type == 3 else None,
            ),
        )
        asyms.append(asym)
        atoms.extend(get_atoms(
            asym=asym,
            seqres=[data['seqres'][i] for i, c in enumerate(mask) if c],
            coords=data['struct'][mask],
            atom_num=data['atom_num'][mask],
            mol_type=mol_type,
            atom_names=data['atom_name'] if data['mol_type'][mask][0] == 3 else None,
        ))
        
    model = AbInitioModel(Assembly(asyms))
    for at in atoms: model.add_atom(at)
    system = System()
    system.model_groups.append(ModelGroup([model]))
    fh = io.StringIO()
    dumper.write(fh, [system])
    out = fh.getvalue()
    return out
    
def compute_tmscore(
    coords1=None, coords2=None, 
    path1=None, path2=None,
    seq1=None, seq2=None,
    mask1=None, mask2=None, 
    seq=False, exc='TMscore',
):
    if path1 is None:
        file1 = tempfile.NamedTemporaryFile()
        prot1 = make_ca_prot(coords1, seq1, mask=mask1)
        open(file1.name, "w").write(protein.to_pdb(prot1))
        path1 = file1.name
    if path2 is None:
        file2 = tempfile.NamedTemporaryFile()
        prot2 = make_ca_prot(coords2, seq2, mask=mask2)
        open(file2.name, "w").write(protein.to_pdb(prot2))
        path2 = file2.name
    cmd = [exc]
    if seq: cmd += ["-seq"]
    cmd += [path1, path2]    
    
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        print("Status : FAIL", exc.returncode, exc.output)

    if exc == 'TMscore':
        start = out.find(b"RMSD")
        end = out.find(b"rotation")
        out = out[start:end]
        try:
            rmsd, _, tm, _, gdt_ts, gdt_ha, _, _ = out.split(b"\n")
        except Exception as e:
            print(' '.join(cmd))
            raise e
    
        rmsd = float(rmsd.split(b"=")[-1])
        tm = float(tm.split(b"=")[1].split()[0])
        gdt_ts = float(gdt_ts.split(b"=")[1].split()[0])
        gdt_ha = float(gdt_ha.split(b"=")[1].split()[0])
    
        return {"rmsd": rmsd, "tm": tm, "gdt_ts": gdt_ts, "gdt_ha": gdt_ha}
    elif exc == 'TMalign':
        start = out.find(b"TM-score=")
        end = out.find(b"(You should use")
        out = out[start:end]
        tm1, tm2, _ = out.split(b"\n")
        tm1 = float(tm1.split(b'=')[1].split()[0])
        tm2 = float(tm2.split(b'=')[1].split()[0])
        return {"tm": tm2} # usually ref is second
        print(out)
