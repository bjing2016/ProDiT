import numpy as np
#from boltz.data.const import prot_token_to_letter, dna_token_to_letter, rna_token_to_letter, prot_letter_to_token
#import ihm
import io
#import modelcif
from modelcif import Assembly, AsymUnit, Entity, System, dumper
from modelcif.model import AbInitioModel, Atom, ModelGroup
#from rdkit import Chem
# from .mmcif import parse_mmcif
# except:
#     print('Cannot import utils.mmcif')
import networkx as nx

#periodic_table = Chem.GetPeriodicTable()

import pickle
try:
    with open('data/boltz/ccd.pkl', 'rb') as f:
        CCD = pickle.load(f)
except:
    print('Could not load CCD')


ATOM = [
    ("name", np.dtype("4i1")),
    ("element", np.dtype("i1")),
    ("charge", np.dtype("i1")),
    ("coords", np.dtype("3f4")),
    ("conformer", np.dtype("3f4")),
    ("is_present", np.dtype("?")),
    ("chirality", np.dtype("i1")),
]

RESIDUE = [
    ("name", np.dtype("<U5")),
    ("res_type", np.dtype("i1")),
    ("res_idx", np.dtype("i4")),
    ("atom_idx", np.dtype("i4")),
    ("atom_num", np.dtype("i4")),
    ("atom_center", np.dtype("i4")),
    ("atom_disto", np.dtype("i4")),
    ("is_standard", np.dtype("?")),
    ("is_present", np.dtype("?")),
]

CHAIN = [
    ("name", np.dtype("<U5")),
    ("mol_type", np.dtype("i1")),
    ("entity_id", np.dtype("i4")),
    ("sym_id", np.dtype("i4")),
    ("asym_id", np.dtype("i4")),
    ("atom_idx", np.dtype("i4")),
    ("atom_num", np.dtype("i4")),
    ("res_idx", np.dtype("i4")),
    ("res_num", np.dtype("i4")),
]

def rmsdalign(
    a, b, weights=None, return_trans=False
):  # alignes B to A  # [*, N, 3]
    B = a.shape[:-2]
    N = a.shape[-2]
    if weights is None:
        weights = np.ones((*B, N))
    weights = weights[...,None]
    a_mean = (a * weights).sum(-2, keepdims=True) / weights.sum(-2, keepdims=True)
    a = a - a_mean
    b_mean = (b * weights).sum(-2, keepdims=True) / weights.sum(-2, keepdims=True)
    b = b - b_mean
    B = np.einsum("...ji,...jk->...ik", weights * a, b)
    u, s, vh = np.linalg.svd(B)

    # Correct improper rotation if necessary (as in Kabsch algorithm)
    sgn = np.sign(np.linalg.det((u @ vh)))
    s[..., -1] *= sgn
    u[..., :, -1] *= sgn[...,None]
    C = u @ vh  # c rotates B to A
    if return_trans:
        return C, a_mean - b_mean @ C.transpose(-1, -2)
    else:
        return b @ C.transpose(-1, -2) + a_mean

    
def convert_atom_name(name: str) -> tuple[int, int, int, int]:
    name = name.strip()
    name = [ord(c) - 32 for c in name]
    name = name + [0] * (4 - len(name))
    return tuple(name)

def unconvert_atom_name(tup):
    # name = name.strip()
    # name = [ord(c) - 32 for c in name]
    # name = name + [0] * (4 - len(name))
    # return tuple(name)
    return "".join(chr(n+32) for n in tup).strip()

class Residue:
    def __repr__(self):
        return f"Residue(name={self.name}, type={self.res_type}, idx={self.res_idx}) with {len(self.atoms)} atoms"
    def to_rdkit(self, ccd=None):
        return (ccd or CCD)[self.name]

    def get_atom_names(self):
        return [unconvert_atom_name(tup) for tup in self.atoms['name']]
        

    def get_letter(self):
        if self.mol_type == 0:
            return prot_token_to_letter.get(self.name, 'X')
        elif self.mol_type == 1:
            return dna_token_to_letter.get(self.name, 'X')
        elif self.mol_type == 2:
            return rna_token_to_letter.get(self.name, 'X')
        else:
            return None

    def from_rdkit(mol, use_coords=True):
        self = Residue()
        atoms = []
        conf = mol.GetConformer()
        pos = conf.GetPositions()
        for a in mol.GetAtoms():
            atoms.append((
                convert_atom_name(a.GetProp('name')),
                a.GetAtomicNum(),
                a.GetFormalCharge(),
                tuple(pos[a.GetIdx()]) if use_coords else (0, 0, 0),
                tuple(pos[a.GetIdx()]),
                False,
                0, # not used
            ))
        self.atoms = np.array(atoms, dtype=ATOM)
        return self

    def get_bonds(self):
        mol = CCD[self.name]
        bonds = []
        names = self.get_atom_names()
        for b in mol.GetBonds():
            i = names.index(b.GetBeginAtom().GetProp('name'))
            j = names.index(b.GetEndAtom().GetProp('name'))
            bonds.append((i, j))
        return bonds
        
    def get_distortion(self):
        bonds = np.array(self.get_bonds())
        lens = self.atoms['coords'][bonds[:,0]] - self.atoms['coords'][bonds[:,1]]
        lens = np.square(lens).sum(-1)**0.5

        ref_lens = self.atoms['conformer'][bonds[:,0]] - self.atoms['conformer'][bonds[:,1]]
        ref_lens = np.square(ref_lens).sum(-1)**0.5
        
        return np.square(ref_lens - lens).mean() ** 0.5
        
        
    def to_graph(self):
        
        nxg = nx.Graph()
        mol = CCD[self.name]
        
    
        for atom in mol.GetAtoms():
            nxg.add_node(atom.GetProp('name'), element=atom.GetSymbol().upper())
    
        # This will list all edges twice - once for every atom of the pair.
        # But as of NetworkX 3.0 adding the same edge twice has no effect,
        # so we're good.
        nxg.add_edges_from([(
            b.GetBeginAtom().GetProp('name'),
            b.GetEndAtom().GetProp('name')
        ) for b in mol.GetBonds()])
        
        return nxg
        
        # if by_atom_index:
        #     nxg = networkx.relabel_nodes(nxg,
        #                                  {a: b for a, b in zip(
        #                                      [a.name for a in residue.atoms],
        #                                      range(len(residue.atoms)))},
        #                                  True)
        # return nxg



class Chain:

    def __repr__(self):
        return f"Chain(name={self.name} mol_type={self.mol_type}) with {len(self.residues)} residues, {len(self.atoms)} atoms"

    def get_residues(self):
        return [self.get_residue(i) for i in range(len(self.residues))]
        
    def get_residue(self, idx):
        resi = Residue()
        row = self.residues[idx]

        astart, acount = row['atom_idx'], row['atom_num']
        
        resi.name = row['name']
        resi.mol_type = self.mol_type
        resi.res_type = row['res_type']
        resi.res_idx = row['res_idx']
        resi.atom_center = row['atom_center'] - astart
        resi.atom_disto = row['atom_disto'] - astart
        resi.is_standard = row['is_standard']
        resi.is_present = row['is_present']

        resi.atoms = self.atoms[astart:astart+acount]
        return resi

    def get_seqres(self):
        return ''.join([r.get_letter() for r in self.get_residues()])

    def get_seqres_mask(self):
        seqres = self.get_seqres()
        seq_mask = np.ones(len(seqres))
        if self.mol_type == 0:
            UNK = 'X'
        else:
            UNK = 'N'
        seq_mask[[c == UNK for c in seqres]] = 0
        return seq_mask
        
    def get_central_atoms(self):
        return self.atoms[self.residues['atom_center']]

    def get_entity(self):
        if self.mol_type == 0:
            alphabet = ihm.LPeptideAlphabet()
            chem_comp = lambda x: ihm.LPeptideChemComp(id=x, code=x, code_canonical="X")  # noqa: E731
        elif self.mol_type == 1:
            alphabet = ihm.DNAAlphabet()
            chem_comp = lambda x: ihm.DNAChemComp(id=x, code=x, code_canonical="N")  # noqa: E731
        elif self.mol_type == 2:
            alphabet = ihm.RNAAlphabet()
            chem_comp = lambda x: ihm.RNAChemComp(id=x, code=x, code_canonical="N")  # noqa: E731
        else:
            alphabet = {}
            chem_comp = lambda x: ihm.NonPolymerChemComp(id=x)  # noqa: E731

        if self.mol_type < 3:
            seq = [
                alphabet[c] if c in alphabet else chem_comp(c)
                for c in self.get_seqres()
            ]
        else:
            seq = [chem_comp(name) for name in self.residues['name']]
        
        return Entity(seq)

    def get_atom_residx(self):
        residx = self.residues['atom_idx']
        residx = (np.arange(len(self.atoms)) >= residx[:,None]).sum(0)
        residx = self.residues['res_idx'][residx-1]
        return residx

class Polypeptide(Chain):
    def __init__(self, seqres, name='A'):
        residues = []
        
        atoms = []
        
        for i, c in enumerate(seqres):
            key = prot_letter_to_token[c]
            mol = CCD[key]
            resi = Residue.from_rdkit(mol, use_coords=False)
            
            residues.append((
                key,
                0, # not used
                i,
                len(atoms),
                len(resi.atoms),
                len(atoms) + resi.get_atom_names().index('CA'),
                len(atoms), # not used
                True,
                True,
            ))    
            atoms.extend(resi.atoms)
        self.atoms = np.array(atoms, dtype=ATOM)
        self.residues = np.array(residues, dtype=RESIDUE)
        
        self.mol_type = 0
        self.name = name
        self.idx = -1 


class Ligand(Chain):
    
        
    def __init__(self, seq, name='A'):
        residues = []
        
        atoms = []
        
        for i, key in enumerate(seq):
            mol = CCD[key]
            resi = Residue.from_rdkit(mol, use_coords=False)
            
            residues.append((
                key,
                0, # not used
                i,
                len(atoms),
                len(resi.atoms),
                len(atoms), # not used
                len(atoms), # not used
                True,
                True,
            ))    
            atoms.extend(resi.atoms)
        self.atoms = np.array(atoms, dtype=ATOM)
        self.residues = np.array(residues, dtype=RESIDUE)
        
        self.mol_type = 3
        self.name = name
        self.idx = -1 

class Structure:
    def __repr__(self):
        return f"Structure with {len(self.chains)} chains, {len(self.residues)} residues, {len(self.atoms)} atoms"
    def from_chains(chains_):
        self = Structure()
        atoms = []
        residues = []
        chains = []
        
        for i, chain in enumerate(chains_):
            chains.append((
                chain.name,
                chain.mol_type,
                0, # entity_id, unused
                0, # sym_id, unused
                i,
                len(atoms),
                len(chain.atoms),
                len(residues),
                len(chain.residues),
            ))
            
            chain_residues = np.copy(chain.residues)
            chain_residues['atom_idx'] += len(atoms)
            chain_residues['atom_center'] += len(atoms)
            chain_residues['atom_disto'] += len(atoms)
            
            residues.extend(chain_residues)
            atoms.extend(chain.atoms)
            
        self.atoms = np.array(atoms, dtype=ATOM)
        self.residues = np.array(residues, dtype=RESIDUE)
        self.chains = np.array(chains, dtype=CHAIN)
        return self

    def get_chains(self):
        return [self.get_chain(i) for i in range(len(self.chains))]
    
    def to_npz(self, path):
        np.savez(
            path,
            chains=self.chains,
            residues=self.residues,
            atoms=self.atoms,
        )
    def from_npz(path):
        self = Structure()
        npz = np.load(path)
        self.chains = npz['chains']
        self.residues = npz['residues']
        self.atoms = npz['atoms']
        return self
        
    def from_mmcif(path):
        self = Structure()
        data = parse_mmcif(path, CCD)
        self.chains = data.data.chains
        self.residues = data.data.residues
        self.atoms = data.data.atoms
        return self
        
    def to_mmcif(self):
        asyms = []
        atoms = []

        for chain in self.get_chains():
            asym = AsymUnit(chain.get_entity(), id=chain.name)
            
            for residue in chain.get_residues():
                for atom in residue.atoms:
                    symbol = periodic_table.GetElementSymbol(
                        int(atom['element'])
                    ).upper()
                    x, y, z = atom['coords']
                    name = unconvert_atom_name(atom['name'])
                    if atom['is_present']:
                        
                        atoms.append(Atom(
                            asym_unit=asym,
                            type_symbol=symbol,
                            seq_id=residue.res_idx+1,
                            atom_id=name,
                            x=x,
                            y=y,
                            z=z,
                            het=(chain.mol_type==3),
                            biso=100,
                            occupancy=1,
                        ))
            asyms.append(asym)
        
        model = AbInitioModel(Assembly(asyms))
        for at in atoms: model.add_atom(at)
        system = System()
        system.model_groups.append(ModelGroup([model]))
        fh = io.StringIO()
        dumper.write(fh, [system])
        out = fh.getvalue()
        return out

    def get_chain(self, idx=None, key=None):
        
        chain = Chain()

        if idx is None:
            idx = np.argwhere(self.chains['name'] == key).flatten()[0]
            
        row = self.chains[idx]
        chain.idx = idx
        chain.name = row['name']
        chain.mol_type = row['mol_type']
        
        rstart, rcount = row['res_idx'], row['res_num']
        astart, acount = row['atom_idx'], row['atom_num']
        
        chain.residues = np.copy(self.residues[rstart:rstart+rcount])
        chain.residues['atom_idx'] -= astart
        chain.residues['atom_center'] -= astart
        chain.residues['atom_disto'] -= astart
        
        chain.atoms = self.atoms[astart:astart+acount]
        
        return chain

    def clash_score(self, ca_only=False, prot_idx=0, lig_idx=1):
        
        lig = self.get_chain(lig_idx)
        lig_atoms = lig.atoms['coords'][lig.atoms['is_present']]
        prot = self.get_chain(prot_idx)
        if ca_only:
            prot_atoms = prot.get_central_atoms()
        else:
            prot_atoms = prot.atoms
        prot_atoms = prot_atoms['coords'][prot_atoms['is_present']]
        dmat = np.square(lig_atoms[None] - prot_atoms[:,None]).sum(-1) ** 0.5
        return dmat.min()

    def ligand_rmsd(self, other, prot_idx=0, lig_idx=1):
        my_prot = self.get_chain(prot_idx)
        my_prot = my_prot.get_central_atoms()

        my_lig = self.get_chain(lig_idx)

        dmat = my_prot['coords'][None] - my_lig.atoms['coords'][:,None]
        dmat = (dmat**2).sum(-1) ** 0.5
        dmat = dmat < 8
        dmat &= my_prot['is_present'][None]
        dmat &= my_lig.atoms['is_present'][:,None]

        bs_mask = np.any(dmat, 0)
        
        other_prot = other.get_chain(prot_idx).get_central_atoms()

        bs_mask &= other_prot['is_present']

        R, t = rmsdalign(
            my_prot['coords'][bs_mask], 
            other_prot['coords'][bs_mask], 
            return_trans=True
        )
        
        
        other.atoms['coords'] = other.atoms['coords'] @ R.T + t
        other_prot = other.get_chain(prot_idx).get_central_atoms()

        other_lig = other.get_chain(lig_idx)
        
        graph = my_lig.get_residue(0).to_graph()
        

        gm = nx.algorithms.isomorphism.GraphMatcher(
            graph, graph, node_match=lambda x, y: x["element"] == y["element"]
        )
        symmetries = []
        for i, isomorphism in enumerate(gm.isomorphisms_iter()):
            """
            if i >= max_symmetries:
                raise TooManySymmetriesError(
                    "Too many symmetries between %s and %s" % (
                        str(model_ligand), str(target_ligand)))
            """
            symmetries.append((list(isomorphism.values()),
                               list(isomorphism.keys())))

        my_names = my_lig.get_residue(0).get_atom_names()
        other_names = other_lig.get_residue(0).get_atom_names()

        best = np.inf
        
        if len(symmetries) > 1e2:
            return np.nan
        
        for my_key, other_key in symmetries:
            my_key = [my_names.index(k) for k in my_key]
            other_key = [other_names.index(k) for k in other_key]

            my_coords = my_lig.atoms['coords'][my_key]
            other_coords = other_lig.atoms['coords'][other_key]

            my_mask = my_lig.atoms['is_present'][my_key]
            other_mask = other_lig.atoms['is_present'][other_key]

            mask = (my_mask & other_mask).astype(float)
            
            dist = np.square(my_coords - other_coords).sum(-1)
            
            rmsd = ((dist * mask).sum() / mask.sum())**0.5
            best = min(best, rmsd)
        return best
        
            
        



# struct = Structure.from_npz(f"/data/cb/scratch/datasets/boltz/processed_data/structures/a3/1a3n.npz")

# chain = struct.get_chain(0)
# resi = chain.get_residue(0)
# print(resi)
if __name__ == '__main__':
    struct = Structure.from_chains([
        Polypeptide('MKVLTVTTVL', name='A'),
        Ligand(['HEM'], name='B'),
    ])
    struct = Structure.from_mmcif(
        'workdir/default/eval_step0/binder/sample0_M7K.mmcif'
    )
    with open('out.cif', 'w') as f:
        f.write(struct.to_mmcif())
