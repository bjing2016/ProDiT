import torch
import numpy as np
import pandas as pd
#import foldcomp
import os
from ..utils import protein
from ..utils import residue_constants as rc
from .data import OpenProtDataset, OpenProtData
import json
#from ..utils.structure import Structure
from io import BytesIO

class UnifiedDataset(OpenProtDataset):
    def setup(self):
        arr = np.load(self.cfg.index)
        if self.cfg.old_fmt:
            lens = np.diff(arr[:,0])
            mask = arr[:,1] >= self.cfg.plddt_thresh
            self.index = arr[mask,0]
            self.lens = lens[mask[:-1]]
        else:
            if self.cfg.alphafill:
                mask = arr[:,-1] >= self.cfg.plddt_thresh
            else:
                mask = arr[:,-2] >= self.cfg.plddt_thresh    
            self.index = arr[mask]
        if self.cfg.struct:
            self.afdb = foldcomp.open(self.cfg.afdb)

        with open(self.cfg.go_vocab, 'r') as file:
            self.go_vocab = json.load(file)

        if self.cfg.site_vocab:
            with open(self.cfg.site_vocab, 'r') as file:
                self.site_vocab = json.load(file)
        
    def __len__(self):
        return len(self.index) - 1

    def __getitem__(self, idx: int):
        import time
        start_t = time.time()
        
        with open(self.cfg.path) as f:
            if self.cfg.old_fmt:
                end = self.index[idx + 1]
                f.seek(self.index[idx])
                line = f.read(self.lens[idx])
            else:
                f.seek(self.index[idx][0])
                line = f.read(self.index[idx][1])
            
        js = json.loads(line)
        dur1 = time.time()-start_t

        def filter_func(entry):
            if entry['afdb'][1] < self.cfg.plddt_thresh:
                return False
            if self.cfg.alphafill and 'alphafill' not in entry:
                return False
            return True
            
        # filter based on plddt
        for key90 in list(js.keys()):
            js[key90] = {
                key100: js[key90][key100] for key100 in js[key90] \
                if filter_func(js[key90][key100])
            }
            if len(js[key90]) == 0:
                del js[key90]
            
            
        entry = np.random.choice(list(
            np.random.choice(list(js.values())).values()
        ))
        dur2 = time.time() - start_t
        seqres = None
        name = None
        if 'ur' in entry: # shouldn't be the case if using assemble_afdb
            with open(self.cfg.uniref) as f:
                f.seek(entry['ur'][0])
                lines = f.read(entry['ur'][1]).split("\n")
            
            header, lines = lines[0], lines[1:]
            name = header if len(header.split()) == 0 else header.split()[0]
            seqres = "".join(lines)
            
            # afdb is not a fragment
            if self.cfg.struct and len(seqres) == entry['afdb'][2]: 
                require_struct = True
            else:
                require_struct = False
        else:
            require_struct = True

        ## The logic in this part is kind of ugly.
        
        struct = None
        struct_mask = None
        dur3 = time.time() - start_t
        if require_struct:
            afdb_name, pdb = self.afdb[entry['afdb'][0]]
            
            prot = protein.from_pdb_string(pdb)
            # print(self.num / self.denom)
            afdb_seqres = "".join([rc.restypes_with_x[c] for c in prot.aatype])
            if (seqres is None) or seqres == afdb_seqres:
                struct = prot.atom_positions[:,1]
                struct_mask = prot.atom_mask[:,1]

            # get uniref100 seq to compare with afdb seq
            if 'uniref_seq' in entry:  
                pos, length = entry['uniref_seq']
                with open(self.cfg.uniref) as f:
                    f.seek(pos)
                    lines = f.read(length).split("\n")
                    header, lines = lines[0], lines[1:]
                    uniref_seqres = "".join(lines)
        
        name = name or afdb_name
        seqres = seqres or afdb_seqres
        seq_mask = np.ones(len(seqres), dtype=np.float32)
        seq_mask[[c not in rc.restype_order for c in seqres]] = 0
        residx = np.arange(len(seqres), dtype=np.float32)

        dur4 = time.time() - start_t
        cath = np.zeros((len(seqres), 3))
        if 'ted' in entry and self.cfg.ted:            
            # print('loading cath')
            with open(self.cfg.ted) as f:
                f.seek(entry['ted'][0])
                line = f.read(entry['ted'][1])
            for dom in line.strip().split('\n'):
                bounds = dom.split()[3]
                mask = np.zeros(len(seqres), dtype=bool)
                for interval in bounds.split('_'):
                    start, end = interval.split('-')
                    mask[int(start)-1:int(end)] = True
                label = dom.split()[13].strip()
                if label == '-': continue
                for i, c in enumerate(label.split('.')[:3]):
                    cath[mask,i] = int(c)

        nma = None
        nma_mask = None
        if 'nma' in entry and self.cfg.nma:
            fidx, pos, length = entry['nma']
            path = f"{self.cfg.nma}.{fidx}"
            with open(path, 'rb') as f:
                f.seek(pos)
                nma = np.load(BytesIO(f.read(length)))
            nma_mask = np.ones(len(nma))

        term_sets = [set() for _ in range(len(seqres))] # start with sets for efficiency when combining

        go_term = 0
        if 'go_uniprotkb' in entry:
            pos, length = entry['go_uniprotkb']
            with open(self.cfg.go_uniprotkb) as f:
                f.seek(pos)
                line = f.read(length)
                
                header, terms = line.split("\n")[:2]
                go_terms = np.array([term for term in terms.strip().split(',') if term in self.go_vocab])
                
                if len(go_terms) > 0:
                    go_idx = np.array([self.go_vocab[term][0] for term in go_terms])
                    go_freqs = np.array([self.go_vocab[term][1] for term in go_terms])
                    go_probs = 1/go_freqs / (1/go_freqs).sum()
                    go_term = np.random.choice(go_idx, p=go_probs)
                        

        if 'go_ips' in entry and self.cfg.go_ips: 
            pos, length = entry['go_ips']
            with open(self.cfg.go_ips) as f:
                f.seek(pos)
                line = f.read(length)
                split_line = line.split("\n")
                header, go_terms = split_line[0], split_line[1:]
                for term_list in go_terms:
                    if len(term_list) > 0: # if term list is not empty
                        if 'uniref_seqres' in locals(): # if we are able to confirm the sequence for mapping
                            terms, positions = term_list.split(maxsplit=1)
                            go_terms_split = terms.split(',') # get term indices
                            go_term_indices = list(set(np.array([go_index for go_term in go_terms_split if (go_index := self.go_vocab.get(go_term)) is not None], dtype=int)))
                            positions_split = positions.split(';') # update term_sets at positions
                            for pos_pair in positions_split:
                                start, end = pos_pair.strip().split()
                                start, end = int(start) - 1, int(end) - 1 # start, end 1-indexed and inclusive

                            # check for position re-mapping
                            if len(uniref_seqres) != len(seqres):
                                positions_list = map_positions_to_subset(uniref_seqres, seqres, [start, end], is_range=True)
                                if positions_list is not None and positions_list:
                                    start, end = positions_list[0], positions_list[-1]
                                else: # subset not found 
                                    # print('mismatch2', flush=True)
                                    start, end = 0, 0 # cannot add labels due to sequence mismatch  

                            for pos in range(start, end + 1):
                                term_sets[pos].update(go_term_indices)

        go_term_array = np.zeros((len(seqres), self.cfg.max_go_terms), dtype=int) 
        for i, term_set in enumerate(term_sets):
            terms = list(term_set)
            go_term_array[i, :min(len(terms), self.cfg.max_go_terms)] = terms[:self.cfg.max_go_terms]

        site_array = np.zeros((len(seqres), self.cfg.max_sites), dtype=int)
        if 'feat_uniprotkb' in entry and self.cfg.sites_uniprotkb:
            pos, length = entry['feat_uniprotkb']
            with open(self.cfg.sites_uniprotkb) as f: 
                f.seek(pos)
                line = f.read(length)
                split_line = line.split("\n")
                header, active_site, binding_site = split_line[0], split_line[1], split_line[2]

                # process active site - always first 
                active_site = active_site.split(':')[1].strip()
                if active_site != 'None':

                    positions_list = list(map(lambda x: int(x) - 1, active_site.split(","))) # convert from 1-index to 0-index

                    # check if we need to re-map positions
                    if 'uniref_seqres' in locals(): # check if uniref_seqres exists (e.g. multiple cluster members)
                        if len(uniref_seqres) != len(seqres): # check if we need to re-map positions
                            positions_list = map_positions_to_subset(uniref_seqres, seqres, positions_list, is_range=False)

                            if positions_list is None: # cannot assign labels due to sequence mismatch
                                # print('mismatch', flush=True)
                                positions_list = []

                    for pos in positions_list: 
                        site_array[pos, 0] = self.site_vocab['active_site'] 

                # process binding site
                binding_site = binding_site.split(':')[1].strip()
                if binding_site != 'None':
                    ligands = binding_site.split(';')[:-1] # last element will always be empty list 
                    for ligand in ligands:
                        ligand_name, positions = ligand.rsplit(None, 1)
                        ligand_name = ligand_name.strip()

                        positions_list = list(map(lambda x: int(x) - 1, positions.split(","))) # convert from 1-index to 0-index

                        # check if we need to re-map positions
                        if 'uniref_seqres' in locals(): # check if uniref_seqres exists (e.g. multiple cluster members)
                            if len(uniref_seqres) != len(seqres):
                                positions_list = map_positions_to_subset(uniref_seqres, seqres, positions_list, is_range=False)

                                if positions_list is None: # cannot assign labels due to sequence mismatch
                                    # print('mismatch', flush=True)
                                    positions_list = []

                        for pos in positions_list: 
                            col = np.argmax(site_array[pos] == 0) # find where to insert
                            site_array[pos, col] = self.site_vocab[ligand_name] 
                              
        
        data = self.make_data(
            name=name,
            seqres=seqres,
            residx=residx,
            seq_mask=seq_mask,
            struct=struct,
            struct_mask=struct_mask,
            cath=cath,
            nma=nma,
            nma_mask=nma_mask,
            #go_terms=go_term_array,
            go_terms=np.ones(len(seqres)) * go_term,
            sites=site_array,
        )

        
        if 'alphafill' in entry and self.cfg.alphafill:
            path = entry['alphafill'][0].replace('.cif.gz', '.npz')
            _, name, _, _ = path.split('-')
            path = f"{self.cfg.alphafill}/{name[:2]}/{path}"
            try:
                struct = Structure.from_npz(path)
                chain = struct.get_chain(np.random.choice(range(1, len(struct.chains))))
            except:
                return data
            

            ones = np.ones(len(chain.atoms))
            ligand = self.make_data(
                name='',
                seqres='*'*len(chain.atoms),
                atom_num=chain.atoms['element'],
                mol_type=ones*chain.mol_type,
                seq_mask=ones,
                struct=chain.atoms['coords'],
                struct_mask=chain.atoms['is_present'],
                residx=chain.get_atom_residx(),
                chain=ones,
            )
            data = OpenProtData.concat([data, ligand])

        return data
    
    from typing import List, Tuple, Optional

def map_positions_to_subset(full_seq, subset_seq, positions, is_range):
    """
    Maps positions in the full sequence to positions in the subset sequence.
    Args:
        full_seq (str): full amino acid sequence (uniref)
        subset_seq (str): contiguous subset of the full sequence (afdb)
        positions (list): original positions (0-indexed)
        range (boolean): whether or not this is a position range
    Returns:
        labels remapped to the subset sequence positions (0-indexed) 
    """
    start_idx = full_seq.find(subset_seq)
    if start_idx == -1:
        # print("full_seq", full_seq, "subset_seq", subset_seq, flush=True)
        # raise ValueError("Subset sequence not found in full sequence.")
        return None # subset sequence is not found in the full sequence 
    
    end_idx = start_idx + len(subset_seq)
    
    if is_range:
        if len(positions) != 2:
            raise ValueError("When is_range=True, positions must have exactly two elements.")
        start, end = positions
        # create inclusive range
        position_list = list(range(start, end + 1))
    else:
        position_list = positions

    # map positions to the subset seq
    return [pos - start_idx for pos in position_list if start_idx <= pos < end_idx]
            

        
