from .task import OpenProtTask
import numpy as np
from ..utils import residue_constants as rc
from scipy.spatial.transform import Rotation as R
from .codesign import CodesignTask

class InverseFolding(CodesignTask):

    def register_loss_masks(self):
        return ["/inv_fold"]
        
    def prep_data(self, data, crop=None, eps=1e-6):

        if crop is not None:
            data.crop(crop)

        # if np.random.rand() < self.cfg.ppi_prob:
        #     is_ppi = self.sample_ppi(data)
        #     if is_ppi:
        #         data["/inv_fold/ppi"] = np.ones((), dtype=np.float32)
                
        self.add_sequence_noise(data)
        self.add_structure_noise(data, noise_level=0)

        # data["_cath_noise"] = np.ones((), dtype=np.float32)
        
        # go_terms_array = data['go_terms'] 
        # num_unique_terms = len(np.unique(go_terms_array[go_terms_array > 0]))
        # data["_go_noise"] = np.ones(num_unique_terms, dtype=np.float32)

        # site_array = data['sites'] 
        # num_unique_sites = len(np.unique(site_array[site_array > 0]))
        # data["_site_noise"] = np.ones(num_unique_sites, dtype=np.float32)

        self.center_random_rot(data)
        
        data["/inv_fold"] = np.ones((), dtype=np.float32)
        return data