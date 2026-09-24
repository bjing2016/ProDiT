from .task import OpenProtTask
import numpy as np
from ..utils import residue_constants as rc
from scipy.spatial.transform import Rotation as R
from .codesign import CodesignTask

class StructurePrediction(CodesignTask):

    def register_loss_masks(self):
        return ["/struct_pred"]

    def prep_data(self, data, crop=None, eps=1e-6):

        if crop is not None:
            data.crop(crop)

        noise_level = self.add_structure_noise(data)
        
        self.center_random_rot(data)

        # data["_cath_noise"] = np.ones((), dtype=np.float32)
        
        # go_terms_array = data['go_terms'] 
        # num_unique_terms = len(np.unique(go_terms_array[go_terms_array > 0]))
        # data["_go_noise"] = np.ones(num_unique_terms, dtype=np.float32)

        # site_array = data['sites'] 
        # num_unique_sites = len(np.unique(site_array[site_array > 0]))
        # data["_site_noise"] = np.ones(num_unique_sites, dtype=np.float32)
        
        data["/struct_pred"] = np.ones((), dtype=np.float32)
        return data
