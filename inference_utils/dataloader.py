import os
import numpy as np
import MinkowskiEngine as ME
from utils.ply import read_ply

class InferenceDataLoader:
    def __init__(self, config):

        self.scan_folder = config.scan_folder
        self.sample_name = config.sample_name
        self.quantization_size = 0.05

    def load_scene(self):
        scene_name, num_obj = self.sample_name.split('_obj_')
        num_obj = int(num_obj)
        
        point_cloud = read_ply(os.path.join(self.scan_folder, scene_name + '.ply'))
        
        coords_full = np.column_stack([point_cloud['x'], point_cloud['y'], point_cloud['z']]).astype(np.float64)
        colors_full = np.column_stack([point_cloud['R'], point_cloud['G'], point_cloud['B']])/255
        labels_full = point_cloud['label'].astype(np.int32)
        
        labels_full_new = labels_full

        coords_qv, unique_map, inverse_map = ME.utils.sparse_quantize(
            coordinates=coords_full,
            quantization_size=self.quantization_size,
            return_index=True,
            return_inverse=True)
        
        raw_coords_qv = coords_full[unique_map]
        feats_qv = colors_full[unique_map]
        labels_qv = labels_full_new[unique_map]

        click_idx_qv = {}

        return coords_qv, raw_coords_qv, feats_qv, labels_qv, labels_full_new, inverse_map, click_idx_qv, scene_name, num_obj