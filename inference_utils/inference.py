from collections import OrderedDict
import copy
import re
import numpy as np
import MinkowskiEngine as ME
import torch
import open3d as o3d

from utils.seg import mean_iou_scene, extend_clicks, get_simulated_clicks
from models import build_model

obj_color = {0: [211, 211, 211], 1: [1, 211, 211], 2: [233,138,0], 3: [41,207,2], 4: [244, 0, 128], 5: [194, 193, 3], 6: [121, 59, 50],
             7: [254, 180, 214], 8: [239, 1, 51], 9: [125, 0, 237], 10: [229, 14, 241]}

class InferenceInteractiveSegmentation():
    def __init__(self, device, config, dataloader):
        self.config = config

        # load model
        self.pretrained_weights_file = config.pretraining_weights

        self.model = build_model(config)
        self.model.to(device)
        self.model.eval()
        self.device = device
        if self.pretrained_weights_file:
            weights = self.pretrained_weights_file
            if not torch.cuda.is_available():
                map_location = 'cpu'
                print('Cuda not found, using CPU')
                checkpoint = torch.load(weights, map_location)
                model_dict = OrderedDict()
                pattern = re.compile('module.')
                for k,v in checkpoint['model'].items():
                    if re.search("module", k):
                        model_dict[re.sub(pattern, '', k)] = v
                    else:
                        model_dict = checkpoint['model']
            else:
                map_location = None
                checkpoint = torch.load(weights, map_location)
                model_dict = OrderedDict()
                pattern = re.compile('module.')
                for k,v in checkpoint['model'].items():
                    if re.search("module", k):
                        model_dict[re.sub(pattern, '', k)] = v
                    else:
                        model_dict = checkpoint['model']
            
            missing_keys, unexpected_keys = self.model.load_state_dict(model_dict, strict=False)
            unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
            if len(missing_keys) > 0:
                print('Missing Keys: {}'.format(missing_keys))
            if len(unexpected_keys) > 0:
                print('Unexpected Keys: {}'.format(unexpected_keys))
        
            self.model.eval()
            # load inference dataset
            self.dataloader = dataloader
            self.device = device
            

    def run_segmentation(self,):

        coords, raw_coords, feats, labels, labels_full, inverse_map, click_idx, scene_name, num_obj=  self.dataloader.load_scene()
        coords = coords.float().to(self.device)
        feats = torch.from_numpy(feats).float().to(self.device)
        raw_coords = torch.from_numpy(raw_coords).float().to(self.device)
        labels = torch.from_numpy(labels).float().to(self.device)
        labels_full = torch.from_numpy(labels_full).float().to(self.device)

        # labels_full = labels_full.to(self.device)

        data = ME.SparseTensor(
                            coordinates= ME.utils.batched_coordinates([coords]),
                            features=feats,
                            device=self.device
                            )
        
        pcd_features, aux, coordinates, pos_encodings_pcd = self.model.forward_backbone(data, raw_coordinates=raw_coords)

        idx = 0
        valid_obj_idxs = torch.unique(labels)
        valid_obj_idxs = valid_obj_idxs[valid_obj_idxs!=-1]
        max_num_obj = len(valid_obj_idxs)
        num_obj = 10
        obj_idxs = valid_obj_idxs[torch.randperm(max_num_obj)[:num_obj]]
        obj_idxs = torch.tensor([26.,  9.,  3., 25.,  6.,  1., 19., 29., 17., 14.], device='cuda:0')
        labels_new = torch.zeros(labels.shape[0], device=self.device)
        click_idx.setdefault(idx, {}) 
        for i, obj_id in enumerate(obj_idxs):
                obj_mask = labels == obj_id
                labels_new[obj_mask] = i+1

                click_idx[idx][str(i+1)] = []

        click_idx[idx]['0'] = [] 
        click_time_idx = copy.deepcopy(click_idx)

        current_num_clicks = 0

        max_num_clicks = num_obj *  self.config.max_num_clicks
        while current_num_clicks <= max_num_clicks:
            if current_num_clicks == 0:
                pred = torch.zeros(labels.shape).to(self.device) 
            else:

                outputs = self.model.forward_mask(pcd_features, aux, coordinates, pos_encodings_pcd,
                                             click_idx=click_idx, click_time_idx=click_time_idx)
                pred_logits = outputs['pred_masks']
                pred = pred_logits[0].argmax(-1)

            if current_num_clicks != 0:
                # update prediction with sparse gt
                for obj_id, cids in click_idx[idx].items():
                    pred[cids] = int(obj_id)
                
            pred_full = pred[inverse_map]
            labels_new_full = labels_new[inverse_map]
            raw_coords_full = raw_coords[inverse_map]

            if current_num_clicks == max_num_clicks:
                self.save_segmention_result(raw_coords_full, labels_new_full, num_obj)

            sample_iou, _ = mean_iou_scene(pred_full, labels_new_full)
            print(scene_name, 'Object: ', num_obj, 'num clicks: ', current_num_clicks/num_obj, 'IOU: ', sample_iou.item())
            
            new_clicks, new_clicks_num, new_click_pos, new_click_time = get_simulated_clicks(pred, labels_new, raw_coords, current_num_clicks, training=False)
             ### add new clicks ###
            if new_clicks is not None:
                click_idx[idx], click_time_idx[idx] = extend_clicks(click_idx[idx], click_time_idx[idx], new_clicks, new_click_time)
            
            if current_num_clicks == 0:
                new_clicks_num = num_obj
            else:
                new_clicks_num = 1
            current_num_clicks += new_clicks_num


    def save_segmention_result(self,raw_coords_full, labels_new_full, num_obj):
        labels_new_full = labels_new_full.cpu().numpy()
        raw_coords_full = raw_coords_full.cpu().numpy()
        colors = np.zeros(raw_coords_full.shape)
        for obj_id in range(num_obj+1):
            obj_mask = labels_new_full == obj_id
            colors[obj_mask,:] = self._get_obj_color(obj_id, normalize=True)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(raw_coords_full)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # 保存为 .ply 文件
        o3d.io.write_point_cloud("output.ply", pcd)
        print("点云已保存为 output.ply 文件")
    
    def _get_obj_color(self,obj_idx, normalize=False):

        r, g, b = obj_color[int(obj_idx)]

        if normalize:
            r /= 256
            g /= 256
            b /= 256

        return np.array([r, g, b])