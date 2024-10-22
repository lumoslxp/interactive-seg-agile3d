import argparse
from inference_utils.dataloader import InferenceDataLoader
from inference_utils.inference import InferenceInteractiveSegmentation

import torch

def main(_):
    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    dataloader = InferenceDataLoader(config)
    infer_model_class = InferenceInteractiveSegmentation(device, config, dataloader)
    print(f"Using {device}")
    infer_model_class.run_segmentation()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # minimal arguments:
    parser.add_argument('--scan_folder', type=str,
                        default='/mnt/cloud_disk/lxp/data/ScanNet/scans')
    parser.add_argument('--sample_name', type=str,
                        default='scene0418_01_obj_29')
    parser.add_argument('--pretraining_weights', type=str,
                        default='/mnt/cloud_disk/lxp/AGILE3D-main/output/2024-10-14-12-36-48/checkpoint1079.pth')
    parser.add_argument('--max_num_clicks', default=20, help='maximum number of clicks per object on average', type=int)
    parser.add_argument('--device', default='cuda')
    
    # model
    ### 1. backbone
    parser.add_argument('--dialations', default=[ 1, 1, 1, 1 ], type=list)
    parser.add_argument('--conv1_kernel_size', default=5, type=int)
    parser.add_argument('--bn_momentum', default=0.02, type=int)
    parser.add_argument('--voxel_size', default=0.05, type=float)
    
    ### 2. transformer
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--dim_feedforward', default=1024, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--num_decoders', default=3, type=int)
    parser.add_argument('--num_bg_queries', default=10, type=int, help='number of learnable background queries')
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--pre_norm', default=False, type=bool)
    parser.add_argument('--normalize_pos_enc', default=True, type=bool)
    parser.add_argument('--positional_encoding_type', default="fourier", type=str)
    parser.add_argument('--gauss_scale', default=1.0, type=float, help='gauss scale for positional encoding')
    parser.add_argument('--hlevels', default=[4], type=list)
    parser.add_argument('--shared_decoder', default=False, type=bool)
    parser.add_argument('--aux', default=True, type=bool, help='whether supervise layer by layer')
    
    config = parser.parse_args()

    main(config)
