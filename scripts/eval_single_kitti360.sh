#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=3 python eval_single_obj.py --dataset=kitti360 \
               --dataset_mode=single_obj \
               --scan_folder=/mnt/cloud_disk/lxp/data/KITTI360/single/crops \
               --crop \
               --val_list=/mnt/cloud_disk/lxp/data/KITTI360/single/object_ids.npy \
               --val_list_classes=/mnt/cloud_disk/lxp/data/KITTI360/single/object_classes.txt \
               --output_dir=results/KITTI_single \
               --checkpoint=/mnt/cloud_disk/lxp/AGILE3D-main/output/2024-10-14-12-36-48/checkpoint1079.pth