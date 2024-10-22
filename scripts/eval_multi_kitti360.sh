#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 python eval_multi_obj.py --dataset_mode=multi_obj \
               --scan_folder=/mnt/cloud_disk/lxp/data/KITTI360/scans \
               --val_list=/mnt/cloud_disk/lxp/data/KITTI360/val_list.json \
               --output_dir=results/KITTI360_multi \
               --checkpoint=weights/checkpoint1099.pth