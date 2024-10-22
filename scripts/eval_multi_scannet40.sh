#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python eval_multi_obj.py --dataset_mode=multi_obj \
               --scan_folder=/mnt/cloud_disk/lxp/data/ScanNet/scans \
               --val_list=/mnt/cloud_disk/lxp/data/ScanNet/val_list.json \
               --output_dir=results/ScanNet_multi \
               --checkpoint=/mnt/cloud_disk/lxp/AGILE3D-main/output/2024-10-14-12-36-48/checkpoint1079.pth