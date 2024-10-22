#!/usr/bin/env bash

python eval_single_obj.py --dataset=s3dis \
               --dataset_mode=single_obj \
               --scan_folder=data/S3DIS/single/crops \
               --crop \
               --val_list=data/S3DIS/single/object_ids.npy \
               --val_list_classes=data/S3DIS/single/object_classes.txt \
               --output_dir=results/S3DIS_single \
               --checkpoint=weights/checkpoint1099.pth

CUDA_VISIBLE_DEVICES=4 python eval_single_obj.py --dataset=s3dis \
               --dataset_mode=single_obj \
               --scan_folder=/mnt/cloud_disk/lxp/data/S3DIS/single/crops \
               --crop \
               --val_list=/mnt/cloud_disk/lxp/data/S3DIS/single/object_ids.npy \
               --val_list_classes=/mnt/cloud_disk/lxp/data/S3DIS/single/object_classes.txt \
               --output_dir=results/S3DIS_single \
               --checkpoint=/mnt/cloud_disk/lxp/AGILE3D-main/output/2024-10-14-12-36-48/checkpoint1079.pth