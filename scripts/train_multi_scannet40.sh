#!/usr/bin/env bash

python main.py --dataset_mode=multi_obj \
               --scan_folder=/mnt/cloud_disk/lxp/data/ScanNet/scans \
               --train_list=/mnt/cloud_disk/lxp/data/ScanNet/train_list.json \
               --val_list=/mnt/cloud_disk/lxp/data/ScanNet/val_list.json \
               --lr=1e-4 \
               --epochs=1100 \
               --lr_drop=[1000] \
               --job_name=train_multi_obj_scannet40

python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=12345 main.py --dataset_mode=multi_obj \
               --scan_folder=/mnt/cloud_disk/lxp/data/ScanNet/scans \
               --train_list=/mnt/cloud_disk/lxp/data/ScanNet/train_list.json \
               --val_list=/mnt/cloud_disk/lxp/data/ScanNet/val_list.json \
               --lr=1e-4 \
               --epochs=1100 \
               --job_name=train_multi_obj_scannet40

python main.py --dataset_mode=multi_obj \
               --scan_folder=/mnt/cloud_disk/lxp/data/ScanNet/scans \
               --train_list=/mnt/cloud_disk/lxp/data/ScanNet/train_list.json \
               --val_list=/mnt/cloud_disk/lxp/data/ScanNet/val_list.json \
               --lr=1e-4 \
               --epochs=1100 \
               --job_name=train_multi_obj_scannet40
