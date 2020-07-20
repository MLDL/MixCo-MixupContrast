#!/bin/bash
echo “[moco] resnet50 model on TinyImageNet dataset”

name="resnet50_woShuffleBN_tinyimg"

CUDA_VISIBLE_DEVICES=2,3 python main_moco.py "$name" \
                                             'data/TinyImageNet' \
                                                --arch 'resnet50' \
                                                --seed '0' \
                                                --epochs 1 \
                                                --batch-size 64 \
                                                --learning-rate 7.5e-4 \
                                                --bn-shuffle false \
                                                --dist-url 'tcp://localhost:10001' \
                                                --multiprocessing-distributed \
                                                --world-size 1 \
                                                --rank 0 \
                                                --mlp \
                                                --moco-t 0.2 \
                                                --aug-plus \
                                                --cos
                                                
#echo “[lincls] resnet50 model on TinyImageNet dataset”

#python3 main_lincls.py "$name" \
#                     'data/TinyImageNet' \
#                         --arch 'resnet50' \
#                         --seed '0' \
#                         --epochs 1 \
#                         --pretrained "results/$name/moco/checkpoint_0000.pth.tar"