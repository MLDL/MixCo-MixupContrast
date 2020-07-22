#!/bin/bash
echo “[moco] resnet50 model on TinyImageNet dataset”

name="resnet50_bnstat_withshuffleBN_tinyimg"

CUDA_VISIBLE_DEVICES=2,3 python main_moco.py "$name" \
                                             '../data/tiny-imagenet-200' \
                                                --arch 'resnet50_bnstat' \
                                                --seed '0' \
                                                --batch-size 64 \
                                                --learning-rate 7.5e-4 \
                                                --bn-shuffle true \
                                                --bn-stat true \
                                                --dist-url 'tcp://localhost:10001' \
                                                --workers 8 \
                                                --multiprocessing-distributed \
                                                --world-size 1 \
                                                --rank 0 \
                                                --mlp \
                                                --moco-t 0.2 \
                                                --aug-plus \
                                                --cos
                                                
echo “[lincls] resnet50 model on TinyImageNet dataset”

CUDA_VISIBLE_DEVICES=2,3 python main_lincls.py "$name" \
                                               '../data/tiny-imagenet-200' \
                                                --arch 'resnet50_bnstat' \
                                                --seed '0' \
                                                --batch-size 64 \
                                                --learning-rate 7.5 \
                                                --pretrained "results/$name/moco/checkpoint_0199.pth.tar" \
                                                --dist-url 'tcp://localhost:10001' \
                                                --workers 8 \
                                                --multiprocessing-distributed \
                                                --world-size 1 \
                                                --rank 0
