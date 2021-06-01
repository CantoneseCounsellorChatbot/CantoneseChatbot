#!/bin/bash


for i in 0 1 2 3 4
do
    singularity exec --nv /home/kent/newhome/singularityBox/GlobalEncoding.sif python3 preprocess.py \
    -load_data GE_combination/fold$i/ \
    -save_data data/GE_combination/fold${i}/ \
    -src_char -tgt_char

done