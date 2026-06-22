#!/bin/bash

# Code
code_path='/home/ch278262/Desktop/FOMO2JOMO/src'
export PYTHONPATH="${code_path}:$PYTHONPATH"

# Data
#data_path='/media/jaume/T7'

# =========== Preprocessing self-supervised data ==========
#src_data=${data_path}/FOMO-MRI
#pretrain_data=${src_data}/fomo-60k_baseline_preprocess/FOMO60k

# For pre-training data
#python ${code_path}/data/fomo-60k/preprocess.py \
#--in_path=${src_data}/fomo-60k \
#--out_path=${src_data}/fomo-60k_baseline_preprocess2 \

# ========== Pretraining data preprocessing done ==========
# Task-specific preprocessing
#dst_path=${data_path}/finetuning_data_preprocess/mimic-pretreaining-preprocessing_2

# For fine-tuning in the challenge tasks
#src_path=${finetuning_data_preprocess}/finetuning_data/fomo-task1
#python ${code_path}/data/preprocess/run_preprocessing.py \
#        --taskid 1 \
#        --source_path ${src_path} \
#        --output_path ${dst_path} \
#        --num_workers 2

#src_path=${finetuning_data_preprocess}/finetuning_data/fomo-task2
#python ${code_path}/data/preprocess/run_preprocessing.py \
#        --taskid 2 \
#        --source_path ${src_path} \
#        --output_path ${dst_path} \
#       --num_workers 2

#src_path=${finetuning_data_preprocess}/finetuning_data/fomo-task3
#python ${code_path}/data/preprocess/run_preprocessing.py \
#        --taskid 3 \
#        --source_path ${src_path} \
#        --output_path ${dst_path} \
#        --num_workers 2

        # =========== ISLES2024 Preprocessing ==========
isles_src_path='/mnt/8tb/Youwan/ISLES2024'
dest_path=${isles_src_path}/preprocessing_fomo2jomo
python ${code_path}/data/preprocess/run_preprocessing.py \
        --taskid 5 \
        --source_path ${isles_src_path} \
        --output_path ${dest_path} \
        --num_workers 1