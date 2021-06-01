#!/bin/bash
#SBATCH --job-name=regression0
#SBATCH -o out/regression0.out
#SBATCH --partition=gpu_7d1g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --mem=92000M
#SBATCH --time=96:00:00
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
echo job start time is `date`
echo `hostname`
singularity exec --nv /home/baikliang2/singularityBox/simpletransformer.sig python3 regressionSlurmFold_eval.py \
# the output path for models
--output_dir="out_models/finalversion05170" \
# The pretrained model path
--bert_model="chinese_roberta_wwm_ext_pytorch" \
# the output evaluation file
--eval_file='results/final_advice05170.csv' \
# the path of training data and test data
--finetune_data='data/final_regression_0517/fold0' \
# the path of candidate advices
--advice_data='data/final_regression_0517' \
--train_epochs=15 
echo job end time is `date` 
