#!/bin/bash
#SBATCH --mem=40gb
#SBATCH -c4
#SBATCH --time=7:0:0
#SBATCH -o JobsOutput/Job_STDOUT_%j.txt  # send stdout to outfile
#SBATCH -e JobsOutput/Job_STDERR_%j.txt  # send stderr to errfile
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=mor.turgeman2@mail.huji.ac.il
#SBATCH --gres=gpu:1,vmem:21g

python3 T5_Trainer.py \
--model_name_or_path \
t5-base \
--text_column \
t5_sentence \
--target_column \
target \
--trained_on \
igg \
--split_type \
no_val \
--train_file \
../Data/humor_datasets/igg/no_val/train.csv \
--test_file \
../Data/humor_datasets/igg/no_val/test.csv \
--datasets_to_predict \
"amazon" "headlines" "igg" "twss" \
--output_dir \
../Model/SavedModels/T5 \
--do_train \
True \
--do_eval \
False \
--do_predict \
True \
--predict_with_generate \
False \
--source_prefix \
"is funny: " \
--report_to \
"wandb" \
--save_total_limit \
2 \
--max_train_samples \
10 \
--max_eval_samples \
10 \
--max_predict_samples \
10 \
#--validation_file \
#../Data/humor_datasets/igg/no_val/val.csv \

