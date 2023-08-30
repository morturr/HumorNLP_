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
"amazon_headlines" \
--compute_on \
"amazon" \
--split_type \
with_val_fixed_train \
--train_file \
../Data/humor_datasets/igg/with_val_fixed_train/train.csv \
--test_file \
../Data/humor_datasets/igg/with_val_fixed_train/test.csv \
--validation_file \
../Data/humor_datasets/igg/with_val_fixed_train/val.csv \
--train_path_template \
../Data/humor_datasets/paired_datasets/{dataset}/{split_type}/{split_name}.csv \
--test_path_template \
../Data/humor_datasets/{dataset}/{split_type}/{split_name}.csv \
--datasets_to_predict \
"amazon" "headlines" "igg" "twss" \
--epochs \
3 \
--batch_sizes \
8 \
--learning_rates \
5e-5 1e-6 \
--seeds \
5 18 27 42 \
--output_dir \
../Model/SavedModels/T5 \
--do_train \
True \
--do_eval \
True \
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
--save_model \
False \
--save_metrics \
False \
--save_state \
False


