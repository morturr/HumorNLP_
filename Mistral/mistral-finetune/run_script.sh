#!/bin/bash
#SBATCH --mem=50gb
#SBATCH -c4
#SBATCH --time=15:0:0
#SBATCH -o ../JobsOutput/Job_STDOUT_%j.txt  # send stdout to outfile
#SBATCH -e ../JobsOutput/Job_STDERR_%j.txt  # send stderr to errfile
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=mor.turgeman2@mail.huji.ac.il
#SBATCH --gres=gpu:4,vmem:42g

torchrun --nproc-per-node 4 --master_port=61111 -m train example/7B.yaml
#python3 -u -m utils.validate_data --train_yaml example/7B.yaml
