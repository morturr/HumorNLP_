#!/bin/bash
#SBATCH --mem=50gb
#SBATCH -c4
#SBATCH --time=1:0:0
#SBATCH --gres=gpu:2,vmem:12g

python3 -u data_loader.py







