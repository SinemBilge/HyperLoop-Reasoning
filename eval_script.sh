#!/bin/bash
#SBATCH --partition=A40short
#SBATCH --time=2:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --job-name=evaluation_test_euclidean_4loops
#SBATCH --output=slurm-%j.out

# Activate your Python environment
source /home/s56fasla/multi_hop_312/bin/activate

# Change to your project directory (use absolute path for safety)
cd /home/s56fasla/HyperLoop-Reasoning/
# Or: cd ~/HyperLoop-Reasoning/

# Run your Python script
python test_parse_then_hop_looped_euclidean.py --dataset 2wikimultihop --knit5_checkpoint_path knit5.pth --parsing_prompt_checkpoint_path runs/loops4_parsing_epoch_44_val_loss_0.2428_em_0.895857.pth --hopping_prompt_checkpoint_path runs/random_euclidean_4loops.pth/Jan22_03-18-58_AdaFactor_linear_LOOPED4_maxnorm0.9_bsize8_prompt_length100_lr0.8_curvature1.0_use_prompt_True/soft_prompt_epoch_20_val_loss_0.1956_em_0.310328.pth --num_loops 4 --batch_size 8