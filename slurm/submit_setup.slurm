#!/bin/bash
#SBATCH --job-name=setup_llama33_80B
#SBATCH --partition=cpu          # Setup doesn't need a GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G                # Requesting 64GB to prevent the 'Killed' error
#SBATCH --time=03:00:00          # 3 hours for download + installs
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=fmahnoor1@umassd.edu

# Run your robust setup script
bash slurm/setup_unity.sh