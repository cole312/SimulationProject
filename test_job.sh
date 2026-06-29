#!/bin/bash
#SBATCH --job-name=testslurm 
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1 
#SBATCH --mem=16G
#SBATCH --output=slurm-%j.out

#SBATCH --partition=hx

# Run computation
cd /nfs/scratch/choover/SimulationProject/
pixi run python src/simulation.py sim_configs/test_config.toml


