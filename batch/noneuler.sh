#!/bin/bash
#SBATCH --job-name=testslurm 
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1 
#SBATCH --mem=16G
#SBATCH --output=/nfs/scratch/choover/SimulationProject2/slurm_outputs/slurm-%j.out
#SBATCH --partition=hx

# Run computation
cd /nfs/scratch/choover/SimulationProject2/

pixi run python src/simulation.py sim_configs/config_noneuler.toml


