#!/bin/bash
#SBATCH --job-name=int_GFO_t 
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node=1 
#SBATCH --mem=128G
#SBATCH --output=/nfs/scratch/choover/SimulationProject2/slurm_outputs/slurm-%j.out
#SBATCH --partition=hx

# Run computation
cd /nfs/scratch/choover/SimulationProject2/

pixi run python src/Simulation.py euler_config.toml


