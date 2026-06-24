# Introduction

This project focuses on a Monte Carlo simulation pipeline to examine signal decay in axons by utilizing CATERPillar generated substrates and the Disimpy simulation engine. All dependencies are managed using Pixi.

# Installation

Install Pixi if it is not already installed:
```bash
curl -fsSL [https://pixi.sh/install.sh](https://pixi.sh/install.sh) | sh
```
Next, enter the Pixi shell:
```bash
pixi shell
```
Note: Ensure you are in the root directory to run scripts for proper file pathing.

# Config
All simulation parameters are managed within a config file. To adjust the simulation, modify the parameters inside sim_configs/config.toml.

# Execution
All substrates are stored within the substrate/ folder. The config file fetches substrates from this folder.

CATERPillar Substrates: To add a new CATERPillar substrate to the folder, run substrateCATERPillar.py using your CATERPillar output file.

Other Mesh Types: For other types of meshes, use the substrate file (works with trimesh).

Ensure the name of the mesh matches the one in your config file, then run the simulation:

```Bash
python src/simulation.py
```
Signals are outputted directly to the outputs/ folder.
