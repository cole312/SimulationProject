# Monte Carlo Diffusion Simulation Pipeline

The project uses:

- [Disimpy](https://github.com/kerkelae/disimpy) for Monte Carlo diffusion simulations
- [DIPY](https://dipy.org/) for DKI analysis
- CATERPillar-generated substrates
- Trimesh meshes
- Pixi for environment management
- SLURM for large scale simulations

## Project Structure

```text
.
├── batch/              # SLURM batch scripts
├── graphOutputs/       # Plots and results
├── outputs/            # Raw simulation outputs
├── rotations/         # Rotation matrices
├── sim_configs/       # Simulation configuration files
├── src/
│   ├── simulation.py   # Monte Carlo simulation
│   └── graphing.py     # Signal and DKI analysis
├── substrate/          # Simulation substrates
│   ├── substrate_faces.csv
│   ├── substrate_vertices.csv
│   └── substrate.png   # PNG of substrate
├── pixi.toml           # Pixi environment
└── README.md
```

## Installation

Install [Pixi](https://pixi.sh/) if needed:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

Enter the project environment:

```bash
pixi shell
```

Run commands from the project root directory so that relative paths resolve correctly.

## Substrates

Substrates are stored in:

```text
substrate/
```

Each substrate contains the required vertices and face CSV files and a PNG visualization:

CATERPillar substrates must be converted to this vertex/face CSV format before use.

Other meshes can be used as long as they can be converted to the same format.

## Simulation Configuration

Simulation parameters are stored in TOML files in:

```text
sim_configs/
```

Multiple configuration files can be used for different substrates and simulation parameters.

## Running a Simulation

### Local

Run a configuration with:

```bash
python src/simulation.py sim_configs/<config>.toml
```

### SLURM

Large simulations can be submitted using the scripts in `batch/`:

```bash
sbatch batch/<batch_script>.sh
```

For very large simulations, increase the memory requested in the batch script as necessary.

## Simulation Outputs

Simulation results are written to:

```text
outputs/
```

Signal files contain the waveform, gradient rotation, b-value, normalized signal, and simulation metadata including the Monte Carlo seed.

A separate trajectory simulation is also generated using 10 walkers for visualization.

## Signal & DKI Analysis

Analyze a generated signal file with:

```bash
python src/graphing.py outputs/<signal_file>.csv
```

The analysis performs:

- 2nd order singal decay fitting
- Diffusivity, kurtosis, and variance calculation
- Frequency dependence fitting
- DKI fitting for FA, MD, AD, and RD
- Walker trajectory visualization when traj file available

Three frequency-dependence models are compared:

- Linear
- Square root
- Squared

The model with the lowest least-squares error is reported as the best fit.

## Analysis Outputs

Results are saved under:

```text
graphOutputs/<signal_file>/
```

`results.csv` contains the calculated FA, MD, AD, RD, kurtosis, and variance values for the simulated waveforms.

## Typical Workflow

```bash
# Enter environment
pixi shell

# Run simulation
python src/simulation.py sim_configs/<config>.toml

# Analyze signal
python src/graphing.py outputs/<signal_file>.csv
```

For cluster simulations:

```bash
sbatch batch/<batch_script>.sh
```

