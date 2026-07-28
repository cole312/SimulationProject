import numpy as np
import pandas as pd
from dipy.core.gradients import gradient_table
from dipy.reconst import dti

csv_file = "outputs/sphereRadius_5.0_signals_config_tester.csv"
df = pd.read_csv(csv_file, comment="#")

R_cols = ["R11", "R12", "R13", "R21", "R22", "R23", "R31", "R32", "R33"]
ref_dir = np.array([1, 0, 0])

for waveform_file, df_wave in df.groupby("file"):
    print(f"\nWaveform: {waveform_file}")

    bvals = df_wave["bval"].to_numpy()
    signals = df_wave["signal"].to_numpy()

    rot_matrices = df_wave[R_cols].to_numpy().reshape(-1, 3, 3)
    bvecs = rot_matrices @ ref_dir  

    gtab = gradient_table(bvals, bvecs)
    fit = dti.TensorModel(gtab).fit(signals)

    print(f"Fractional Anisotropy (FA): {fit.fa:.3f}")
    print(f"Mean Diffusivity (MD):     {fit.md:.3e} mm^2/s")
    print(f"Axial Diffusivity (AD):    {fit.ad:.3e} mm^2/s")
    print(f"Radial Diffusivity (RD):   {fit.rd:.3e} mm^2/s")