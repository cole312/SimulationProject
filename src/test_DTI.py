import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dipy.core.gradients import gradient_table
from dipy.reconst import dti

csv_file = ""  
df = pd.read_csv(csv_file, comment="#")

R_cols = ["R11", "R12", "R13", "R21", "R22", "R23", "R31", "R32", "R33"]
ref_dir = np.array([0, 0, 1])

unique_waveforms = df["file"].unique()
num_waveforms = len(unique_waveforms)
fig = plt.figure(figsize=(14, 8))

for idx, waveform_file in enumerate(unique_waveforms):
    if idx < 3:
        print(f"waveform: {waveform_file}")
      
        df_wave = df[df["file"] == waveform_file]
        
        bvals = df_wave["bval"].to_numpy()
        signals = df_wave["signal"].to_numpy()

        rot_matrices = df_wave[R_cols].to_numpy().reshape(-1, 3, 3)
        bvecs = np.array([rot @ ref_dir for rot in rot_matrices])
        
        bvecs = bvecs / np.linalg.norm(bvecs, axis=1, keepdims=True)
        
        gtab = gradient_table(bvals, bvecs=bvecs)
        data = signals.reshape(1, 1, 1, -1) 
        mask = np.ones((1, 1, 1), dtype=bool)
        
        tensor_model = dti.TensorModel(gtab)
        tensor_fit = tensor_model.fit(data, mask=mask)
        
        fa = tensor_fit.fa[0, 0, 0]
        md = tensor_fit.md[0, 0, 0]
        ad = tensor_fit.ad[0, 0, 0]
        rd = tensor_fit.rd[0, 0, 0]
        
        print(f"Fractional Anisotropy (FA): {fa:.3e}")
        print(f"Mean Diffusivity (MD):     {md:.3e} mm^2/s")
        print(f"Axial Diffusivity (AD):    {ad:.3e} mm^2/s")
        print(f"Radial Diffusivity (RD):   {rd:.3e} mm^2/s")
