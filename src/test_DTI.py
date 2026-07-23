import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dipy.core.gradients import gradient_table
from dipy.reconst import dti

# 1. Load your simulation CSV output (update with your actual filename)
csv_file = "outputs/1_cylindersUP_signals_config_tester.csv"  
df = pd.read_csv(csv_file, comment="#")

R_cols = ["R11", "R12", "R13", "R21", "R22", "R23", "R31", "R32", "R33"]
ref_dir = np.array([0, 0, 1])  # Base direction of your waveform

unique_waveforms = df["file"].unique()
num_waveforms = len(unique_waveforms)

# Set up a multi-panel 3D figure layout
fig = plt.figure(figsize=(14, 8))

# Pre-calculate wireframe sphere coordinates for the 3D plots
u = np.linspace(0, 2 * np.pi, 30)
v = np.linspace(0, np.pi, 30)
xs = np.outer(np.cos(u), np.sin(v))
ys = np.outer(np.sin(u), np.sin(v))
zs = np.outer(np.ones(np.size(u)), np.cos(v))

for idx, waveform_file in enumerate(unique_waveforms):
    if idx < 3:
        print(f"\n========================================")
        print(f"Processing Waveform: {waveform_file}")
        print(f"========================================")
        
        # Filter rows belonging only to this specific waveform
        df_wave = df[df["file"] == waveform_file]
        
        bvals = df_wave["bval"].to_numpy()
        signals = df_wave["signal"].to_numpy()
        
        # Extract rotation matrices for this waveform subset
        rot_matrices = df_wave[R_cols].to_numpy().reshape(-1, 3, 3)
        bvecs = np.array([rot @ ref_dir for rot in rot_matrices])
        
        # Normalize bvecs safely
        bvecs = bvecs / np.linalg.norm(bvecs, axis=1, keepdims=True)
        
        # 2. Create the DIPY gradient table for this specific waveform
        gtab = gradient_table(bvals, bvecs=bvecs)
        
        # 3. Fit the DTI Tensor Model
        data = signals.reshape(1, 1, 1, -1) 
        mask = np.ones((1, 1, 1), dtype=bool)
        
        tensor_model = dti.TensorModel(gtab)
        tensor_fit = tensor_model.fit(data, mask=mask)
        
        # 4. Extract and print DTI Metrics
        fa = tensor_fit.fa[0, 0, 0]
        md = tensor_fit.md[0, 0, 0]
        ad = tensor_fit.ad[0, 0, 0]
        rd = tensor_fit.rd[0, 0, 0]
        
        print(f"Fractional Anisotropy (FA): {fa:.4f}")
        print(f"Mean Diffusivity (MD):     {md:.4f} mm^2/s")
        print(f"Axial Diffusivity (AD):    {ad:.4f} mm^2/s")
        print(f"Radial Diffusivity (RD):   {rd:.4f} mm^2/s")

        # 5. Plotting rotation vectors on a 3D subplot
        ax = fig.add_subplot(2, 3, idx + 1, projection='3d')
        ax.plot_wireframe(xs, ys, zs, color='gray', alpha=0.15)
        
        # Plot unique directions to avoid overplotting duplicates across b-shells
        unique_bvecs = np.unique(bvecs, axis=0)
        ax.scatter(unique_bvecs[:, 0], unique_bvecs[:, 1], unique_bvecs[:, 2], 
                color='dodgerblue', s=40, edgecolor='k', depthshade=True)
        
        # Subplot styling
        short_name = waveform_file.split('/')[-1]
        ax.set_title(f"{short_name}\nFA: {fa:.2f}", fontsize=11, fontweight='bold')
        ax.set_box_aspect([1, 1, 1])

plt.suptitle("Rotation Vectors & DTI Metrics Across Waveforms", fontsize=15, y=0.95)
plt.tight_layout()
plt.savefig("outputs/test")