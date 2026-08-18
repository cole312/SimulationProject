import numpy as np
import pandas as pd
from dipy.core.gradients import gradient_table
from dipy.reconst import dki
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "signals",
    type=str,

)
args = parser.parse_args()

csv_file = args.signals
df = pd.read_csv(csv_file, comment="#")

R_cols = ["R11", "R12", "R13", "R21", "R22", "R23", "R31", "R32", "R33"]
ref_dir = np.array([1, 0, 0])

count = 0
md = []
ad = []
rd = []
for waveform_file, df_wave in df.groupby("file", sort=False):
    if count <= 2:
        print(f"\nWaveform: {waveform_file}")

        bvals = df_wave["bval"].to_numpy()
        signals = df_wave["signal"].to_numpy()

        rot_matrices = df_wave[R_cols].to_numpy().reshape(-1, 3, 3)
        bvecs = rot_matrices[:,0,:]

        gtab = gradient_table(bvals=bvals, bvecs=bvecs)
        fit = dki.DiffusionKurtosisModel(gtab).fit(signals)

        print(f"Fractional Anisotropy (FA): {fit.fa:.3f}")
        print(f"Mean Diffusivity (MD):     {fit.md:.3e} mm^2/s")
        md.append(fit.md)
        print(f"Axial Diffusivity (AD):    {fit.ad:.3e} mm^2/s")
        ad.append(fit.ad)
        print(f"Radial Diffusivity (RD):   {fit.rd:.3e} mm^2/s")
        rd.append(fit.rd)
        count += 1


freq = np.array([0, 50, 100])
x1 = np.linspace(0, 200, 100)

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

metrics = [
    ("Mean Diffusivity (MD)", np.array(md), axes[0]),
    ("Axial Diffusivity (AD)", np.array(ad), axes[1]),
    ("Radial Diffusivity (RD)", np.array(rd), axes[2]),
]

for metric_name, diff, ax in metrics:

    lin_params = np.polyfit(freq, diff, 1)
    lin_fit_vals = lin_params[0] * freq + lin_params[1]
    lin_error = np.sum((diff - lin_fit_vals) ** 2)
    y_linear = lin_params[0] * x1 + lin_params[1]

    sqrt_params = np.polyfit(np.sqrt(freq), diff, 1)
    sqrt_fit_vals = sqrt_params[0] * np.sqrt(freq) + sqrt_params[1]
    sqrt_error = np.sum((diff - sqrt_fit_vals) ** 2)
    y_sqrt = sqrt_params[0] * np.sqrt(x1) + sqrt_params[1]

    sq_params = np.polyfit(freq**2, diff, 1)
    sq_fit_vals = sq_params[0] * (freq**2) + sq_params[1]
    sq_error = np.sum((diff - sq_fit_vals) ** 2)
    y_squared = sq_params[0] * (x1**2) + sq_params[1]

    errors = {
        "Square Root": sqrt_error,
        "Linear": lin_error,
        "Squared": sq_error,
    }
    best_fit_name = min(errors, key=errors.get)

    ls_squared = "-" if best_fit_name == "Squared" else ":"
    ls_linear = "-" if best_fit_name == "Linear" else ":"
    ls_sqrt = "-" if best_fit_name == "Square Root" else ":"

    ax.scatter(
        freq,
        diff,
        color="red",
        marker="o",
        s=30,
        zorder=5,
        label="Data Points",
    )
    ax.plot(
        x1,
        y_squared,
        color="#004949",
        linestyle=ls_squared,
        linewidth=2,
        label=f'Squared {"(Best)" if best_fit_name == "Squared" else ""}',
    )

    ax.plot(
        x1,
        y_linear,
        color="#FF6B6B",
        linestyle=ls_linear,
        linewidth=2,
        label=f'Linear {"(Best)" if best_fit_name == "Linear" else ""}',
    )

    ax.plot(
        x1,
        y_sqrt,
        color="#009999",
        linestyle=ls_sqrt,
        linewidth=2,
        label=f'Square Root {"(Best)" if best_fit_name == "Square Root" else ""}',
    )

    if best_fit_name == "Linear":
        eq = f"y = {lin_params[0]:.3e}x {lin_params[1]:+.3e}"
    elif best_fit_name == "Square Root":
        eq = f"y = {sqrt_params[0]:.3e}x^1/2 {sqrt_params[1]:+.3e}"
    else:
        eq = f"y = {sq_params[0]:.3e}x^2 {sq_params[1]:+.3e}"

    eq_text = (
        f"Best Fit: {best_fit_name}\n"
        f"{eq}\n\n"
        f"0 Hz   = {diff[0]:.3e}\n"
        f"50 Hz  = {diff[1]:.3e}\n"
        f"100 Hz = {diff[2]:.3e}"
    )

    ax.text(
        0.02,
        0.98,
        eq_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="black"),
    )

    
    ax.set_ylim(min(ad)*0.5,max(ad)*1.50) if metric_name == "Axial Diffusivity (AD)" else None
    ax.set_ylabel(f"{metric_name} ($\\mathrm{{mm^2/s}}$)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_title(f"{metric_name} vs Frequency")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)

plt.tight_layout()

output_path = f"graphOutputs/{Path(csv_file).name}/MD_AD_RD.png"
plt.savefig(output_path, dpi=300)\


print()