import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "signals",
    type=str,

)
args = parser.parse_args()

signal_file = args.signals
graph_title = os.path.basename(signal_file)

os.makedirs(f'graphOutputs/{graph_title}',exist_ok=True)
output = f"graphOutputs/{graph_title}"

df = pd.read_csv(f"{signal_file}", comment="#")

df_averaged = df.groupby(['waveform_idx', 'bval'])['signal'].mean()

unique_waveforms = sorted(df['waveform_idx'].unique())

md = []
kurt = []
curvesA = []
curvesB = []
curvesC = []

fig, ax = plt.subplots(figsize=(9, 5))
x_fit = np.linspace(0, max(df['bval']), 100)


for wf in unique_waveforms:
    wf_data = df_averaged.loc[wf]

    b_values = wf_data.index.tolist()
    signals = wf_data.values.tolist()

    b_arr = np.array(b_values)
    signals_arr = np.array(signals)


    mask = b_arr <= 10000
    b_mask = b_arr[mask]
    signals_mask = signals_arr[mask]

    A, B, C = np.polyfit(b_mask/1000, np.log(signals_mask), 2)
    if abs(A) < 1e-9:
        A = 0
    D = -B

    V = []
    V.append(A*2)

    # v is a*2
    #plot v and k with f
    kurt.append((6 * A) / (D**2))
    md.append(D)

    curvesA.append(A)
    curvesB.append(B)
    curvesC.append(C)

    raw_path = df[df['waveform_idx'] == wf]['file'].iloc[0]
    clean_filename = os.path.basename(raw_path) 
    label_name = clean_filename.replace(".csv", "") 

    y_fit = np.exp(A * (x_fit**2) + B * x_fit + C)

    ax.scatter(b_values, signals, marker='o', label=f"{label_name} Data")
    ax.plot(x_fit, y_fit, linestyle='--', label=f"Fit (MD: {md[-1]:.4e}, K: {kurt[-1]:.4f}, V: {V[-1]})")

ax.set_yscale('log')
ax.set_xlabel("b-value ($s/mm^2$)")
ax.set_ylabel("Normalized Signal ($S/S_0$)")
ax.set_title(f"Signal Decay")
ax.grid(True, which="both", linestyle='--', alpha=0.5)

ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))

plt.subplots_adjust(right=0.7)

plt.savefig(f"{output}/{graph_title}.png", dpi=300, bbox_inches='tight')

fig, ax = plt.subplots(figsize=(8, 5))

freq = np.array([0, 50, 100])
diff = np.array([md[0], md[1], md[2]])
kurt = np.array([kurt[0], kurt[1], kurt[2]])
variance = np.array([V[0],V[1],V[2]])
print([md[0], md[1], md[2]])

paramList = [("Diffusivity",diff), ("Kurtosis", kurt), ("Variance",variance)]

for name, param in paramList:
    ax.scatter(param, diff, color='red', marker='o', s=10, zorder=5, label='MD Data Points')

    x1 = np.linspace(0, 200, 100)

    lin_params = np.polyfit(param, diff, 1)
    lin_fit_vals = lin_params[0] * param + lin_params[1]
    lin_error = np.sum((diff - lin_fit_vals) ** 2)
    y_linear = lin_params[0] * x1 + lin_params[1]

    sqrt_params = np.polyfit(np.sqrt(param), diff, 1)
    sqrt_fit_vals = sqrt_params[0] * np.sqrt(param) + sqrt_params[1]
    sqrt_error = np.sum((diff - sqrt_fit_vals) ** 2)
    y_sqrt = sqrt_params[0] * np.sqrt(x1) + sqrt_params[1]

    sq_params = np.polyfit(param**2, diff, 1)
    sq_fit_vals = sq_params[0] * param**2 + sq_params[1]
    sq_error = np.sum((diff - sq_fit_vals) ** 2)
    y_squared = sq_params[0] * (x1**2) + sq_params[1]

    errors = {"Square Root": sqrt_error, "Linear": lin_error, "Squared": sq_error}
    best_fit_name = min(errors, key=errors.get)
    print(f"Best Fit: {best_fit_name}")

    ls_squared = '-' if best_fit_name == "Squared" else ':'
    ls_linear  = '-' if best_fit_name == "Linear" else ':'
    ls_sqrt    = '-' if best_fit_name == "Square Root" else ':'

    ax.plot(x1, y_squared, color='#004949', linestyle=ls_squared, linewidth=2, 
        label=f'Squared Fit {"(Best)" if best_fit_name == "Squared" else ""}')

    ax.plot(x1, y_linear, color='#FF6B6B', linestyle=ls_linear, linewidth=2, 
        label=f'Linear Fit  {"(Best)" if best_fit_name == "Linear" else ""}')

    ax.plot(x1, y_sqrt, color='#009999', linestyle=ls_sqrt, linewidth=2, 
        label=f'Square Root Fit {"(Best)" if best_fit_name == "Square Root" else ""}')

    ax.set_ylabel("Mean Diffusivity (MD)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_title(f"Frequency Dependence")
    ax.grid(True, which="both", linestyle='--', alpha=0.5)
    ax.legend(loc='best')

    plt.tight_layout()
    plt.savefig(f"{output}/{graph_title}_{name}.png", dpi=300)

try:
    traj_file = f"outputs/traj_{graph_title}"

    print(traj_file)

    data = np.loadtxt(traj_file)

    # data shape = (time steps, walkers*3)
    n_steps = data.shape[0]
    n_walkers = data.shape[1] // 3

    # reshape into (walker, timestep, xyz)
    traj = data.reshape(n_steps, n_walkers, 3).transpose(1, 0, 2)

    print("Trajectory shape:", traj.shape)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(10): 
        ax.plot(traj[i, :, 0],traj[i, :, 1],traj[i, :, 2],alpha=0.7,linewidth=0.3)

        # ax.scatter(traj[i,:,0],traj[i,:,1],traj[i,:,2],s=1)

    ax.view_init(elev=0, azim=0)
    ax.set_title("Disimpy Trajectories")
    ax.set_ylabel("Y")
    ax.set_xlabel("X")
    ax.set_zlabel("Z")

    ax.set_aspect('equal')

    plt.savefig(f"{output}/{graph_title}_traj.png", dpi=300)
    plt.close()

except:
    print("no traj file found")
