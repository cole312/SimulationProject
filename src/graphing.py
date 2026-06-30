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

config_file_path = args.signals

signal_file = "test_signals_test_config"
os.makedirs(f'graphOutputs/{signal_file}',exist_ok=True)
output = f"graphOutputs/{signal_file}"

df = pd.read_csv(f"outputs/{signal_file}", comment="#")

df_averaged = df.groupby(['waveform', 'bval'])['signal'].mean()

unique_waveforms = sorted(df['waveform'].unique())

md = []
kurt = []
curvesA = []
curvesB = []
curvesC = []

fig, ax = plt.subplots(figsize=(9, 5))
x_fit = np.linspace(0, max(df['bval']), 100)


for wf in unique_waveforms:
    wf_data = df_averaged.loc[wf]

    b_values = [0.0] + wf_data.index.tolist()
    signals = [1.0] + wf_data.values.tolist()

    A, B, C = np.polyfit(b_values, signals, 2)
    D = -B
    
    md.append(D)
    kurt.append((6 * A) / (D**2))
    curvesA.append(A)
    curvesB.append(B)
    curvesC.append(C)

    raw_path = df[df['waveform'] == wf]['file'].iloc[0]
    clean_filename = os.path.basename(raw_path) 
    label_name = clean_filename.replace(".csv", "") 

    y_fit = A * (x_fit**2) + B * x_fit + C

    ax.scatter(b_values, signals, marker='o', label=f"{label_name} Data")
    ax.plot(x_fit, y_fit, linestyle='--', label=f"Fit (MD: {D:.2e}, K: {kurt[-1]:.2f})")

ax.set_yscale('log')
ax.set_xlabel("b-value ($s/mm^2$)")
ax.set_ylabel("Normalized Signal ($S/S_0$)")
ax.set_title("Simulation Decay")
ax.grid(True, which="both", linestyle='--', alpha=0.5)

ax.legend(bbox_to_anchor=(1.05, 1), loc='best')

plt.tight_layout()
plt.savefig(f"{output}/test1", dpi=300)
plt.show()

fig, ax = plt.subplots(figsize=(8, 5))

freq = np.array([0,50,100])
diff = np.array([md[0]*1e3,md[1]*1e3,md[2]*1e3])

ax.scatter(freq, diff, color='red',label='MD Data Points')

x1 = np.linspace(0,200,99)

lin_params = np.polyfit(freq, diff, 1)
lin_fit = lin_params[0] * freq + lin_params[1]
lin_error = np.sum((diff - lin_fit) ** 2)

sqrt_params = np.polyfit(np.sqrt(freq), diff, 1)
sqrt_fit= sqrt_params[0] * np.sqrt(freq) + sqrt_params[1]
sqrt_error = np.sum((diff - sqrt_fit) ** 2)


sq_params = np.polyfit(freq**2, diff, 1)
sq_fit= sq_params[0] * freq**2 + sq_params[1]
sq_error = np.sum((diff - sq_fit) ** 2)

if sqrt_error < lin_error and sqrt_error < sq_error:
    y1 = sqrt_params[0] * np.sqrt(x1) + sqrt_params[1]
    y2 = lin_params[0] * x1 + lin_params[1]
    y3 = sq_params[0] * (x1**2) + sq_params[1]
    print("Best Fit: Square Root")
    
elif lin_error < sqrt_error and lin_error < sq_error:

    y1 = lin_params[0] * x1 + lin_params[1]
    y2 = sqrt_params[0] * np.sqrt(x1) + sqrt_params[1]
    y3 = sq_params[0] * (x1**2) + sq_params[1]
    print("Best Fit: Linear")
    
else:
    y1 = sq_params[0] * (x1**2) + sq_params[1]
    y2 = sqrt_params[0] * np.sqrt(x1) + sqrt_params[1]
    y3 = lin_params[0] * x1 + lin_params[1]
    print("Best Fit: Squared")

ax.set_ylabel("MD")
ax.set_xlabel("Hz")
plt.plot(x1,y1)
plt.plot(x1,y2, ls= ":", alpha = 0.7)
plt.plot(x1,y3, ls = ':', alpha= 0.7)

plt.savefig(f"{output}/test1_fmd")




"""
md = []
kurt = []
curvesA = []
curvesB = []
curvesC = []

A, B, C = np.polyfit(b_targets, log_signals[i], 2)
D = -B
md.append(D)
kurt.append((6 * A) / (D**2))
curvesA.append(A)
curvesB.append(B)
curvesC.append(C)

fig, ax = plt.subplots(figsize=(8, 5))
x = np.linspace(0,4500,5)

for i, signal_list in enumerate(shape_signals):
    y = curvesA[i]*x**2 + curvesB[i]*x +curvesC[i]   
    ax.plot(x,np.exp(y))
    ax.scatter(b_targets, signal_list, marker='o',linestyle='--', label=f"{waveforms[i]}, MD: {md[i]:.4e}, K: {kurt[i]:.4f}")

ax.set_yscale('log')
ax.set_xlabel("b values")
ax.set_ylabel("Normalized Signal")
ax.grid(True, which="both", linestyle='--', alpha=0.5)
ax.legend()
plt.savefig("CATERPILLAR_aligned_beading")
plt.show()

fig, ax = plt.subplots(figsize=(8, 5))

freq = np.array([0,50,100])
diff = np.array([md[0]*1e3,md[1]*1e3,md[2]*1e3])

ax.scatter(freq, diff, color='red',label='MD Data Points')

x1 = np.linspace(0,200,99)

lin_params = np.polyfit(freq, diff, 1)
lin_fit = lin_params[0] * freq + lin_params[1]
lin_error = np.sum((diff - lin_fit) ** 2)

sqrt_params = np.polyfit(np.sqrt(freq), diff, 1)
sqrt_fit= sqrt_params[0] * np.sqrt(freq) + sqrt_params[1]
sqrt_error = np.sum((diff - sqrt_fit) ** 2)


sq_params = np.polyfit(freq**2, diff, 1)
sq_fit= sq_params[0] * freq**2 + sq_params[1]
sq_error = np.sum((diff - sq_fit) ** 2)

if sqrt_error < lin_error and sqrt_error < sq_error:
    y1 = sqrt_params[0] * np.sqrt(x1) + sqrt_params[1]
    y2 = lin_params[0] * x1 + lin_params[1]
    y3 = sq_params[0] * (x1**2) + sq_params[1]
    print("Best Fit: Square Root")
    
elif lin_error < sqrt_error and lin_error < sq_error:

    y1 = lin_params[0] * x1 + lin_params[1]
    y2 = sqrt_params[0] * np.sqrt(x1) + sqrt_params[1]
    y3 = sq_params[0] * (x1**2) + sq_params[1]
    print("Best Fit: Linear")
    
else:
    y1 = sq_params[0] * (x1**2) + sq_params[1]
    y2 = sqrt_params[0] * np.sqrt(x1) + sqrt_params[1]
    y3 = lin_params[0] * x1 + lin_params[1]
    print("Best Fit: Squared")

ax.set_ylabel("MD")
ax.set_xlabel("Hz")
plt.plot(x1,y1)
plt.plot(x1,y2, ls= ":", alpha = 0.7)
plt.plot(x1,y3, ls = ':', alpha= 0.7)

plt.savefig("CATERPILLAR_aligned_beading_mdWRTf")

"""