import numpy as np
from scipy.spatial.transform import Rotation as R
import trimesh
import matplotlib.pyplot as plt
from disimpy import gradients, simulations, substrates
from dipy.core.sphere import fibonacci_sphere
import pandas as pd


def get_substrate(meshName, folder):

    print(meshName)
    data_verts = pd.read_csv(f'/home/summer1/simulation/SLURM/outputs/{folder}/{meshName}_vertices')
    data_faces = pd.read_csv(f'/home/summer1/simulation/SLURM/outputs/{folder}/{meshName}_faces')

    vertices = data_verts.to_numpy()
    faces = data_faces.to_numpy()

    substrate = substrates.mesh(
        vertices,
        faces,
        periodic=True,
        n_sv=np.array([10, 10, 50]),
        quiet= False,
        init_pos='intra'
    )

    return substrate

def read_shape(filename):
    """ 
    Takes x,y,z CSV vaules from MATLAB. 
    Returns x,y,z list of gradient values.
    """
    x_grad = []
    y_grad = []
    z_grad = []

    with open(filename) as f:
        for line in f:
            vals = line.strip().split(',')
            if len(vals)>1:
                x_grad.append(float(vals[0]))
                y_grad.append(float(vals[1]))
                z_grad.append(float(vals[2]))
            else:
                x_grad.append(float(vals[0]))
                y_grad.append(0)
                z_grad.append(0)

    return x_grad,y_grad,z_grad

#Params
meshName = "testSphere"
folder = f"{meshName}_outputs"
n_walkers = 100
n_t = 1e3
diffusivity = 1e-9

#List of file names to iterate through
filenames = [
    "/home/summer1/simulation/waveform1.csv",
    "/home/summer1/simulation/waveform2.csv",
    "/home/summer1/simulation/waveform3.csv",
    "/home/summer1/simulation/waveform4.csv",
    "/home/summer1/simulation/waveform5.csv"
]
substrate = get_substrate(meshName, folder)

shape_signals = []

for filecount, file in enumerate(filenames):

    x_grad, y_grad, z_grad = read_shape(file)
    
    time = len(x_grad)*0.02
    time_points = np.arange(0,time,0.02)

    gradient = np.zeros([1,len(time_points),3])

    #Store the x,y,z data into a single gradient file. 
    gradient[0,:,0] = x_grad
    gradient[0,:,1] = y_grad
    gradient[0,:,2] = z_grad

    gradient *= 1e-3

    print(f"Bval: {(gradients.calc_b(gradient,0.02e-3)*1e-6)[0]:.0f}")

    #30 equally spaced points around a sphere. These are target vectors for the rotation
    bvecs = fibonacci_sphere(n_points=30)


    gradient_final = np.zeros([len(bvecs), len(time_points), 3])

    for i in range(0, len(bvecs)):

        target = bvecs[i]

        rotation, _ = R.align_vectors([target], [1,0,0])
        rot_matrix = rotation.as_matrix()

        rot_waveform = gradient @ rot_matrix.T

        gradient_final[i, : , : ] = rot_waveform

    #Interpolate gradient to variable step size (n_t), dt is the new time step
    gradient_final, dt = gradients.interpolate_gradient(gradient_final, 0.02e-3, int(n_t))
    
    b_base = (gradients.calc_b(gradient_final, dt) * 1e-6)
    print(f"B-base: {b_base[0]}")

    signals = []
    b_targets = np.linspace(0, 4500, 6) 

    #Scale gradient values to achieve different b's. Scale is calcuated via b_base
    for j, b in enumerate(b_targets):
        print(f"\n\nGradient: {filecount + 1} / {len(filenames)}. B value: {j} / {len(b_targets)}\n\n")
        if b == 0:
            signals.append(n_walkers) 
            continue

        scale = np.sqrt(b / b_base[0])
        scaled_gradient = gradient_final * scale

        #Simulate with the new scaled gradient. All signals are stored in shape_signals

        #traj_file = "example_traj.txt"
        signal = simulations.simulation(
        n_walkers=int(n_walkers),
        diffusivity=diffusivity,
        gradient=scaled_gradient,
        dt=dt,
        substrate=substrate,
        #traj=traj_file,
        )
        
        signals.append(np.mean(signal))

    signals = [abs(signal / n_walkers) for signal in signals]

    shape_signals.append(signals)

log_signals = np.log(shape_signals)

md = []
kurt = []
curvesA = []
curvesB = []
curvesC = []

for i in range(len(filenames)):
    A, B, C = np.polyfit(b_targets, log_signals[i], 2)
    D = -B
    md.append(D)
    kurt.append((6 * A) / (D**2))
    curvesA.append(A)
    curvesB.append(B)
    curvesC.append(C)

waveforms = [
    "LTE 0HZ",
    "LTE 50HZ",
    "LTE 100Hz",
    "STE iso",
    "STE Aniso"
]

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
plt.savefig(f"/home/summer1/simulation/SLURM/outputs/{folder}/{meshName}_signals_{n_walkers}walkers_{n_t}steps.png")
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

plt.savefig(f"/home/summer1/simulation/SLURM/outputs/{folder}/{meshName}_Md_F_graph.png")



