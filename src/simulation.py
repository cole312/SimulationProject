import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from disimpy import gradients, simulations, substrates
from dipy.core.sphere import fibonacci_sphere
import pandas as pd
import tomli
import csv

with open("sim_configs/test_config.toml", "rb") as f:
    config = tomli.load(f)

#Params
meshName = config["substrate"]["name"]
folder = f"{meshName}_outputs"
n_walkers = config["simulation"]["n_walkers"]
n_t = config["simulation"]["n_t"]
diffusivity = config["simulation"]["diffusivity"]
waveforms = config["waveform"]["waveform_file"]
directions = config["waveform"]["num_directions"]
b_num = config["waveform"]["num_b"]
position = config["substrate"]["position"]


def get_substrate(meshName):

    print(meshName)
    data_verts = pd.read_csv(f'substrate/{meshName}/{meshName}_vertices')
    data_faces = pd.read_csv(f'substrate/{meshName}/{meshName}_faces')

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


substrate = get_substrate(meshName)

#waveform, x, y, z, b value, 

csv_filename=f"outputs/{meshName}_signals"
with open(csv_filename, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["waveform","x","y","z","bval","signal"])

    shape_signals = []
    for filecount, file in enumerate(waveforms):

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
        bvecs = fibonacci_sphere(n_points=directions)


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
        b_targets = np.linspace(0, 4500, b_num) 

        #Scale gradient values to achieve different b's. Scale is calcuated via b_base
        for j, b in enumerate(b_targets):
            print(f"\n\nGradient: {filecount + 1} / {len(waveforms)}. B value: {j} / {len(b_targets)}\n\n")
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
            
            # Loop through each direction to log its specific x, y, z component and signal.
            for i, bvec in enumerate(bvecs):

                norm_signal = abs(signal / n_walkers)
                
                # Write a row to the CSV file
                writer.writerow([filecount+1, bvec[0], bvec[1], bvec[2], b, norm_signal[i]])




