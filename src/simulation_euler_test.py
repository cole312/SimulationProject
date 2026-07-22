import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from disimpy import gradients, simulations, substrates
from dipy.core.sphere import fibonacci_sphere
import pandas as pd
import tomli
import csv
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "config",
    type=str,

)

args = parser.parse_args()

config_file_path = args.config

config_name = os.path.basename(config_file_path)
config_name,_ = os.path.splitext(config_name)

with open(config_file_path, "rb") as f:
    config = tomli.load(f)

#Params
meshName = config["substrate"]["name"]
n_walkers = config["simulation"]["n_walkers"]
n_t = config["simulation"]["n_t"]
diffusivity = config["simulation"]["diffusivity"]
waveforms = config["waveform"]["waveform_file"]
eulerFile = config["waveform"]["direction_file"]
b_num = config["waveform"]["num_b"]
position = config["substrate"]["position"]

rotations = np.loadtxt(f"euler_rotations/{eulerFile}", comments="#")
rot_matrix = rotations.reshape(-1, 3, 3)

def get_substrate(meshName):

    print(meshName)
    data_verts = pd.read_csv(f'substrate/{meshName}/{meshName}_vertices.csv')
    data_faces = pd.read_csv(f'substrate/{meshName}/{meshName}_faces.csv')

    vertices = data_verts.to_numpy()
    faces = data_faces.to_numpy()

    substrate = substrates.mesh(
        vertices,
        faces,
        periodic=True,
        n_sv=np.array([10, 10, 50]),
        quiet= False,
        init_pos=position
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
os.makedirs("traj_files", exist_ok=True)
csv_filename=f"outputs/{meshName}_signals_{config_name}.csv"

with open(csv_filename, mode="w", newline="") as f:
    writer = csv.writer(f)
    
    f.write(f"# Config file used: {config_name}\n")
    writer.writerow(["file", "waveform_idx", "R11", "R12", "R13", "R21", "R22", "R23", "R31", "R32", "R33", "bval", "signal"])

    shape_signals = []
    for filecount, file in enumerate(waveforms):

        x_grad, y_grad, z_grad = read_shape(file)
        
        time = len(x_grad)*0.02
        time_points = np.arange(0,time,0.02)

        gradient = np.zeros([1,len(time_points),3])

        gradient[0,:,0] = x_grad
        gradient[0,:,1] = y_grad
        gradient[0,:,2] = z_grad

        gradient *= 1e-3

        print(f"Bval: {(gradients.calc_b(gradient,0.02e-3)*1e-6)[0]:.0f}")

        gradient_final = np.zeros([len(rot_matrix), len(time_points), 3])
        mega_gradient = []

        for i in range(0, len(rot_matrix)):

            rot_waveform = gradient @ rot_matrix[i].T

            gradient_final[i, : , : ] = rot_waveform

        gradient_final, dt = gradients.interpolate_gradient(gradient_final, 0.02e-3, int(n_t))
        
        b_base = (gradients.calc_b(gradient_final, dt) * 1e-6)
        print(f"B-base: {b_base[0]}")

        signals = []
        b_targets = np.linspace(0, 4500, b_num) 

        #Scale gradient values to achieve different b's. Scale is calcuated via b_base
        for j, b in enumerate(b_targets):
            if b == 0:
                signals.append(n_walkers) 
                continue

            scale = np.sqrt(b / b_base[0])
            scaled_gradient = gradient_final * scale

            mega_gradient.append(scaled_gradient)

        mega_gradient = np.concatenate(mega_gradient, axis=0)

        print(f"Running mega simulation for waveform: {file}")

        signal = simulations.simulation(
        n_walkers=int(n_walkers),
        diffusivity=diffusivity,
        gradient=mega_gradient,
        dt=dt,
        substrate=substrate,
        )
        
        norm_signal = abs(signal / n_walkers)

        print(norm_signal)

        signal_idx = 0
        for b in b_targets:
            if b == 0:
                for i in range(len(rot_matrix)):
                    writer.writerow([file, filecount + 1, *rotations[i], b, 1.0])
            else:
                for i in range(len(rot_matrix)):
                    writer.writerow([file, filecount + 1, *rotations[i], b, norm_signal[signal_idx]])
                    signal_idx += 1



