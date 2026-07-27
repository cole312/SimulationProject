from disimpy import simulations, substrates, gradients
import numpy as np

gradient = np.zeros((1,100,3))
gradient[0,:,0] = 0.01

gradient, dt = gradients.interpolate_gradient(
    gradient,
    0.02e-3,
    100
)

print("starting")

signal = simulations.simulation(
    n_walkers=10,
    diffusivity=1e-9,
    gradient=gradient,
    dt=dt,
    substrate=substrates.free(),
    cuda_bs=128
)

print(signal)