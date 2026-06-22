import trimesh
import matplotlib.pyplot as plt
import pandas as pd
import os

#Name mesh here
name = 'test'


folder = (f'{name}_outputs')
os.mkdir(f'outputs/{folder}')

#Create mesh here
mesh = []
mesh.apply_scale(1e-6)

#Plotting mesh.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
mesh.apply_scale(1e-6)
ax.plot_trisurf(
    mesh.vertices[:, 0], 
    mesh.vertices[:, 1], 
    mesh.vertices[:, 2], 
    triangles=mesh.faces, 
    cmap='viridis',
    edgecolor='k', 
    linewidth=0.5
)

ax.set_box_aspect([1, 1, 1])
plt.savefig(f'outputs/{folder}/{name}_mesh', dpi= 300, bbox_inches='tight')

#csv export
data_verticies = pd.DataFrame(mesh.vertices, columns=['x','y','z'])
data_verticies.to_csv(f'outputs/{folder}/{name}_vertices', index=False)

data_faces = pd.DataFrame(mesh.faces, columns=['v1','v2','v3'])
data_faces.to_csv(f'outputs/{folder}/{name}_faces', index=False)



