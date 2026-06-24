import trimesh
import matplotlib.pyplot as plt
import pandas as pd
import os
import tomli

#Name mesh here

with open("sim_configs/test_config.toml", "rb") as f:
    config = tomli.load(f)

name = config["substrate"]["name"]

os.mkdir(f'substrate/{name}',exist_ok=True)

#Create mesh here
mesh = trimesh.creation.cylinder(radius=1, height=10)

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
plt.savefig(f'substrate/{name}/{name}_mesh.png', dpi= 300, bbox_inches='tight')
plt.close(fig)

#csv export
data_verticies = pd.DataFrame(mesh.vertices, columns=['x','y','z'])
data_verticies.to_csv(f'substrate/{name}/{name}_vertices.csv', index=False)

data_faces = pd.DataFrame(mesh.faces, columns=['v1','v2','v3'])
data_faces.to_csv(f'substrate/{name}/{name}_faces.csv', index=False)



