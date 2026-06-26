import trimesh
import matplotlib.pyplot as plt
import pandas as pd
import os
import tomli
import numpy as np



#with open("sim_configs/test_config.toml", "rb") as f:
    #config = tomli.load(f)

#name = config["substrate"]["name"]
name = "multiSphere_radiusMany"
os.makedirs(f'substrate/{name}',exist_ok=True)

#start mesh
mesh8 = trimesh.creation.uv_sphere(1)
mesh2 = trimesh.creation.uv_sphere(1.5).apply_translation([2.5,3,-3])
mesh3 = trimesh.creation.uv_sphere(3).apply_translation([4,4,4])
mesh4 = trimesh.creation.uv_sphere(2).apply_translation([-3,0,4])
mesh5 = trimesh.creation.uv_sphere(2.5).apply_translation([0,6,2])
mesh6 = trimesh.creation.uv_sphere(2).apply_translation([-2,-4,-2])
mesh7 = trimesh.creation.uv_sphere(1.5).apply_translation([-4,4,-4])

mesh = trimesh.util.concatenate([mesh8,mesh2,mesh3,mesh4,mesh5,mesh6,mesh7])

#end mesh

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

ax.axis('equal')
plt.savefig(f'substrate/{name}/{name}_mesh.png', dpi= 300, bbox_inches='tight')
plt.close(fig)

#csv export
data_verticies = pd.DataFrame(mesh.vertices, columns=['x','y','z'])
data_verticies.to_csv(f'substrate/{name}/{name}_vertices.csv', index=False)

data_faces = pd.DataFrame(mesh.faces, columns=['v1','v2','v3'])
data_faces.to_csv(f'substrate/{name}/{name}_faces.csv', index=False)



