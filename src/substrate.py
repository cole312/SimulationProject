import trimesh
import matplotlib.pyplot as plt
import pandas as pd
import os
import tomli
import numpy as np



#with open("sim_configs/test_config.toml", "rb") as f:
    #config = tomli.load(f)

#name = config["substrate"]["name"]
name = "multiCylinder_radius10_height_500"
os.makedirs(f'substrate/{name}',exist_ok=True)

mesh8 = trimesh.creation.cylinder(1,50)
mesh2 = (trimesh.creation.cylinder(1,50)).apply_translation([3,0,0])
mesh2.apply_transform(trimesh.transformations.rotation_matrix(np.radians(45), [1,0,0]))
mesh3 = (trimesh.creation.cylinder(1,50)).apply_translation([0,3,0])
mesh3.apply_transform(trimesh.transformations.rotation_matrix(np.radians(60), [1,1,0]))
mesh4 = (trimesh.creation.cylinder(1,50)).apply_translation([0,-3,0])
mesh4.apply_transform(trimesh.transformations.rotation_matrix(np.radians(30), [1,-1,0]))
mesh5 = (trimesh.creation.cylinder(1,50)).apply_translation([-3,0,2])
mesh5.apply_transform(trimesh.transformations.rotation_matrix(np.radians(90), [1,0,0]))

mesh = trimesh.util.concatenate([mesh8,mesh2,mesh3,mesh4,mesh5])

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



