import trimesh
import matplotlib.pyplot as plt
import pandas as pd
import os

#Name mesh here
name = 'testCat'
folder = (f'{name}_outputs')

#Set to false for complex meshs.
plotMesh = False

#make new directory for all outputs
os.mkdir(f'/home/summer1/simulation/SLURM/outputs/{folder}')


#Create mesh here
meshs = []

with open("/home/summer1/simulation/CATERPillar/testing/singleaxoncurve.csv") as f:
    next(f)  # Skip the header line

    for line in f:
        vals = line.strip().split()
        if not vals:
            continue

        x = float(vals[4])
        y = float(vals[5])
        z = float(vals[6])
        r = float(vals[7])

        # Create the sphere primitive
        mesh = trimesh.creation.icosphere(radius=r,subdivisions=1)

        # Apply the translation in-place
        mesh.apply_translation([x, y, z])
        meshs.append(mesh)

print(f"Number of spheres: {len(meshs)}")
mesh = trimesh.boolean.union(meshs, engine="manifold")
independent_volumes = mesh.split()
num_volumes = len(independent_volumes)
print(f"Union completed. Number of volumes: {num_volumes}")
trimesh.smoothing.filter_laplacian(mesh, iterations=1)
print(f"Smoothing completed")

if plotMesh:
    
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
    plt.savefig(f'/home/summer1/simulation/SLURM/outputs/{folder}/{name}_mesh', dpi= 300, bbox_inches='tight')


#csv export
data_verticies = pd.DataFrame(mesh.vertices, columns=['x','y','z'])
data_verticies.to_csv(f'/home/summer1/simulation/SLURM/outputs/{folder}/{name}_vertices', index=False)

data_faces = pd.DataFrame(mesh.faces, columns=['v1','v2','v3'])
data_faces.to_csv(f'/home/summer1/simulation/SLURM/outputs/{folder}/{name}_faces', index=False)



