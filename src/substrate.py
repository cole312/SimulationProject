import trimesh
import matplotlib.pyplot as plt
import pandas as pd
import os
import tomli
import numpy as np



#with open("sim_configs/test_config.toml", "rb") as f:
    #config = tomli.load(f)

#name = config["substrate"]["name"]
name = f""
os.makedirs(f'substrate/{name}',exist_ok=True)

#start mesh

#Params
length = 60
radius_fiberOG = 0.6
spacing = 5

n_t = int(2e3) 
n_walkers = int(1e4)
T = 20e-3
diffusivity = 3e-9

n = int(length / spacing)


def create_ba(length, radius_fiber, radius_bead, spacing ):

    """
    Returns beaded axon mesh (trimesh)
    """

    shaft = trimesh.creation.cylinder(radius=radius_fiber, height=length)
    
    beads = []
    z_positions = np.arange(-length/2, length/2, spacing)
    
    if radius_bead > 0:
        for z in z_positions:
            num = np.random.randint(0,spacing)
            bead = trimesh.creation.uv_sphere(radius=radius_bead)
            bead.apply_translation([0, 0, z +num])
            if z+num < length/2 and z+num>-length/2:
                beads.append(bead)

        all_parts = [shaft] + beads

        beaded_mesh = trimesh.boolean.union(all_parts,'manifold')
    else:
        beaded_mesh = shaft
          
    return beaded_mesh

bvecs = fibonacci_sphere(n_points=30)
bvecs.shape

bvecs = np.vstack([np.zeros((1,3)), bvecs])
bvals = np.array([0] + [1e9]*30)

gradient= np.zeros((1,100,3))

gradient[0,1:40,0] = 1
gradient[0,60:99,0] = -1

dt = T / (gradient.shape[1] - 1)
gradient, dt = gradients.interpolate_gradient(gradient, dt, int(n_t))

full_gradient = np.zeros((len(bvals), gradient.shape[1], 3))

for idx, (bval,bvec) in enumerate(zip(bvals,bvecs)):
    if bval == 0:
        continue

    full_gradient[idx, :, 0] = gradient[0, :, 0] * bvec[0] 
    full_gradient[idx, :, 1] = gradient[0, :, 0] * bvec[1] 
    full_gradient[idx, :, 2] = gradient[0, :, 0] * bvec[2]

full_gradient[1:] = gradients.set_b(full_gradient[1:], dt, bvals[1:])

print(full_gradient.shape)
print("Gradient created.")


#Size down vals for calculations
lengthS = length*1e-6
spacingS = spacing*1e-6


#Find baseline (no beading)
baselineMesh = create_ba(length, radius_fiberOG, 0, spacing)
baselineMesh.apply_scale(1e-6)
baseline = baselineMesh.area

max_beading = np.sqrt(baseline/(n*4*np.pi))

print(f"\nBaseline: {baseline:.3e}")
print(f"Number of beads: {n}")
print(f"Maximum bead radius allowed: {max_beading:.3e}")

fa_list = []
md_list = []
ad_list = []

radius_list = []

for radius_bead in np.arange(radius_fiberOG*1e-6, max_beading, 1e-7):
#for i in range(1):

    #radius_bead = 0.6e-6


    radius_list.append(radius_bead)

    #Calcs what radius of fiber needs to be
    if radius_bead > radius_fiberOG*1e-6:
        term_b = np.pi * (lengthS - 2 * radius_bead * (n - 1))
        inner_sqrt = (np.pi**2 * (lengthS - 2 * radius_bead * (n - 1))**2 + 2 * (n - 1) * np.pi * (4 * n * np.pi * radius_bead**2 - baseline))
        denominator = -2 * (n - 1) * np.pi

        radius_fiber = (-term_b + np.sqrt(inner_sqrt)) / denominator

        if radius_fiber < 0:
            radius_fiber = (-term_b - np.sqrt(inner_sqrt)) / denominator

        #print(f"Baseline(No beading): {baseline:.3e}")
        #print(f"Requested beading radius: {radius_bead}" )

        print(f"New Fiber Radius:      {radius_fiber:.3e}")

        area_beads = n*4*np.pi*radius_bead**2-2*(n-1)*np.pi*radius_fiber**2
        area_fiber_segments = 2*np.pi*radius_fiber*(lengthS-2*radius_bead*(n-1))
    else:
        print("Fiber radius > bead radius.")
        radius_fiber = radius_fiberOG*1e-6

    #print(f"\nDifference:{abs(area_beads + area_fiber_segments - baseline):.0e}")


    #create new axons with the fiber radius

    axons = []

    for i in range(3):
        for j in range(3):
            x = j* (2*radius_fiberOG+0.5) +  i%2 * radius_fiberOG 
            y = i * (2*radius_fiberOG+0.5)

            axon = create_ba(length,radius_fiber*1e6, radius_bead*1e6, spacing)
            axon.apply_translation([x,y,0])

            axons.append(axon)


    axons = trimesh.util.concatenate(axons)

    axons = trimesh.intersections.slice_mesh_plane(axons, [0,0,1],[0,0,-length/2 + 0.1])
    axons = trimesh.intersections.slice_mesh_plane(axons, [0,0,-1],[0,0,length/2 - 0.1]) 


    axons.apply_scale(1e-6)

    #Simulate each set of axons

    paddingVal = 1e-8
    padding = np.array([paddingVal,paddingVal,0])

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



