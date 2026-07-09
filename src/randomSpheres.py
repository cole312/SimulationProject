import os
import trimesh
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Setup names and folders
substrate_name = "spheres_3d_distribution_2_to_10_um"
output_dir = f"substrate/{substrate_name}"
os.makedirs(output_dir, exist_ok=True)

np.random.seed(42) 
num_spheres = 25
radii = np.random.uniform(2.0, 10.0, num_spheres)

box_size = 60.0
buffer = 1.0 

placed_positions = []
sphere_meshes = []

print("Positioning spheres randomly in 3D space...")

for i, r in enumerate(radii):
    placed = False
    attempts = 0
    max_attempts = 2000
    
    while not placed and attempts < max_attempts:
        attempts += 1
        
        candidate_pos = np.random.uniform(-box_size/2, box_size/2, size=3)
        
        overlapping = False
        for prev_pos, prev_radius in placed_positions:
            distance = np.linalg.norm(candidate_pos - prev_pos)
            
            # They overlap if distance is less than the sum of their radii + buffer
            if distance < (r + prev_radius + buffer):
                overlapping = True
                break
                
        # If no overlap found, lock in the position
        if not overlapping:
            placed_positions.append((candidate_pos, r))
            
            # Create the uv_sphere and translate it to its 3D coordinate
            sphere = trimesh.creation.uv_sphere(radius=r)
            sphere.apply_translation(candidate_pos)
            sphere_meshes.append(sphere)
            
            placed = True
            print(f"Sphere {i+1}/10 placed at: XYZ={np.round(candidate_pos, 1)} | Radius={r:.2f} μm")

    if not placed:
        print(f"⚠️ Warning: Could not find a non-overlapping spot for sphere {i+1} after {max_attempts} attempts.")

# 4. Concatenate all individual spheres into a single mesh substrate
final_mesh = trimesh.util.concatenate(sphere_meshes)

# Convert from microns to meters to match your simulation pipeline scale
final_mesh.apply_scale(1e-6)

# 5. Generate and export the clean 3D validation plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot_trisurf(
    final_mesh.vertices[:, 0], 
    final_mesh.vertices[:, 1], 
    final_mesh.vertices[:, 2], 
    triangles=final_mesh.faces, 
    cmap='viridis',
    edgecolor='none', 
    alpha=0.9
)

# Set equal bounding box constraints to avoid stretching visual aspect ratios
max_range = np.array([
    final_mesh.vertices[:, 0].max() - final_mesh.vertices[:, 0].min(),
    final_mesh.vertices[:, 1].max() - final_mesh.vertices[:, 1].min(),
    final_mesh.vertices[:, 2].max() - final_mesh.vertices[:, 2].min()
]).max() / 2.0

mid_x = (final_mesh.vertices[:, 0].max() + final_mesh.vertices[:, 0].min()) * 0.5
mid_y = (final_mesh.vertices[:, 1].max() + final_mesh.vertices[:, 1].min()) * 0.5
mid_z = (final_mesh.vertices[:, 2].max() + final_mesh.vertices[:, 2].min()) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("Non-Overlapping 3D Random Sphere Cloud Substrate")

plt.savefig(f'{output_dir}/{substrate_name}_mesh.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# 6. Export the data arrays in your standard CSV format
data_vertices = pd.DataFrame(final_mesh.vertices, columns=['x', 'y', 'z'])
data_vertices.to_csv(f'{output_dir}/{substrate_name}_vertices.csv', index=False)

data_faces = pd.DataFrame(final_mesh.faces, columns=['v1', 'v2', 'v3'])
data_faces.to_csv(f'{output_dir}/{substrate_name}_faces.csv', index=False)

print(f"\nSuccess. Cloud mesh files exported cleanly to: {output_dir}/")