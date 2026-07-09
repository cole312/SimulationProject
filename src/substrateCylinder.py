import os
import trimesh
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

n_cylinders = 50
cyl_radius = 0.5
cyl_height = 10.0
domain_size = 30.0 
output_dir = f"substrate/{n_cylinders}_cylindersUP"
os.makedirs(output_dir, exist_ok=True)

cylinder_list = []
placed_positions = [] 

def is_overlapping(x, y, existing_positions, min_dist):
    for (ex, ey) in existing_positions:
        dist = np.sqrt((x - ex)**2 + (y - ey)**2)
        if dist < min_dist:
            return True
    return False

for i in range(n_cylinders):
    placed = False
    attempts = 0
    max_attempts = 1000
    
    while not placed and attempts < max_attempts:

        x_pos = np.random.uniform(-domain_size/2, domain_size/2)
        y_pos = np.random.uniform(-domain_size/2, domain_size/2)
        
        if not is_overlapping(x_pos, y_pos, placed_positions, cyl_radius * 2.2):
            cyl = trimesh.creation.cylinder(radius=cyl_radius, height=cyl_height)
            cyl.apply_translation([x_pos, y_pos, 0])
            
            cylinder_list.append(cyl)
            placed_positions.append((x_pos, y_pos))
            placed = True
        
        attempts += 1
    
    if not placed:
        print(f"Warning: Could not place cylinder {i+1}")


combined_mesh = trimesh.util.concatenate(cylinder_list)


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')


ax.plot_trisurf(
    combined_mesh.vertices[:, 0], 
    combined_mesh.vertices[:, 1], 
    combined_mesh.vertices[:, 2], 
    triangles=combined_mesh.faces, 
    cmap='viridis', edgecolor='none', alpha=0.9
)

ax.set_xlabel("X (μm)"); ax.set_ylabel("Y (μm)"); ax.set_zlabel("Z (μm)")
plt.savefig(f'{output_dir}/substrate_mesh.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# Save for Disimpy
data_vertices = pd.DataFrame(combined_mesh.vertices, columns=['x', 'y', 'z'])
data_vertices.to_csv(f'{output_dir}/substrate_vertices.csv', index=False)
data_faces = pd.DataFrame(combined_mesh.faces, columns=['v1', 'v2', 'v3'])
data_faces.to_csv(f'{output_dir}/substrate_faces.csv', index=False)

print(f"done")