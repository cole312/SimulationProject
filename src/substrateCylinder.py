import os
import trimesh
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

n_cylinders = 2000
cyl_height = 10
domain_size = 30

output_dir = f"substrate/{n_cylinders}_cylinders_rad_dis_0.05_0.5"
os.makedirs(output_dir, exist_ok=True)

cylinder_list = []
placed_positions = []  # (x, y, radius)


def is_overlapping(x, y, radius, existing_positions):
    for ex, ey, eradius in existing_positions:
        dist = np.sqrt((x - ex) ** 2 + (y - ey) ** 2)
        if dist < (radius + eradius):
            return True
    return False


for i in range(n_cylinders):
    placed = False
    attempts = 0
    max_attempts = 5000

    cyl_radius = np.random.uniform(0.05, 0.5)

    while not placed and attempts < max_attempts:

        x_pos = np.random.uniform(-domain_size / 2, domain_size / 2)
        y_pos = np.random.uniform(-domain_size / 2, domain_size / 2)

        if not is_overlapping(x_pos, y_pos, cyl_radius, placed_positions):

            cyl = trimesh.creation.cylinder(
                radius=cyl_radius,
                height=cyl_height
            )
            cyl.apply_translation([x_pos, y_pos, 0])

            cylinder_list.append(cyl)
            placed_positions.append((x_pos, y_pos, cyl_radius))
            placed = True

        attempts += 1

    if not placed:
        print(f"Warning: Could not place cylinder {i + 1}")

print(f"Placed {len(cylinder_list)} cylinders.")

combined_mesh = trimesh.util.concatenate(cylinder_list)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

ax.plot_trisurf(
    combined_mesh.vertices[:, 0],
    combined_mesh.vertices[:, 1],
    combined_mesh.vertices[:, 2],
    triangles=combined_mesh.faces,
    cmap="viridis",
    edgecolor="none",
    alpha=0.9,
)

ax.set_xlabel("X (μm)")
ax.set_ylabel("Y (μm)")
ax.set_zlabel("Z (μm)")

plt.savefig(
    f"{output_dir}/substrate_mesh.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

print("Bounds before scaling:")
print(combined_mesh.bounds)

combined_mesh.apply_scale(1e-6)

print("Bounds after scaling:")
print(combined_mesh.bounds)

data_vertices = pd.DataFrame(
    combined_mesh.vertices,
    columns=["x", "y", "z"]
)
data_vertices.to_csv(
    f"{output_dir}/{n_cylinders}_cylindersUP_vertices.csv",
    index=False
)

data_faces = pd.DataFrame(
    combined_mesh.faces,
    columns=["v1", "v2", "v3"]
)
data_faces.to_csv(
    f"{output_dir}/{n_cylinders}_cylindersUP_faces.csv",
    index=False
)

print("Watertight:", combined_mesh.is_watertight)
print("Volume:", combined_mesh.is_volume)
print("Done.")