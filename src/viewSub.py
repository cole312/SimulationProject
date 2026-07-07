import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# 1. Paths to your files (change if names differ)
substrate_file = "substrate/beaded_bundle_rfiber_0.337_rbead_1.000_VOL"

v_file = f"{substrate_file}_vertices.csv"
f_file = f"{substrate_file}_faces.csv"
output_png = f"{substrate_file}_view.png"

vertices = np.loadtxt(v_file, delimiter=",")
faces = np.loadtxt(f_file, delimiter=",")

# 3. Create an off-screen 3D plot
fig = plt.figure(figsize=(10, 8), dpi=150)
ax = fig.add_subplot(111, projection='3d')

# Render the surface mesh cleanly
ax.plot_trisurf(
    vertices[:, 0], vertices[:, 1], vertices[:, 2], 
    triangles=faces, 
    cmap='viridis', 
    edgecolor='black', 
    linewidth=0.2, 
    alpha=0.9
)

# 4. Set equal aspect ratio so the axon beads don't look stretched or squished
max_range = np.array([
    vertices[:, 0].max() - vertices[:, 0].min(),
    vertices[:, 1].max() - vertices[:, 1].min(),
    vertices[:, 2].max() - vertices[:, 2].min()
]).max() / 2.0

mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

# Clean labels
ax.set_xlabel("X (μm)")
ax.set_ylabel("Y (μm)")
ax.set_zlabel("Z (μm)")
ax.set_title("Beaded Axon Mesh Substrate")

# 5. Export directly to a PNG file
os.makedirs(os.path.dirname(output_png), exist_ok=True)
plt.savefig(output_png, bbox_inches='tight')
plt.close()
