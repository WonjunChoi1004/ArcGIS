import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# Generate random samples for soil properties
num_simulations = 10000
cohesion_samples = np.random.normal(loc=25, scale=5, size=num_simulations)
phi_samples = np.random.normal(loc=30, scale=5, size=num_simulations)

# Create a heatmap of these values
bins = 50
heatmap, xedges, yedges = np.histogram2d(cohesion_samples, phi_samples, bins=bins)

# Define custom colormap: white → red → black
cmap = colors.LinearSegmentedColormap.from_list("white_red_black", ["white", "red", "black"])
norm = colors.Normalize(vmin=0.5, vmax=heatmap.max())

# Plot heatmap
plt.figure(figsize=(10, 6))
mesh = plt.pcolormesh(xedges, yedges, heatmap.T, cmap=cmap, norm=norm, shading='auto')
cbar = plt.colorbar(mesh)
cbar.set_label('Number of Simulations', fontsize=16)
cbar.ax.tick_params(labelsize=14)

# Labels and larger title
plt.xlabel('Cohesion c (kPa)', fontsize=18)
plt.ylabel('Friction Angle φ (degrees)', fontsize=18)
plt.title('Monte Carlo Simulation: Sampled Soil Properties', fontsize=22)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.tight_layout()
plt.show()
