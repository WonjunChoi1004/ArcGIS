import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Set up predicted failure probabilities (0–35%)
bins = np.arange(0, 35, 1)

# Simulate predicted probabilities for observed and unobserved landslide sites
np.random.seed(0)
observed_probs = np.random.beta(2, 8, 1000) * 35
unobserved_probs = np.random.beta(1, 20, 1000) * 35

# Histogram frequency distributions
observed_hist, _ = np.histogram(observed_probs, bins=bins, density=True)
unobserved_hist, _ = np.histogram(unobserved_probs, bins=bins, density=True)
bin_centers = (bins[:-1] + bins[1:]) / 2

# Create the main plot with larger fonts
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(bin_centers, unobserved_hist, label='no observations', color='black', linewidth=2)
ax.plot(bin_centers, observed_hist, label='observed landslides', color='black', marker='s', linewidth=2)
ax.set_xlabel('Predicted Failure Probability [%]', fontsize=24)
ax.set_ylabel('Frequency of Observed Landslides', fontsize=24)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.legend(fontsize=16)
ax.set_title('Discrimination Diagram: Landslide Prediction', fontsize=30)

# Add inset plot with log y-axis
ax_inset = inset_axes(ax, width="40%", height="40%", loc='upper right')
ax_inset.plot(bin_centers, unobserved_hist, color='black', linewidth=1)
ax_inset.plot(bin_centers, observed_hist, color='black', marker='s', linewidth=1)
ax_inset.set_yscale('log')
ax_inset.set_xticks([])
ax_inset.set_yticks([])

plt.tight_layout()
plt.show()