# rainfall_window_timeline_1x2.py

import matplotlib.pyplot as plt

OUT_PNG = "rainfall_window_timeline_30days_1x2.png"

# 1 (width) x 2 (height) style figure
fig, ax = plt.subplots(figsize=(6, 3), dpi=300)

# Timeline range (days before event)
ax.set_xlim(-30, 1)
ax.set_ylim(0, 5)

# Remove axes
ax.axis("off")

# Baseline
ax.hlines(y=1, xmin=-30, xmax=0, linewidth=2)

# Event day marker
ax.vlines(x=0, ymin=0.7, ymax=1.3, linewidth=2)
ax.text(0, 0.45, "Event Day\n(Landslide)", ha="center", fontsize=8, weight="bold")

# Rainfall windows
windows = [
    (-1, "R1d"),
    (-3, "R3d"),
    (-7, "R7d"),
    (-30, "R30d"),
]

y_levels = [2, 2.8, 3.6, 4.4]

for (start, label), y in zip(windows, y_levels):
    ax.hlines(y=y, xmin=start, xmax=0, linewidth=7)
    ax.text(start - 0.8, y, label, ha="right", va="center", fontsize=8)

# Time labels
ax.text(-30, 0.6, "-30 days", ha="center", fontsize=7)
ax.text(-15, 0.6, "Antecedent\nRainfall", ha="center", fontsize=7)
ax.text(0, 0.6, "0", ha="center", fontsize=7)

plt.tight_layout()
plt.savefig(OUT_PNG, bbox_inches="tight")
plt.close()

print(f"Saved: {OUT_PNG}")
