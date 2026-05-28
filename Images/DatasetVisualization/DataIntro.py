# plot_prism_csv_square.py

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------- Settings --------
OUT_DIR = Path("prism_sep_2024_buncombe_800m")  # same folder you used
CSV_PATH = OUT_DIR / "buncombe_prism_daily_ppt_sep_2024.csv"
OUT_PNG = OUT_DIR / "buncombe_prism_daily_ppt_sep_2024_top10pct_shaded_square.png"

COUNTY_NAME = "Buncombe"
TOP_PCT = 0.10  # top 10%

# -------- Load CSV --------
df = pd.read_csv(CSV_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# -------- Compute top 10% threshold + regime --------
threshold = float(np.nanquantile(df["ppt_mm"].values, 1 - TOP_PCT))
df["regime"] = df["ppt_mm"] >= threshold

# -------- Square plot --------
fig, ax = plt.subplots(figsize=(21, 7), dpi=250)  # <-- square figure

ax.plot(df["date"], df["ppt_mm"], linewidth=2.0)

# Shade the top 10% days
for dts in df.loc[df["regime"], "date"]:
    ax.axvspan(dts, dts + pd.Timedelta(days=1), alpha=0.25)

ax.set_ylabel("Daily precipitation (mm)")
ax.set_xlabel("")
ax.set_title(
    f"{COUNTY_NAME} County Mean Daily Rainfall (PRISM) — Sep 2024\n"
    f"Shaded = top 10% rainfall days (≥ {threshold:.2f} mm/day)"
)
ax.grid(True, alpha=0.3)

# Keep plot region square-ish and remove extra whitespace
plt.tight_layout()
plt.savefig(OUT_PNG, bbox_inches="tight")
plt.close(fig)

print(f"Saved square plot: {OUT_PNG}")
print(f"90th percentile threshold: {threshold:.2f} mm/day")
