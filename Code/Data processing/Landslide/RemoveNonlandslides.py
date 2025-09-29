# Downsample the majority class in All_Combined.csv so
# non-landslides match landslides (randomly, reproducible).

import pandas as pd
from pathlib import Path

DATA_DIR = Path("/Users/wonjunchoi/PycharmProjects/ArcGIS/LandslideData")
IN_PATH  = DATA_DIR / "All_Combined.csv"
OUT_PATH = DATA_DIR / "All_Combined_Balanced_EqualCounts.csv"
SEED     = 42  # change if you want a different random draw

df = pd.read_csv(IN_PATH)

# Ensure label is integer 0/1
if "IsLandslide" not in df.columns:
    raise ValueError("Expected 'IsLandslide' column in All_Combined.csv")
df["IsLandslide"] = df["IsLandslide"].astype(int)

# Split classes
pos = df[df["IsLandslide"] == 1].copy()  # landslides (e.g., 302)
neg = df[df["IsLandslide"] == 0].copy()  # non-landslides (e.g., 398)

# Target = minority size → downsample the larger class
target = min(len(pos), len(neg))

pos_bal = pos.sample(n=target, random_state=SEED) if len(pos) > target else pos
neg_bal = neg.sample(n=target, random_state=SEED) if len(neg) > target else neg

balanced = pd.concat([pos_bal, neg_bal], ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

balanced.to_csv(OUT_PATH, index=False)
print(f"✅ Saved balanced dataset → {OUT_PATH}")
print(f"Counts → landslides: {(balanced['IsLandslide']==1).sum()}, non-landslides: {(balanced['IsLandslide']==0).sum()}")
