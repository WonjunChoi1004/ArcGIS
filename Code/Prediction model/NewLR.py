import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / ".project-root").exists()), Path(__file__).resolve().parent)

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

# ---------- Paths ----------
DATA_PATH = f"{PROJECT_ROOT}/LandslideData/FinalData/All_Combined_Balanced_EqualCounts_with_soil_depth_elev_slope.csv"
ROOT_OUT  = Path(f"{PROJECT_ROOT}/LandslideData/FinalData/ML_Outputs")
LR_DIR    = ROOT_OUT / "LogisticRegression"
SPLIT_DIR = LR_DIR / "_split_indices"

OUT_PNG   = LR_DIR / "roc_curves_Terrain_vs_All.png"

# ---------- Config ----------
RANDOM_STATE = 42  # (not used here since we reuse saved split)
TERRAIN = ["Elevation_m", "Slope_deg"]
RAIN    = ['Max_Rainfall_30day', 'R3d', 'R7d', 'R30d']
SOIL    = ["Soil_Depth_Deep200_Flag"]

FEATURE_SETS = {
    "Terrain": TERRAIN,                  # replaces "F0"
    "All": RAIN + TERRAIN + SOIL         # replaces "F2"
}

def lr_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, solver="lbfgs"))
    ])

# ---------- Load data ----------
df = pd.read_csv(DATA_PATH)
y  = df["IsLandslide"].astype(int)

# ---------- Reuse the SAME split you already saved ----------
train_idx = pd.read_csv(SPLIT_DIR / "train_indices.csv")["train_idx"].astype(int).values
test_idx  = pd.read_csv(SPLIT_DIR / "test_indices.csv")["test_idx"].astype(int).values

y_train = y.loc[train_idx]
y_test  = y.loc[test_idx]

# ---------- Train + ROC plot ----------
plt.figure()

for label, cols in FEATURE_SETS.items():
    X_train = df.loc[train_idx, cols]
    X_test  = df.loc[test_idx, cols]

    pipe = lr_pipeline()
    pipe.fit(X_train, y_train)

    y_prob = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    plt.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Logistic Regression (Terrain vs All)")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300)
plt.close()

print(f"Saved: {OUT_PNG}")
