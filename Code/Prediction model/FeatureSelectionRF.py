import itertools
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **k: x

# ---------- Paths ----------
DATA_PATH = "/Users/wonjunchoi/PycharmProjects/ArcGIS/LandslideData/FinalData/All_Combined_Balanced_EqualCounts_with_soil_depth_elev_slope.csv"
ROOT_OUT  = Path("/Users/wonjunchoi/PycharmProjects/ArcGIS/LandslideData/FinalData/ML_Outputs")
SUBSEL_DIR = ROOT_OUT / "RF_SubsetSearch_RAIN_with_TERRAIN_SOIL"
SUBSEL_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Config ----------
TARGET = "IsLandslide"
RANDOM_STATE = 42
N_SPLITS = 5
MODEL_NAME = "RF"

RAIN    = ["Max_Rainfall_30day","Max_Rainfall_3day","R1d","R3d","R7d","R30d",]
TERRAIN = ["Elevation_m","Slope_deg"]
SOIL    = ["Soil_Depth_Deep200_Flag"]

def rf_pipe():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=500,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])

# ---------- Load ----------
df = pd.read_csv(DATA_PATH)
if TARGET not in df.columns:
    raise ValueError(f"Missing target column: {TARGET}")
for c in RAIN + TERRAIN + SOIL:
    if c not in df.columns:
        raise ValueError(f"Missing feature column: {c}")

y = df[TARGET].astype(int).values
X_all = df[RAIN + TERRAIN + SOIL].copy()

# ---------- Helpers ----------
def all_nonempty_subsets(features):
    n = len(features)
    for k in range(1, n + 1):
        for comb in itertools.combinations(features, k):
            yield list(comb)

def evaluate_subset(X, y, rain_cols):
    cols = rain_cols + TERRAIN + SOIL
    Xsub = X[cols].values
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    pipe = rf_pipe()
    proba = cross_val_predict(pipe, Xsub, y, cv=cv, method="predict_proba")[:, 1]
    pred  = (proba >= 0.5).astype(int)
    return {
        "k_rain": len(rain_cols),
        "rain_features": rain_cols,
        "all_features": cols,
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y, proba),
        "Brier": brier_score_loss(y, proba),
    }

def tie_break_key(row):
    return (row["Accuracy"], -row["k_rain"], row["ROC_AUC"], -row["Brier"])

# ---------- Exhaustive search over all rain subsets (terrain+soil fixed) ----------
rain_subsets = list(all_nonempty_subsets(RAIN))
t0 = time.time()
rows = []
for rain_cols in tqdm(rain_subsets, desc="RF subset search (Rain only)", total=len(rain_subsets)):
    t_subset = time.time()
    rec = {"k_rain": len(rain_cols), "rain_features": rain_cols}
    try:
        rec.update(evaluate_subset(X_all, y, rain_cols))
        rec["error"] = ""
    except Exception as e:
        rec.update({
            "Accuracy": np.nan, "Precision": np.nan, "Recall": np.nan,
            "F1": np.nan, "ROC_AUC": np.nan, "Brier": np.nan,
            "error": str(e)
        })
    rec["TimeSec"] = round(time.time() - t_subset, 4)
    rows.append(rec)

elapsed = round(time.time() - t0, 2)

res = pd.DataFrame(rows)
res_ok = res[(~res["Accuracy"].isna()) & (res["error"] == "")]
res_ok = res_ok.sort_values(
    by=["Accuracy","k_rain","ROC_AUC","Brier"],
    ascending=[False, True, False, True]
).reset_index(drop=True)

# Save outputs
res_ok.to_csv(SUBSEL_DIR / "rf_subset_search_leaderboard.csv", index=False)
res.to_csv(SUBSEL_DIR / "rf_subset_search_all_rows.csv", index=False)

# ---------- Best subset ----------
best_idx = max(res_ok.index, key=lambda i: tie_break_key(res_ok.loc[i]))
best = res_ok.loc[best_idx]
BEST_RAIN = list(best["rain_features"]) if isinstance(best["rain_features"], (list, tuple)) else eval(best["rain_features"])
BEST_ALL  = BEST_RAIN + TERRAIN + SOIL

with open(SUBSEL_DIR / "rf_best_subset.json", "w") as f:
    json.dump({
        "best_rain_features": BEST_RAIN,
        "always_included": {"terrain": TERRAIN, "soil": SOIL},
        "final_feature_list": BEST_ALL,
        "metrics": {m: float(best[m]) for m in ["Accuracy","Precision","Recall","F1","ROC_AUC","Brier"]},
        "k_rain": int(best["k_rain"]),
        "timing": {"total_rain_subsets": len(rain_subsets), "elapsed_sec": elapsed, "mean_subset_sec": float(res["TimeSec"].mean())},
        "config": {"N_SPLITS": N_SPLITS, "RANDOM_STATE": RANDOM_STATE}
    }, f, indent=2)

print(f"\nBest RF subset (Rain k={len(BEST_RAIN)}): {BEST_RAIN}")
print(f"Final features (Rain + Terrain + Soil): {BEST_ALL}")
print(f"Accuracy={best['Accuracy']:.4f} | ROC_AUC={best['ROC_AUC']:.4f} | Brier={best['Brier']:.4f}")
print(f"Total rain subsets: {len(rain_subsets)} | Elapsed: {elapsed}s")

# ---------- Final fit on full data (best Rain subset + Terrain + Soil) ----------
X_best = X_all[BEST_ALL].values
pipe = rf_pipe()
pipe.fit(X_best, y)
joblib.dump(pipe, SUBSEL_DIR / f"model_{MODEL_NAME}_best_subset.joblib")

proba = pipe.predict_proba(X_best)[:,1]
pred  = (proba >= 0.5).astype(int)
pd.DataFrame({"y_true": y, "y_pred": pred, "y_proba": proba}).to_csv(
    SUBSEL_DIR / "rf_full_fit_predictions_best_subset.csv", index=False
)
