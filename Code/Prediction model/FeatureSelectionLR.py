import itertools
import json
import time
from pathlib import Path
PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / ".project-root").exists()), Path(__file__).resolve().parent)

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **k: x  # fallback if tqdm not installed

# ---------- Paths ----------
DATA_PATH = f"{PROJECT_ROOT}/LandslideData/FinalData/All_Combined_Balanced_EqualCounts_with_soil_depth_elev_slope.csv"
ROOT_OUT  = Path(f"{PROJECT_ROOT}/LandslideData/FinalData/ML_Outputs")
SUBSEL_DIR = ROOT_OUT / "LR_SubsetSearch_RAIN"
INF_DIR = ROOT_OUT / "Inference"
SUBSEL_DIR.mkdir(parents=True, exist_ok=True)
INF_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Config ----------
TARGET = "IsLandslide"
RANDOM_STATE = 42
N_SPLITS = 5
MODEL_NAME = "LR"

RAIN = ["Max_Rainfall_30day","Max_Rainfall_3day","R1d","R3d","R7d","R30d",]

def lr_pipe():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, solver="lbfgs", random_state=RANDOM_STATE))
    ])

# ---------- Load ----------
df = pd.read_csv(DATA_PATH)
if TARGET not in df.columns:
    raise ValueError(f"Missing target column: {TARGET}")
for c in RAIN:
    if c not in df.columns:
        raise ValueError(f"Missing feature column: {c}")

y = df[TARGET].astype(int).values
X_all = df[RAIN].copy()

# ---------- Helpers ----------
def all_nonempty_subsets(features):
    n = len(features)
    for k in range(1, n + 1):
        for comb in itertools.combinations(features, k):
            yield list(comb)

def evaluate_subset(X, y, cols):
    Xsub = X[cols].values
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    pipe = lr_pipe()
    oof_proba = cross_val_predict(pipe, Xsub, y, cv=cv, method="predict_proba")[:, 1]
    oof_pred  = (oof_proba >= 0.5).astype(int)
    return {
        "k": len(cols),
        "features": cols,
        "Accuracy": accuracy_score(y, oof_pred),
        "Precision": precision_score(y, oof_pred, zero_division=0),
        "Recall": recall_score(y, oof_pred, zero_division=0),
        "F1": f1_score(y, oof_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y, oof_proba),
        "Brier": brier_score_loss(y, oof_proba),
    }

def tie_break_key(row):
    return (row["Accuracy"], -row["k"], row["ROC_AUC"], -row["Brier"])

# ---------- Exhaustive search over ALL 2^8-1 = 255 subsets ----------
subsets = list(all_nonempty_subsets(RAIN))
total_subsets = len(subsets)

t0 = time.time()
rows = []
for cols in tqdm(subsets, desc="Searching subsets", total=total_subsets):
    t_subset = time.time()
    rec = {"k": len(cols), "features": cols}
    try:
        rec.update(evaluate_subset(X_all, y, cols))
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
    by=["Accuracy","k","ROC_AUC","Brier"],
    ascending=[False, True, False, True]
).reset_index(drop=True)

# Save leaderboard + timing
res_ok.to_csv(SUBSEL_DIR / "subset_search_leaderboard.csv", index=False)
res.to_csv(SUBSEL_DIR / "subset_search_all_rows.csv", index=False)

# ---------- Best subset ----------
best_idx = max(res_ok.index, key=lambda i: tie_break_key(res_ok.loc[i]))
best = res_ok.loc[best_idx]
BEST_FEATS = list(best["features"]) if isinstance(best["features"], (list, tuple)) else eval(best["features"])

with open(SUBSEL_DIR / "best_subset.json", "w") as f:
    json.dump({
        "best_features": BEST_FEATS,
        "metrics": {m: float(best[m]) for m in ["Accuracy","Precision","Recall","F1","ROC_AUC","Brier"]},
        "k": int(best["k"]),
        "timing": {"total_subsets": total_subsets, "elapsed_sec": elapsed, "mean_subset_sec": float(res["TimeSec"].mean())},
        "config": {"N_SPLITS": N_SPLITS, "RANDOM_STATE": RANDOM_STATE}
    }, f, indent=2)

print(f"\nBest subset (k={len(BEST_FEATS)}): {BEST_FEATS}")
print(f"Accuracy={best['Accuracy']:.4f} | ROC_AUC={best['ROC_AUC']:.4f} | Brier={best['Brier']:.4f}")
print(f"Total subsets tried: {total_subsets} | Elapsed: {elapsed}s")

# ---------- Final fit on full data (best subset) ----------
X_best = X_all[BEST_FEATS].values
pipe = lr_pipe()
pipe.fit(X_best, y)
joblib.dump(pipe, SUBSEL_DIR / f"model_{MODEL_NAME}_best_subset.joblib")

proba = pipe.predict_proba(X_best)[:,1]
pred  = (proba >= 0.5).astype(int)
pd.DataFrame({"y_true": y, "y_pred": pred, "y_proba": proba}).to_csv(
    SUBSEL_DIR / "full_fit_predictions_best_subset.csv", index=False
)

# ---------- HC1-robust inference on standardized X (best subset) ----------
imp = SimpleImputer(strategy="median")
sc  = StandardScaler()
X_imp = imp.fit_transform(X_all[BEST_FEATS].values)
X_std = sc.fit_transform(X_imp)

X_sm = sm.add_constant(X_std)
glm  = sm.GLM(y, X_sm, family=sm.families.Binomial())
res_glm  = glm.fit(cov_type="HC1")

coefs = np.asarray(res_glm.params)
conf_arr = np.asarray(res_glm.conf_int())
feat_names = ["const"] + BEST_FEATS
conf_df = pd.DataFrame(conf_arr, columns=["2.5%","97.5%"])
if conf_df.shape[0] != len(feat_names):
    conf_df = pd.DataFrame(conf_arr[:len(feat_names)], columns=["2.5%","97.5%"])
conf_df.index = feat_names

se = np.asarray(res_glm.bse)
zvals = np.asarray(res_glm.tvalues)
pvals = np.asarray(res_glm.pvalues)
coef_df = pd.DataFrame({
    "feature": feat_names,
    "coef": coefs,
    "se": se,
    "z": zvals,
    "pval": pvals,
    "OR": np.exp(coefs),
    "CI_lower": np.exp(conf_df["2.5%"].values),
    "CI_upper": np.exp(conf_df["97.5%"].values),
})

const_row = coef_df.iloc[[0]]
body = coef_df.iloc[1:].copy()
body = body.reindex(body["z"].abs().sort_values(ascending=False).index)
out_df = pd.concat([const_row, body], ignore_index=True)

inf_dir = INF_DIR / "LR_Subset_Best_RAIN"
inf_dir.mkdir(parents=True, exist_ok=True)
out_df.to_csv(inf_dir / "logreg_coefficients_best_subset.csv", index=False)
joblib.dump(imp, inf_dir / "preprocess_imputer.joblib")
joblib.dump(sc,  inf_dir / "preprocess_scaler.joblib")

# ---------- Compact main-table row for the best subset ----------
row = {
    "Model": "LR",
    "FeatureSet": f"RAIN_best_k{len(BEST_FEATS)}",
    "Accuracy": accuracy_score(y, pred),
    "Precision": precision_score(y, pred, zero_division=0),
    "Recall": recall_score(y, pred, zero_division=0),
    "F1": f1_score(y, pred, zero_division=0),
    "ROC_AUC": roc_auc_score(y, proba),
    "Brier": brier_score_loss(y, proba)
}
pd.DataFrame([row]).to_csv(ROOT_OUT / "results_main_V0.csv", index=False)
