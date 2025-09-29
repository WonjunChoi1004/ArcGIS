import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from pathlib import Path
from collections import OrderedDict

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, confusion_matrix, roc_curve
)

# ---------- Paths ----------
DATA_PATH = "/Users/wonjunchoi/PycharmProjects/ArcGIS/LandslideData/FinalData/All_Combined_Balanced_EqualCounts_with_soil_depth_elev_slope.csv"
ROOT_OUT = Path("/Users/wonjunchoi/PycharmProjects/ArcGIS/LandslideData/FinalData/ML_Outputs")
LR_DIR   = ROOT_OUT / "LogisticRegression"
INF_DIR  = ROOT_OUT / "Inference"
(LR_DIR / "_split_indices").mkdir(parents=True, exist_ok=True)
INF_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Config ----------
RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_NAME = "LogReg"
# 'Avg_Rainfall_30day', 'Max_Rainfall_30day', 'R3d', 'R7d'
RAIN    = ['Max_Rainfall_30day', 'R3d', 'R7d', 'R30d']
TERRAIN = ["Elevation_m","Slope_deg"]
SOIL    = ["Soil_Depth_Deep200_Flag"]

FEATURE_SETS = OrderedDict({
    "F0": RAIN,
    "F1": RAIN + TERRAIN,
    "F2": RAIN + TERRAIN + SOIL
})
ORDERED_FS = ["F0","F1","F2"]

# ---------- Helpers ----------
def lr_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, solver="lbfgs"))
    ])

def compute_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "ROC_AUC": roc_auc_score(y_true, y_prob),
        "Brier": brier_score_loss(y_true, y_prob)
    }

def plot_cm(cm, out_png, title):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Control","Landslide"], yticklabels=["Control","Landslide"])
    plt.title(title); plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()

def plot_roc(y_true, y_prob, out_png, label):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=label); plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve"); plt.legend()
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()

def run_inference_and_save(X_train_sup, y_train, fs_name, cols):
    subdir = INF_DIR / f"LR_{fs_name}"
    subdir.mkdir(parents=True, exist_ok=True)

    imp = SimpleImputer(strategy="median")
    scl = StandardScaler()

    X_train_imp = imp.fit_transform(X_train_sup[cols])
    X_train_s   = scl.fit_transform(X_train_imp)

    X_sm = sm.add_constant(X_train_s)
    glm  = sm.GLM(y_train, X_sm, family=sm.families.Binomial())
    res  = glm.fit(cov_type="HC1")

    coefs = res.params
    conf  = res.conf_int()
    conf.columns = ["2.5%","97.5%"]

    feat_names = ["const"] + cols
    coef_df = pd.DataFrame({
        "feature": feat_names,
        "coef": coefs,
        "se": res.bse,
        "z": res.tvalues,
        "pval": res.pvalues,
        "OR": np.exp(coefs),
        "CI_lower": np.exp(conf["2.5%"]),
        "CI_upper": np.exp(conf["97.5%"])
    })

    # sort: const first, then by |z| desc
    const_row = coef_df.iloc[[0]]
    body = coef_df.iloc[1:].copy()
    body = body.reindex(body["z"].abs().sort_values(ascending=False).index)
    out_df = pd.concat([const_row, body], ignore_index=True)

    out_df.to_csv(subdir / f"logreg_coefficients_{fs_name}.csv", index=False)
    joblib.dump(imp, subdir / "preprocess_imputer.joblib")
    joblib.dump(scl, subdir / "preprocess_scaler.joblib")

# ---------- Load ----------
df = pd.read_csv(DATA_PATH)
y  = df["IsLandslide"].astype(int)

# Superset split for consistency
X_superset = df[FEATURE_SETS["F2"]]
X_train_sup, X_test_sup, y_train, y_test = train_test_split(
    X_superset, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

pd.Series(X_train_sup.index, name="train_idx").to_csv(LR_DIR/"_split_indices"/"train_indices.csv", index=False)
pd.Series(X_test_sup.index,  name="test_idx").to_csv(LR_DIR/"_split_indices"/"test_indices.csv",  index=False)

# ---------- Train/Eval per FS ----------
all_rows = []
for fs in ORDERED_FS:
    cols  = FEATURE_SETS[fs]
    fsdir = LR_DIR / fs
    fsdir.mkdir(parents=True, exist_ok=True)

    X_train = X_train_sup[cols]
    X_test  = X_test_sup[cols]

    pipe = lr_pipeline()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:,1]

    m = compute_metrics(y_test, y_pred, y_prob)
    m.update({"Model": MODEL_NAME, "FeatureSet": fs})
    all_rows.append(m)

    preds = pd.DataFrame({
        "index": X_test.index, "y_true": y_test.values,
        "y_pred": y_pred, "y_proba": y_prob
    }).sort_values("index")
    preds.to_csv(fsdir / f"test_predictions_{MODEL_NAME}_{fs}.csv", index=False)

    plot_cm(confusion_matrix(y_test, y_pred),
            fsdir / f"cmatrix_{MODEL_NAME}_{fs}.png",
            f"Confusion Matrix - {MODEL_NAME} ({fs})")
    plot_roc(y_test, y_prob,
             fsdir / f"roc_curve_{MODEL_NAME}_{fs}.png",
             f"{MODEL_NAME}-{fs} (AUC={m['ROC_AUC']:.3f})")

    joblib.dump(pipe, fsdir / f"model_{MODEL_NAME}_pipe_{fs}.joblib")

# ---------- Save results + combined ROC ----------
res_df = pd.DataFrame(all_rows)[["Model","FeatureSet","Accuracy","Precision","Recall","F1","ROC_AUC","Brier"]]
res_df["FeatureSet"] = pd.Categorical(res_df["FeatureSet"], categories=ORDERED_FS, ordered=True)
res_df = res_df.sort_values(["Model","FeatureSet"]).reset_index(drop=True)
res_df.to_csv(LR_DIR / "results_main_V0.csv", index=False)

plt.figure()
for fs in ORDERED_FS:
    preds = pd.read_csv(LR_DIR / fs / f"test_predictions_{MODEL_NAME}_{fs}.csv")
    y_true = preds["y_true"]; y_prob = preds["y_proba"]
    fpr, tpr, _ = roc_curve(y_true, y_prob); auc = roc_auc_score(y_true, y_prob)
    plt.plot(fpr, tpr, label=f"{MODEL_NAME}-{fs} (AUC={auc:.3f})")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Logistic Regression (F0 vs F1 vs F2)")
plt.legend(); plt.tight_layout(); plt.savefig(LR_DIR / "roc_curves_all.png", dpi=200); plt.close()

# ---------- Inference for all FS (GLM with robust SE) ----------
for fs in ORDERED_FS:
    run_inference_and_save(X_train_sup, y_train, fs, FEATURE_SETS[fs])

# ---------- Manifest ----------
manifest = {
    "data_path": DATA_PATH,
    "output_root": str(ROOT_OUT),
    "split": {"test_size": TEST_SIZE, "random_state": RANDOM_STATE, "stratified": True},
    "feature_sets": FEATURE_SETS,
    "models": [{"name": MODEL_NAME, "solver": "lbfgs", "max_iter": 2000}],
    "artifacts": {
        "V0_results_csv": str(LR_DIR / "results_main_V0.csv"),
        "combined_roc": str(LR_DIR / "roc_curves_all.png"),
        "per_set_folder": "ML_Outputs/LogisticRegression/F*/",
        "inference_tables": {
            "F0": "ML_Outputs/Inference/LR_F0/logreg_coefficients_F0.csv",
            "F1": "ML_Outputs/Inference/LR_F1/logreg_coefficients_F1.csv",
            "F2": "ML_Outputs/Inference/LR_F2/logreg_coefficients_F2.csv"
        }
    }
}
with open(ROOT_OUT / "run_manifest_lr_v0.json","w") as f:
    json.dump(manifest, f, indent=2)
