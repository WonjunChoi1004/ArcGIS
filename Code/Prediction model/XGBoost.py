import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from pathlib import Path
PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / ".project-root").exists()), Path(__file__).resolve().parent)
from collections import OrderedDict

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, confusion_matrix, roc_curve
)

from xgboost import XGBClassifier

# -------- Paths --------
DATA_PATH = f"{PROJECT_ROOT}/LandslideData/FinalData/All_Combined_Balanced_EqualCounts_with_soil_depth_elev_slope.csv"
ROOT_OUT  = Path(f"{PROJECT_ROOT}/LandslideData/FinalData/ML_Outputs")
XGB_DIR   = ROOT_OUT / "XGBoost"
(XGB_DIR / "_split_indices").mkdir(parents=True, exist_ok=True)

# -------- Config --------
RANDOM_STATE    = 42
TEST_SIZE       = 0.2
MODEL_NAME      = "XGB"
N_ESTIMATORS    = 400
LEARNING_RATE   = 0.05
MAX_DEPTH       = 3
SUBSAMPLE       = 0.8
COLSAMPLE_BT    = 0.8
REG_LAMBDA      = 1.0
MAX_WATERFALLS  = 3
SHAP_SAMPLE_MAX = 2000

RAIN    = ['Max_Rainfall_30day', 'Max_Rainfall_3day', 'R7d', 'R30d']
TERRAIN = ["Elevation_m","Slope_deg"]
SOIL    = ["Soil_Depth_Deep200_Flag"]

FEATURE_SETS = OrderedDict({
    "F0": RAIN,
    "F1": RAIN + TERRAIN,
    "F2": RAIN + TERRAIN + SOIL
})

# -------- Helpers --------
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

def plot_feature_importance(model, feat_names, out_png, title):
    imp = model.feature_importances_
    order = np.argsort(imp)[::-1]
    names_sorted = np.array(feat_names)[order]
    vals_sorted  = imp[order]
    plt.figure(figsize=(7, max(3, 0.35*len(feat_names)+1)))
    plt.barh(names_sorted[::-1], vals_sorted[::-1])
    plt.title(title); plt.xlabel("Gain importance")
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()

def normalize_shap_values(sv_raw, positive_class_index=1):
    arr = np.array(sv_raw)
    if arr.ndim == 3:
        cls = positive_class_index if arr.shape[2] > positive_class_index else 0
        arr = arr[:, :, cls]
    elif isinstance(sv_raw, list):
        arr = np.array(sv_raw[positive_class_index if len(sv_raw)>positive_class_index else 0])
    return arr

def normalize_expected_value(exp_val, positive_class_index=1):
    if isinstance(exp_val, (list, tuple, np.ndarray)):
        ev = np.atleast_1d(exp_val)
        idx = positive_class_index if ev.size > positive_class_index else 0
        return float(ev[idx])
    return float(exp_val)

# -------- Load & split --------
df = pd.read_csv(DATA_PATH)
y  = df["IsLandslide"].astype(int)
X_superset = df[FEATURE_SETS["F2"]]

X_train_sup, X_test_sup, y_train, y_test = train_test_split(
    X_superset, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

pd.Series(X_train_sup.index, name="train_idx").to_csv(XGB_DIR/"_split_indices"/"train_indices.csv", index=False)
pd.Series(X_test_sup.index,  name="test_idx").to_csv(XGB_DIR/"_split_indices"/"test_indices.csv",  index=False)

# -------- Train/Eval --------
all_metrics = []
imputer = SimpleImputer(strategy="median")

for fs, cols in FEATURE_SETS.items():
    fs_dir = XGB_DIR / fs
    fs_dir.mkdir(parents=True, exist_ok=True)

    X_train_raw = X_train_sup[cols]
    X_test_raw  = X_test_sup[cols]
    X_train = imputer.fit_transform(X_train_raw)
    X_test  = imputer.transform(X_test_raw)

    xgb = XGBClassifier(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BT,
        reg_lambda=REG_LAMBDA,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="auto"
    )
    xgb.fit(X_train, y_train)

    y_prob = xgb.predict_proba(X_test)[:,1]
    y_pred = (y_prob >= 0.5).astype(int)

    m = compute_metrics(y_test, y_pred, y_prob)
    m.update({"Model": MODEL_NAME, "FeatureSet": fs})
    all_metrics.append(m)

    preds = pd.DataFrame({
        "index": X_test_sup.index, "y_true": y_test.values, "y_pred": y_pred, "y_proba": y_prob
    }).sort_values("index")
    preds.to_csv(fs_dir/f"test_predictions_{MODEL_NAME}_{fs}.csv", index=False)

    plot_cm(confusion_matrix(y_test, y_pred), fs_dir/f"cmatrix_{MODEL_NAME}_{fs}.png",
            f"Confusion Matrix - {MODEL_NAME} ({fs})")
    plot_roc(y_test, y_prob, fs_dir/f"roc_curve_{MODEL_NAME}_{fs}.png",
             f"{MODEL_NAME}-{fs} (AUC={m['ROC_AUC']:.3f})")

    plot_feature_importance(xgb, cols, fs_dir/f"feature_importance_{MODEL_NAME}_{fs}.png",
                            f"Feature Importance - {MODEL_NAME} ({fs})")

    # -------- SHAP --------
    X_shap = X_test.copy()
    if X_shap.shape[0] > SHAP_SAMPLE_MAX:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(X_shap.shape[0], SHAP_SAMPLE_MAX, replace=False)
        X_shap = X_shap[idx]

    explainer = shap.TreeExplainer(xgb)
    sv_raw = explainer.shap_values(X_shap)
    shap_values = normalize_shap_values(sv_raw, positive_class_index=1)
    feat_df = pd.DataFrame(X_shap, columns=cols)

    plt.figure()
    shap.summary_plot(shap_values, feat_df, plot_type="bar", show=False)
    plt.title(f"SHAP Summary (bar) - {MODEL_NAME} ({fs})")
    plt.tight_layout(); plt.savefig(fs_dir/f"shap_summary_bar_{MODEL_NAME}_{fs}.png", dpi=200); plt.close()

    expected_val = normalize_expected_value(explainer.expected_value, positive_class_index=1)
    top_idx = np.argsort(-y_prob)[:MAX_WATERFALLS]
    for k, ix in enumerate(top_idx, start=1):
        sv = shap_values[ix]
        shap.plots._waterfall.waterfall_legacy(expected_val, sv, feature_names=cols, show=False)
        plt.title(f"SHAP Waterfall - {MODEL_NAME} ({fs}) sample{k}")
        plt.tight_layout(); plt.savefig(fs_dir/f"shap_waterfall_{MODEL_NAME}_{fs}_sample{k}.png", dpi=200); plt.close()

    joblib.dump(xgb, fs_dir/f"model_{MODEL_NAME}_{fs}.joblib")
    joblib.dump(imputer, fs_dir/f"preprocess_imputer_{MODEL_NAME}_{fs}.joblib")

# -------- Results + Combined ROC --------
results_df = pd.DataFrame(all_metrics)[["Model","FeatureSet","Accuracy","Precision","Recall","F1","ROC_AUC","Brier"]]
results_df = results_df.sort_values("FeatureSet")
results_df.to_csv(XGB_DIR/"results_main_V0.csv", index=False)

plt.figure()
for fs in FEATURE_SETS.keys():
    preds = pd.read_csv(XGB_DIR / fs / f"test_predictions_{MODEL_NAME}_{fs}.csv")
    y_true = preds["y_true"]; y_prob = preds["y_proba"]
    fpr, tpr, _ = roc_curve(y_true, y_prob); auc = roc_auc_score(y_true, y_prob)
    plt.plot(fpr, tpr, label=f"{MODEL_NAME}-{fs} (AUC={auc:.3f})")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curves - XGBoost (F0 vs F1 vs F2)")
plt.legend(); plt.tight_layout(); plt.savefig(XGB_DIR/"roc_curves_all.png", dpi=200); plt.close()
