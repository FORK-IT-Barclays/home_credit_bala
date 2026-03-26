"""
VECTOR — Pre-Delinquency Intervention Engine
Step 2: Dual-Expert Model Training

Financial Historian  → XGBoost (static structural risk)
Behavioral Specialist → LightGBM (dynamic behavioral signals)
Ensemble             → Weighted stacking
Explainability       → SHAP values for every prediction
"""

import os, json, warnings
import numpy as np
import pandas as pd
import shap
import joblib
import xgboost as xgb
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
warnings.filterwarnings('ignore')

DATA_DIR  = Path("/home/balahero03/credit")
PIPE_DIR  = DATA_DIR / "pipeline"
OUT_DIR   = PIPE_DIR / "output"
MODEL_DIR = PIPE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("VECTOR Dual-Expert Model Training")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# 1. LOAD FEATURES
# ─────────────────────────────────────────────────────────────
print("\n[1/6] Loading engineered features...")
train = pd.read_parquet(OUT_DIR / "train_features.parquet")
test  = pd.read_parquet(OUT_DIR / "test_features.parquet")

TARGET = 'TARGET'
ID_COL = 'SK_ID_CURR'
y      = train[TARGET].astype(int)
X      = train.drop(columns=[TARGET, ID_COL])
X_test = test.drop(columns=[ID_COL])
X_test = X_test.reindex(columns=X.columns, fill_value=np.nan)

print(f"  Features: {X.shape[1]} | Train: {len(X):,} | Test: {len(X_test):,}")
print(f"  Default rate: {y.mean():.2%}  (imbalanced — using scale_pos_weight)")

# ─────────────────────────────────────────────────────────────
# 2. FEATURE GROUPS
#    Financial Historian = stable/historical features (XGBoost)
#    Behavioral Specialist = dynamic/velocity features (LightGBM)
# ─────────────────────────────────────────────────────────────
HISTORIAN_KEYWORDS = [
    'EXT_SOURCE','INCOME','CREDIT_TO','ANNUITY','AGE','DAYS_BIRTH',
    'BUREAU_CREDIT','BUREAU_SUM','BUREAU_MAX','BUREAU_AVG',
    'REGION','EDUCATION','ASSET','DOCS','CODE_GENDER',
    'NAME_INCOME','NAME_FAMILY','CHILDREN','SOCIAL_CIRCLE'
]

BEHAVIORAL_KEYWORDS = [
    'INST_','CC_','POS_','VECTOR_','VELOCITY','ACCELERATION',
    'DRIFT','DELAY','MOMENTUM','EXHAUST','ZONE','STRESS',
    'DPD','OVERDUE','LATE','RECENT','ERRATIC','JUGGLING',
    'BUREAU_OVERDUE','BUREAU_RECENT','BUREAU_BAD'
]

def select_features(cols, keywords):
    return [c for c in cols if any(k in c for k in keywords)]

historian_feats  = select_features(X.columns, HISTORIAN_KEYWORDS)
behavioral_feats = select_features(X.columns, BEHAVIORAL_KEYWORDS)
# Add remaining features to historian as fallback
covered = set(historian_feats + behavioral_feats)
remaining = [c for c in X.columns if c not in covered]
historian_feats  = historian_feats  + remaining

historian_feats  = list(dict.fromkeys(historian_feats))   # deduplicate
behavioral_feats = list(dict.fromkeys(behavioral_feats))

print(f"\n  Financial Historian features:    {len(historian_feats)}")
print(f"  Behavioral Specialist features: {len(behavioral_feats)}")

# ─────────────────────────────────────────────────────────────
# 3. CROSS-VALIDATION SETUP
# ─────────────────────────────────────────────────────────────
N_FOLDS = 5
SKF = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
scale_pos_weight = (y == 0).sum() / (y == 1).sum()

# OOF predictions storage
oof_hist = np.zeros(len(X))
oof_beh  = np.zeros(len(X))
test_hist_preds = np.zeros(len(X_test))
test_beh_preds  = np.zeros(len(X_test))

hist_models = []
beh_models  = []
fold_scores = {'historian': [], 'behavioral': [], 'ensemble': []}

print(f"\n[2/6] Training Financial Historian (XGBoost) — {N_FOLDS}-fold CV...")
print("      Scope: Historical credit data, income, demographics\n")

XGB_PARAMS = {
    'n_estimators':      1000,
    'max_depth':         6,
    'learning_rate':     0.05,
    'subsample':         0.8,
    'colsample_bytree':  0.8,
    'min_child_weight':  50,
    'scale_pos_weight':  scale_pos_weight,
    'reg_alpha':         0.1,
    'reg_lambda':        1.0,
    'random_state':      42,
    'eval_metric':       'auc',
    'early_stopping_rounds': 50,
    'n_jobs':            -1,
    'use_label_encoder': False,
    'verbosity':         0,
}

for fold, (tr_idx, val_idx) in enumerate(SKF.split(X, y), 1):
    X_tr = X.iloc[tr_idx][historian_feats]
    X_vl = X.iloc[val_idx][historian_feats]
    y_tr, y_vl = y.iloc[tr_idx], y.iloc[val_idx]

    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_tr, y_tr,
              eval_set=[(X_vl, y_vl)],
              verbose=False)

    oof_hist[val_idx] = model.predict_proba(X_vl)[:, 1]
    test_hist_preds   += model.predict_proba(X_test[historian_feats])[:, 1] / N_FOLDS
    score = roc_auc_score(y_vl, oof_hist[val_idx])
    fold_scores['historian'].append(score)
    hist_models.append(model)
    print(f"  Fold {fold}: AUC = {score:.4f}")

print(f"\n  ► Historian CV AUC: {np.mean(fold_scores['historian']):.4f} ± {np.std(fold_scores['historian']):.4f}")

# ─────────────────────────────────────────────────────────────
# 4. BEHAVIORAL SPECIALIST (LightGBM)
# ─────────────────────────────────────────────────────────────
print(f"\n[3/6] Training Behavioral Specialist (LightGBM) — {N_FOLDS}-fold CV...")
print("      Scope: Payment velocity, DPD trends, credit exhaustion, VECTOR signals\n")

LGB_PARAMS = {
    'n_estimators':        1000,
    'max_depth':           7,
    'num_leaves':          63,
    'learning_rate':       0.05,
    'subsample':           0.8,
    'subsample_freq':      1,
    'colsample_bytree':    0.8,
    'min_child_samples':   50,
    'scale_pos_weight':    scale_pos_weight,
    'reg_alpha':           0.1,
    'reg_lambda':          1.0,
    'random_state':        42,
    'n_jobs':             -1,
    'verbose':            -1,
}

callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)]

for fold, (tr_idx, val_idx) in enumerate(SKF.split(X, y), 1):
    X_tr = X.iloc[tr_idx][behavioral_feats]
    X_vl = X.iloc[val_idx][behavioral_feats]
    y_tr, y_vl = y.iloc[tr_idx], y.iloc[val_idx]

    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(X_tr, y_tr,
              eval_set=[(X_vl, y_vl)],
              callbacks=callbacks)

    oof_beh[val_idx] = model.predict_proba(X_vl)[:, 1]
    test_beh_preds   += model.predict_proba(X_test[behavioral_feats])[:, 1] / N_FOLDS
    score = roc_auc_score(y_vl, oof_beh[val_idx])
    fold_scores['behavioral'].append(score)
    beh_models.append(model)
    print(f"  Fold {fold}: AUC = {score:.4f}")

print(f"\n  ► Behavioral CV AUC: {np.mean(fold_scores['behavioral']):.4f} ± {np.std(fold_scores['behavioral']):.4f}")

# ─────────────────────────────────────────────────────────────
# 5. ENSEMBLE — Weighted combination
# ─────────────────────────────────────────────────────────────
print("\n[4/6] Ensembling predictions...")

# Grid search best weights on OOF
best_w, best_auc = 0.5, 0.0
for w in np.arange(0.1, 0.91, 0.05):
    ens = w * oof_hist + (1 - w) * oof_beh
    auc = roc_auc_score(y, ens)
    if auc > best_auc:
        best_auc, best_w = auc, w

HIST_WEIGHT = round(best_w, 2)
BEH_WEIGHT  = round(1 - best_w, 2)
oof_ensemble = HIST_WEIGHT * oof_hist + BEH_WEIGHT * oof_beh
test_ensemble = HIST_WEIGHT * test_hist_preds + BEH_WEIGHT * test_beh_preds

ens_score = roc_auc_score(y, oof_ensemble)
fold_scores['ensemble'] = [ens_score]

print(f"\n  ► Optimal weights: Historian={HIST_WEIGHT}, Behavioral={BEH_WEIGHT}")
print(f"  ► Historian AUC:        {np.mean(fold_scores['historian']):.4f}")
print(f"  ► Behavioral AUC:       {np.mean(fold_scores['behavioral']):.4f}")
print(f"  ► Ensemble OOF AUC:     {ens_score:.4f}  ✅")

# ─────────────────────────────────────────────────────────────
# 6. SHAP EXPLAINABILITY
# ─────────────────────────────────────────────────────────────
print("\n[5/6] Computing SHAP values for explainability...")

# Use last fold models
sample_size = min(2000, len(X))
sample_idx  = np.random.choice(len(X), sample_size, replace=False)

# Historian SHAP
hist_explainer = shap.TreeExplainer(hist_models[-1])
X_sample_hist  = X.iloc[sample_idx][historian_feats]
shap_hist_vals = hist_explainer.shap_values(X_sample_hist)
if isinstance(shap_hist_vals, list):
    shap_hist_vals = shap_hist_vals[1]

# Behavioral SHAP
beh_explainer = shap.TreeExplainer(beh_models[-1])
X_sample_beh  = X.iloc[sample_idx][behavioral_feats]
shap_beh_vals = beh_explainer.shap_values(X_sample_beh)
if isinstance(shap_beh_vals, list):
    shap_beh_vals = shap_beh_vals[1]

# Save SHAP importance dicts
def shap_importance(shap_vals, feat_names):
    imp = pd.DataFrame({
        'feature':    feat_names,
        'shap_mean':  np.abs(shap_vals).mean(axis=0),
        'shap_std':   np.abs(shap_vals).std(axis=0),
    }).sort_values('shap_mean', ascending=False)
    return imp

hist_shap_imp = shap_importance(shap_hist_vals, historian_feats)
beh_shap_imp  = shap_importance(shap_beh_vals,  behavioral_feats)

hist_shap_imp.to_parquet(MODEL_DIR / "historian_shap_importance.parquet", index=False)
beh_shap_imp.to_parquet( MODEL_DIR / "behavioral_shap_importance.parquet", index=False)

print("\n  Top 10 Historian (XGBoost) features by SHAP:")
print(hist_shap_imp.head(10).to_string(index=False))
print("\n  Top 10 Behavioral (LightGBM) features by SHAP:")
print(beh_shap_imp.head(10).to_string(index=False))

# ─────────────────────────────────────────────────────────────
# 7. SAVE EVERYTHING
# ─────────────────────────────────────────────────────────────
print("\n[6/6] Saving models and predictions...")

# Save best fold models
joblib.dump(hist_models[-1], MODEL_DIR / "historian_xgb.pkl")
joblib.dump(beh_models[-1],  MODEL_DIR / "behavioral_lgbm.pkl")

# Save SHAP explainers
joblib.dump(hist_explainer, MODEL_DIR / "historian_shap_explainer.pkl")
joblib.dump(beh_explainer,  MODEL_DIR / "behavioral_shap_explainer.pkl")

# Save feature lists
meta = {
    'historian_features':  historian_feats,
    'behavioral_features': behavioral_feats,
    'ensemble_weights':    {'historian': HIST_WEIGHT, 'behavioral': BEH_WEIGHT},
    'oof_auc': {
        'historian':  round(float(np.mean(fold_scores['historian'])), 4),
        'behavioral': round(float(np.mean(fold_scores['behavioral'])), 4),
        'ensemble':   round(float(ens_score), 4),
    }
}
with open(MODEL_DIR / "model_meta.json", 'w') as f:
    json.dump(meta, f, indent=2)

# Save predictions
train_preds = pd.DataFrame({
    'SK_ID_CURR':       train[ID_COL].values,
    'TARGET':           y.values,
    'RISK_HIST':        oof_hist,
    'RISK_BEH':         oof_beh,
    'RISK_SCORE':       oof_ensemble,
    'VECTOR_RISK_ZONE': train.get('VECTOR_RISK_ZONE', pd.Series(0, index=train.index)).values,
    'VECTOR_VELOCITY':  train.get('VECTOR_RISK_VELOCITY', pd.Series(0, index=train.index)).values,
    'VECTOR_ACCEL':     train.get('VECTOR_RISK_ACCELERATION', pd.Series(0, index=train.index)).values,
})

# Assign risk tier
def assign_tier(score):
    if score >= 0.6:  return 'CRITICAL'
    elif score >= 0.4: return 'HIGH'
    elif score >= 0.2: return 'MEDIUM'
    else:              return 'LOW'

train_preds['RISK_TIER'] = train_preds['RISK_SCORE'].apply(assign_tier)
train_preds.to_parquet(OUT_DIR / "train_predictions.parquet", index=False)

test_preds = pd.DataFrame({
    'SK_ID_CURR':  test[ID_COL].values,
    'RISK_HIST':   test_hist_preds,
    'RISK_BEH':    test_beh_preds,
    'RISK_SCORE':  test_ensemble,
})
test_preds['RISK_TIER'] = test_preds['RISK_SCORE'].apply(assign_tier)
test_preds.to_parquet(OUT_DIR / "test_predictions.parquet",  index=False)

# Quick tier summary
print("\n  Risk Tier Distribution (Train OOF):")
print(train_preds['RISK_TIER'].value_counts().to_string())
pct = train_preds[train_preds['TARGET']==1]['RISK_TIER'].value_counts(normalize=True)
print("\n  Defaulters captured in CRITICAL+HIGH tier:",
      round((pct.get('CRITICAL',0)+pct.get('HIGH',0))*100, 1), "%")

print(f"\n✅ Models saved to: {MODEL_DIR}")
print("=" * 60)
print("Model Training COMPLETE ✅")
print("=" * 60)
