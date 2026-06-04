#!/usr/bin/env encoding: utf-8
"""
🏭 Defect Detection Pro — Tata Steel AI Hackathon
Optimized Pipeline to achieve 90+ Score using Calibrated Stacking, 
Advanced Feature Engineering, and Dynamic Threshold Search.
"""

import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

# ─── SMART FILE PATH RESOLVER ────────────────────────────────────────────────
def locate_file(filename):
    """Locates a file in current or subdirectories to prevent FileNotFoundError."""
    for root, dirs, files in os.walk('.'):
        if filename in files:
            return os.path.join(root, filename)
    for root, dirs, files in os.walk('..'):
        if filename in files:
            return os.path.join(root, filename)
    return filename

# ─── ADVANCED FEATURE ENGINEERING ───────────────────────────────────────────
def engineer_industrial_features(df, train_cols=None):
    """Engineers powerful industrial interaction features and row statistics."""
    df_feat = df.copy()
    
    # Identify primary sensor features (X1 to X49)
    feature_cols = [f'X{i}' for i in range(1, 50) if f'X{i}' in df_feat.columns]
    
    # 1. Capture missingness as a structural signal
    missing_cols = ['X8', 'X10', 'X15', 'X16', 'X21', 'X23', 'X24', 'X25', 'X26', 'X27', 'X42', 'X48']
    for col in missing_cols:
        if col in df_feat.columns:
            df_feat[f'{col}_nan'] = df_feat[col].isnull().astype(int)
            
    # 2. Row-wise process variability profiles (extremely strong for rolling defects)
    df_feat['row_mean'] = df_feat[feature_cols].mean(axis=1)
    df_feat['row_std'] = df_feat[feature_cols].std(axis=1)
    df_feat['row_median'] = df_feat[feature_cols].median(axis=1)
    df_feat['row_skew'] = df_feat[feature_cols].skew(axis=1)
    df_feat['row_min'] = df_feat[feature_cols].min(axis=1)
    df_feat['row_max'] = df_feat[feature_cols].max(axis=1)
    df_feat['row_range'] = df_feat['row_max'] - df_feat['row_min']
    
    # 3. High-impact non-linear interactions among highly separable variables
    # (Top positive and negative correlated features from exploratory analysis)
    if 'X13' in df_feat.columns and 'X35' in df_feat.columns:
        df_feat['X13_over_X35'] = df_feat['X13'] / (df_feat['X35'] + 1e-5)
        df_feat['X13_times_X35'] = df_feat['X13'] * df_feat['X35']
        df_feat['X13_minus_X35'] = df_feat['X13'] - df_feat['X35']
        
    if 'X10' in df_feat.columns and 'X13' in df_feat.columns:
        df_feat['X10_times_X13'] = df_feat['X10'] * df_feat['X13']
        
    if 'X36' in df_feat.columns and 'X34' in df_feat.columns:
        df_feat['X36_over_X34'] = df_feat['X36'] / (df_feat['X34'] + 1e-5)
        
    # Standardize column presence for alignment between train and test
    if train_cols is not None:
        # Fill missing columns in test with 0
        for col in train_cols:
            if col not in df_feat.columns:
                df_feat[col] = 0
        df_feat = df_feat[train_cols]
        
    return df_feat

# ─── MAIN PIPELINE RUNNER ───────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("⚡ DEFECT DETECTION PRO: STARTING OPTIMIZED PIPELINE ⚡")
    print("=" * 60)
    
    # Find dataset files
    train_path = locate_file('train.csv')
    test_path = locate_file('test.csv')
    
    print(f"Loading datasets:\n  Train: {train_path}\n  Test : {test_path}")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Extract IDs and Target
    y = train_df['Y'].astype(int).values
    coil_ids_test = test_df['CoilID'].values
    
    X_train_raw = train_df.drop(columns=['CoilID', 'Y'])
    X_test_raw = test_df.drop(columns=['CoilID'])
    
    # Process features
    print("\nEngineering advanced industrial features...")
    X_train_eng = engineer_industrial_features(X_train_raw)
    train_columns = list(X_train_eng.columns)
    X_test_eng = engineer_industrial_features(X_test_raw, train_cols=train_columns)
    
    # Clean and impute missing cells
    print("Imputing and scaling process values...")
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train_eng)
    X_test_imp = imputer.transform(X_test_eng)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)
    
    # Class imbalance weight calculation
    spw = (y == 0).sum() / (y == 1).sum()
    print(f"Dataset imbalance ratio: {spw:.2f}:1")
    
    # ─── MODEL CONSTITUENTS ─────────────────────────────────────────────────
    # Define highly-tuned constituent models covering diverse hypotheses spaces
    base_models = {
        'RF': RandomForestClassifier(n_estimators=400, class_weight={0: 1, 1: int(spw)}, random_state=42, n_jobs=-1),
        'ET': ExtraTreesClassifier(n_estimators=400, class_weight={0: 1, 1: int(spw)}, random_state=42, n_jobs=-1),
        'HGB': HistGradientBoostingClassifier(max_iter=300, class_weight='balanced', learning_rate=0.03, random_state=42),
        'XGB': xgb.XGBClassifier(n_estimators=400, scale_pos_weight=spw * 1.5, learning_rate=0.03, max_depth=5,
                                 subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss'),
        'LGB': lgb.LGBMClassifier(n_estimators=400, scale_pos_weight=spw * 1.5, learning_rate=0.03, max_depth=5,
                                 subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
    }
    
    # ─── OUT-OF-FOLD STRATIFIED TRAINING ────────────────────────────────────
    print("\nTraining and calibrating Stacking Constituents with 5-Fold Stratified CV...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_predictions = {name: np.zeros(len(y)) for name in base_models.keys()}
    test_predictions = {name: np.zeros(len(X_test_scaled)) for name in base_models.keys()}
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, y)):
        print(f"  ➜ Processing Fold {fold + 1}/5...")
        X_tr, y_tr = X_train_scaled[train_idx], y[train_idx]
        X_va, y_va = X_train_scaled[val_idx], y[val_idx]
        
        for name, clf in base_models.items():
            # Use Calibrated Classifier on each fold to produce reliable confidence scores
            calibrated_clf = CalibratedClassifierCV(estimator=clf, method='isotonic', cv='prefit')
            
            # Fit primary model on the training fold
            clf.fit(X_tr, y_tr)
            
            # Fit calibrator on the validation fold (using independent fold evaluation)
            calibrated_clf.fit(X_va, y_va)
            
            # Store OOF calibrated predictions
            oof_predictions[name][val_idx] = calibrated_clf.predict_proba(X_va)[:, 1]
            # Accumulate test set predictions
            test_predictions[name] += calibrated_clf.predict_proba(X_test_scaled)[:, 1] / 5.0
            
    # Print Individual Model AUC Diagnostics
    print("\nIndividual Calibrated Model OOF Performance:")
    for name in base_models.keys():
        auc = roc_auc_score(y, oof_predictions[name])
        print(f"  • {name:5s} Calibrated OOF AUC: {auc:.5f}")
        
    # ─── OPTIMIZED BLENDING SEARCH ──────────────────────────────────────────
    print("\nSearching for optimal blending weights...")
    
    # We find the blending weights that maximize the OOF F1-Score
    def loss_func(weights):
        # Normalize weights to sum to 1
        w = weights / np.sum(weights)
        blended_oof = np.zeros(len(y))
        for idx, name in enumerate(base_models.keys()):
            blended_oof += w[idx] * oof_predictions[name]
            
        # We optimize for overall AUC first to find robust weights
        return -roc_auc_score(y, blended_oof)
    
    init_weights = [1.0 / len(base_models)] * len(base_models)
    bounds = [(0, 1)] * len(base_models)
    res = minimize(loss_func, init_weights, bounds=bounds, method='SLSQP')
    
    opt_weights = res.x / np.sum(res.x)
    print("Optimal Blending Weights:")
    for idx, name in enumerate(base_models.keys()):
        print(f"  - {name:5s} Weight: {opt_weights[idx]:.4f}")
        
    # Generate final blended OOF and Test probabilities
    blended_oof = np.zeros(len(y))
    blended_test = np.zeros(len(X_test_scaled))
    for idx, name in enumerate(base_models.keys()):
        blended_oof += opt_weights[idx] * oof_predictions[name]
        blended_test += opt_weights[idx] * test_predictions[name]
        
    print(f"\n🚀 Optimized Ensemble OOF AUC: {roc_auc_score(y, blended_oof):.5f}")
    
    # ─── DYNAMIC THRESHOLD SEARCH (90+ SCORE EMPOWERMENT) ───────────────────
    print("\nOptimizing prediction threshold for maximum F1 score...")
    best_threshold = 0.5
    best_f1 = 0.0
    best_precision = 0.0
    best_recall = 0.0
    
    # Scan precision-recall spectrum
    threshold_grid = np.linspace(0.01, 0.99, 1000)
    for t in threshold_grid:
        preds = (blended_oof >= t).astype(int)
        f1 = f1_score(y, preds, zero_division=0)
        prec = precision_score(y, preds, zero_division=0)
        rec = recall_score(y, preds, zero_division=0)
        
        # We target maximizing F1-score as a robust driver of precision/recall balance
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            best_precision = prec
            best_recall = rec
            
    print(f"--- Calibration & Threshold Search Results ---")
    print(f"  Selected Optimal Threshold : {best_threshold:.4f}")
    print(f"  Expected OOF F1-Score      : {best_f1 * 100:.2f}%")
    print(f"  Expected OOF Precision     : {best_precision * 100:.2f}%")
    print(f"  Expected OOF Recall        : {best_recall * 100:.2f}%")
    
    # Confusion Matrix Printout
    final_oof_preds = (blended_oof >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, final_oof_preds).ravel()
    print(f"  OOF Confusion Matrix Details:")
    print(f"    - True Negatives  (Normal Coils)      : {tn}")
    print(f"    - False Positives (Normal Flagged)    : {fp}")
    print(f"    - False Negatives (Missed Defects)    : {fn}")
    print(f"    - True Positives  (Detected Defects)  : {tp}")
    
    # ─── GENERATE SUBMISSION ────────────────────────────────────────────────
    test_preds_binary = (blended_test >= best_threshold).astype(int)
    submission_df = pd.DataFrame({
        'CoilID': coil_ids_test,
        'Y': test_preds_binary
    })
    
    # Guardrail checks
    assert len(submission_df) == 339, "Submission must contain exactly 339 rows."
    assert list(submission_df.columns) == ['CoilID', 'Y'], "Submission must contain 'CoilID' and 'Y' columns."
    
    submission_filename = 'expected_submission.csv'
    submission_df.to_csv(submission_filename, index=False)
    
    print("\n" + "=" * 60)
    print("🎉 PIPELINE COMPLETE AND EXPORTED SUCCESSFULLY 🎉")
    print("=" * 60)
    print(f"  Output File         : {submission_filename}")
    print(f"  Total test samples  : {len(submission_df)}")
    print(f"  Flagged defects     : {test_preds_binary.sum()} (~{test_preds_binary.mean() * 100:.2f}% defect rate)")
    print(f"  Normal predictions  : {len(submission_df) - test_preds_binary.sum()}")
    print("=" * 60)

if __name__ == "__main__":
    main()