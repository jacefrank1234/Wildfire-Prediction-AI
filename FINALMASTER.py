import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, roc_curve, auc
)
os.makedirs('final_results', exist_ok=True)


# These line laods and collects the data that we needa use
WD = pd.read_csv('data/Wildfire_Dataset.csv')
output_column = 'Wildfire'
excluded_columns = ['latitude', 'longitude', 'datetime', 'timedate']
weather_columns = [c for c in WD.columns if c not in excluded_columns + [output_column]]
total_blocks = len(WD) // 75
weather_stuff = WD[weather_columns].values.reshape(total_blocks, 75, len(weather_columns))
target_stuff = WD[output_column].values.reshape(total_blocks, 75)

# We found out the dataset actually has the 15 days after the fire as well, which is rlly cool and neat but we only want the days up until the ignition
#When we were testing XGBoost, it had a super hard time collecting patterns when given the features of everyday up to 60, 
#so we basically "summarized" those in to trend data over the period of days for the tree models
w60 = weather_stuff[:, :60, :]

summary_dict = {}
for idx, col in enumerate(weather_columns):
    data = w60[:, :, idx]
    summary_dict[f'{col}_day60'] = data[:, -1]
    summary_dict[f'{col}_last7_mean'] = data[:, -7:].mean(axis=1)
    summary_dict[f'{col}_last7_max'] = data[:, -7:].max(axis=1)
    summary_dict[f'{col}_last30_mean'] = data[:, -30:].mean(axis=1)
    summary_dict[f'{col}_trend'] = data[:, -7:].mean(axis=1) - data[:, :30].mean(axis=1)

if 'pr' in weather_columns:
    pr_idx = weather_columns.index('pr')
    summary_dict['pr_last14_sum'] = w60[:, -14:, pr_idx].sum(axis=1)
    summary_dict['pr_last60_sum'] = w60[:, :, pr_idx].sum(axis=1)

compiled_WD = pd.DataFrame(summary_dict)

date_col = 'timedate' if 'timedate' in WD.columns else 'datetime'
dates_day1 = WD[date_col].astype(str).values.reshape(total_blocks, 75)[:, 0]
parsed_dates = pd.to_datetime(pd.Series(dates_day1), format='%m/%d', errors='coerce')
compiled_WD.insert(0, 'month', parsed_dates.dt.month)
compiled_WD.insert(1, 'day_of_month', parsed_dates.dt.day)

longitude_col = WD['longitude'].values.reshape(total_blocks, 75)[:, 0]
latitude_col = WD['latitude'].values.reshape(total_blocks, 75)[:, 0]
target_col = target_stuff[:, 60]

valid_mask = pd.Series(target_col).astype(str).str.lower().isin(['yes', 'no']).values
X_tab = compiled_WD.loc[valid_mask].reset_index(drop=True)
X_seq = w60[valid_mask]
y_all = LabelEncoder().fit_transform(pd.Series(target_col)[valid_mask].str.lower())
groups = pd.qcut(longitude_col[valid_mask], q=5, labels=False)

neg_pos_ratio = (y_all == 0).sum() / (y_all == 1).sum()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"      Ready: {len(y_all):,} sequences | {X_tab.shape[1]} engineered features | Device: {device}")


# This is a hybrid CNN LSTM which was actually made last despite it shownig up first here, and is what the 
# actual researchers who made this dataset used for their model. Sadly, we were not able to like get it
# As accurate enough as them, but it still holds a high ensemble weight cuz its the only mode with
# access to the WHOLE dataset, and we think its better at capturing trends and period of prolonged heat
# or smth. This is our least accurate model but we still wanted to have some fun and make a deep learning
# one since that was what everybody who hears about ML thinks about and we thought it was cool.
class TunedWildfireCNN_LSTM(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(num_features, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(
            input_size=64, hidden_size=64, num_layers=2,
            batch_first=True, dropout=0.25
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        return self.classifier(lstm_out[:, -1, :]).squeeze(1)

def train_predict_cnn_lstm(X_tr_seq, y_tr, X_te_seq, epochs=10):
    scaler = StandardScaler()
    N_tr, T, F = X_tr_seq.shape
    X_tr_s = scaler.fit_transform(X_tr_seq.reshape(-1, F)).reshape(N_tr, T, F)
    X_te_s = scaler.transform(X_te_seq.reshape(-1, F)).reshape(len(X_te_seq), T, F)

    loader = DataLoader(
        TensorDataset(torch.tensor(X_tr_s, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32)),
        batch_size=256, shuffle=True
    )
    pos_w = torch.tensor([(y_tr == 0).sum() / (y_tr == 1).sum()]).to(device)
    model = TunedWildfireCNN_LSTM(F).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for _ in range(epochs):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X_te_s, dtype=torch.float32).to(device))).cpu().numpy()


# This thing takes in splitters, loops all 5 folds, and blends + tunes thresholds
def run_full_benchmark(split_name, splitter, group_labels=None):
    oof_P = np.zeros((len(y_all), 4))
    last_xgb_model = None

    splits = splitter.split(X_tab, y_all, group_labels) if group_labels is not None else splitter.split(X_tab, y_all)
    for fold, (tr_idx, te_idx) in enumerate(splits):
        print(f"  Fold {fold + 1}/5 -> Training XGBoost, LightGBM, CatBoost, and Tuned CNN-LSTM...")
        X_tr, X_te = X_tab.iloc[tr_idx], X_tab.iloc[te_idx]
        y_tr = y_all[tr_idx]

        #tuned xgboost
        m_xgb = xgb.XGBClassifier(
            n_estimators=500, max_depth=7, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            scale_pos_weight=neg_pos_ratio, random_state=42, n_jobs=-1
        )
        m_xgb.fit(X_tr, y_tr)
        oof_P[te_idx, 0] = m_xgb.predict_proba(X_te)[:, 1]
        last_xgb_model = m_xgb

        #tuned lightgbm
        m_lgb = lgb.LGBMClassifier(
            n_estimators=500, num_leaves=45, max_depth=7, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=neg_pos_ratio,
            random_state=42, n_jobs=-1, verbose=-1
        )
        m_lgb.fit(X_tr, y_tr)
        oof_P[te_idx, 1] = m_lgb.predict_proba(X_te)[:, 1]

        #tuned catboost
        m_cat = CatBoostClassifier(
            iterations=600, depth=7, learning_rate=0.04, l2_leaf_reg=5,
            auto_class_weights='Balanced', random_seed=42, verbose=0
        )
        m_cat.fit(X_tr, y_tr)
        oof_P[te_idx, 2] = m_cat.predict_proba(X_te)[:, 1]

        #tuned cnn lstm
        oof_P[te_idx, 3] = train_predict_cnn_lstm(X_seq[tr_idx], y_tr, X_seq[te_idx], epochs=10)

    
    blended_probs = np.zeros(len(y_all))
    fold_weights = []
    meta_splits = splitter.split(oof_P, y_all, group_labels) if group_labels is not None else splitter.split(oof_P, y_all)

    for tr_idx, te_idx in meta_splits:
        P_tr, y_tr = oof_P[tr_idx], y_all[tr_idx]
        def loss_fn(w):
            w = w / np.sum(w)
            p = np.clip(P_tr @ w, 1e-7, 1 - 1e-7)
            return -np.mean(y_tr * np.log(p) + (1 - y_tr) * np.log(1 - p))
        res = minimize(loss_fn, x0=[0.25, 0.25, 0.25, 0.25], bounds=[(0, 1)] * 4)
        w_opt = res.x / np.sum(res.x)
        fold_weights.append(w_opt)
        blended_probs[te_idx] = oof_P[te_idx] @ w_opt

    avg_weights = np.mean(fold_weights, axis=0)

    
    best_thresh, best_f1 = 0.50, 0.0
    for t in np.linspace(0.40, 0.75, 36):
        f1 = f1_score(y_all, (blended_probs >= t).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    final_preds = (blended_probs >= best_thresh).astype(int)
    return oof_P, blended_probs, final_preds, avg_weights, best_thresh, last_xgb_model


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
P_norm, probs_norm, preds_norm, weights_norm, thresh_norm, xgb_shap_model = run_full_benchmark(
    "Nationwide Stratified 5-Fold", skf
)

gkf = GroupKFold(n_splits=5)
P_geo, probs_geo, preds_geo, weights_geo, thresh_geo, _ = run_full_benchmark(
    "Geographic 5-Fold on Unseen Regions", gkf, group_labels=groups
)

model_names = ["XGBoost (Tuned)", "LightGBM (Tuned)", "CatBoost (Tuned)", "PyTorch CNN-LSTM"]



model_names = ["XGBoost (Tuned)", "LightGBM (Tuned)", "CatBoost (Tuned)", "PyTorch CNN-LSTM"]

for label, P_mat, ens_probs, ens_preds, w_vec, opt_t in [
    ("NATIONWIDE STRATIFIED 5-FOLD", P_norm, probs_norm, preds_norm, weights_norm, thresh_norm),
    ("GEOGRAPHIC UNSEEN-REGION 5-FOLD", P_geo, probs_geo, preds_geo, weights_geo, thresh_geo)
]:
    print(f"\n--- {label} (Optimal Threshold = {opt_t:.2f}) ---")
    for i, name in enumerate(model_names):
        p_i = (P_mat[:, i] >= 0.50).astype(int)
        prec_i, rec_i, _ = precision_recall_curve(y_all, P_mat[:, i])
        pr_auc_i = auc(rec_i, prec_i)
        print(f"  {name:18s} | Acc: {accuracy_score(y_all, p_i):.4f} | Yes F1: {f1_score(y_all, p_i):.4f} | PR-AUC: {pr_auc_i:.4f}")
    ens_prec, ens_rec, _ = precision_recall_curve(y_all, ens_probs)
    ens_pr_auc = auc(ens_rec, ens_prec)
    print(f"  {'OPTIMIZED ENSEMBLE':18s} | Acc: {accuracy_score(y_all, ens_preds):.4f} | Yes F1: {f1_score(y_all, ens_preds):.4f} | PR-AUC: {ens_pr_auc:.4f}")
    print("  Learned Ensemble Weights:", {n: f"{w*100:.1f}%" for n, w in zip(model_names, w_vec)})

#saved predictions
pd.DataFrame({
    'Actual_Wildfire': y_all,
    'Latitude': latitude_col[valid_mask],
    'Longitude': longitude_col[valid_mask],
    'Norm_XGB': P_norm[:, 0], 'Norm_LGBM': P_norm[:, 1],
    'Norm_CatBoost': P_norm[:, 2], 'Norm_CNNLSTM': P_norm[:, 3],
    'Norm_Ensemble_Prob': probs_norm, 'Norm_Ensemble_Pred': preds_norm,
    'Geo_Ensemble_Prob': probs_geo, 'Geo_Ensemble_Pred': preds_geo
}).to_csv('final_results/master_predictions.csv', index=False)

#generated graphs because we wanted to make it beautiful yk

#Precision-Recall & ROC Curves chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
series_dict = {
    "XGBoost": P_norm[:, 0], "LightGBM": P_norm[:, 1],
    "CatBoost": P_norm[:, 2], "PyTorch CNN-LSTM": P_norm[:, 3],
    "Optimized Ensemble": probs_norm
}
for name, p_vec in series_dict.items():
    lw = 2.8 if "Ensemble" in name else 1.4
    ls = '-' if "Ensemble" in name else '--'
    prec, rec, _ = precision_recall_curve(y_all, p_vec)
    ax1.plot(rec, prec, label=f"{name} (PR-AUC={auc(rec, prec):.3f})", lw=lw, ls=ls)
    fpr, tpr, _ = roc_curve(y_all, p_vec)
    ax2.plot(fpr, tpr, label=f"{name} (ROC-AUC={auc(fpr, tpr):.3f})", lw=lw, ls=ls)

ax1.set_title("Precision-Recall Curve (Nationwide Test)", fontweight='bold')
ax1.set_xlabel("Recall (Wildfire Catch Rate)")
ax1.set_ylabel("Precision (Alarm Accuracy)")
ax1.legend(loc="upper right", fontsize=9)
ax1.grid(alpha=0.3)

ax2.axline((0, 0), slope=1, color='black', linestyle=':', alpha=0.5)
ax2.set_title("ROC Curve (Nationwide Test)", fontweight='bold')
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.legend(loc="lower right", fontsize=9)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('final_results/final_chart1_pr_roc_curves.png')
plt.close()

#Cost benefit curve
thresholds = np.linspace(0.15, 0.85, 71)
costs = [
    (((y_all == 1) & ((probs_norm >= t).astype(int) == 0)).sum() * 10.0 +
     ((y_all == 0) & ((probs_norm >= t).astype(int) == 1)).sum() * 1.0) / len(y_all)
    for t in thresholds
]
best_cost_t = thresholds[np.argmin(costs)]

plt.figure(figsize=(8, 5), dpi=300)
plt.plot(thresholds, costs, color='crimson', lw=2.5, label='Operational Cost (10:1 Miss-to-Alarm Penalty)')
plt.axvline(best_cost_t, color='navy', ls='--', label=f'Min-Cost Cutoff = {best_cost_t:.2f}')
plt.axvline(thresh_norm, color='forestgreen', ls='-.', label=f'Max-F1 Cutoff = {thresh_norm:.2f}')
plt.title("Operational Cost-Benefit Threshold Analysis", fontweight='bold')
plt.xlabel("Decision Probability Threshold")
plt.ylabel("Relative Cost per Forecast Event (Lower is Better)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('final_results/final_chart2_cost_benefit_curve.png')
plt.close()

#shap graph
X_sample = X_tab.sample(n=2000, random_state=42)
explainer = shap.TreeExplainer(xgb_shap_model)
shap_vals = explainer.shap_values(X_sample)
plt.figure(figsize=(9, 6), dpi=300)
shap.summary_plot(shap_vals, X_sample, max_display=12, show=False)
plt.title("SHAP Feature Impact on Wildfire Ignition Risk", fontweight='bold')
plt.tight_layout()
plt.savefig('final_results/final_chart3_shap_summary.png')
plt.close()

#confusion matrix ofc and ensemble weights so that accurate models get heigher "voting power"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
cm = confusion_matrix(y_all, preds_norm)
ConfusionMatrixDisplay(cm, display_labels=["No Fire (0)", "Wildfire (1)"]).plot(ax=ax1, cmap='Blues', colorbar=False)
ax1.set_title(f"Optimized Ensemble Confusion Matrix\n(Threshold = {thresh_norm:.2f})", fontweight='bold')

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']
bars = ax2.bar(model_names, weights_norm * 100, color=colors)
ax2.set_title("Learned Meta-Ensemble Weights (Nationwide)", fontweight='bold')
ax2.set_ylabel("Contribution Weight (%)")
ax2.set_ylim(0, max(weights_norm * 100) + 15)
for bar in bars:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, h + 1.5, f"{h:.1f}%", ha='center', fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('final_results/final_chart4_confusion_and_weights.png')
plt.close()
