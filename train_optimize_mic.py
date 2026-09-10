import os
import hashlib
import argparse
import warnings
from itertools import product
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

import torch
import torch.serialization
from esm.pretrained import load_model_and_alphabet_core

from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    ExtraTreesRegressor, AdaBoostRegressor,
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')

# ============================================================
# USER CONFIGURATION - edit the paths below before running
# ============================================================
SUMMARY_CSV = 'summary.csv'                    # training data: Species, Seq, MIC
ESM_MODEL_PATH = 'esm2_t33_650M_UR50D.pt'      # local ESM-2 weights (see README)
BEST_MODEL_TXT = 'best_model_name.txt'         # output: name of the global best algorithm

ESM_LAYER = 33
ESM_HIDDEN_SIZE = 1280
N_SPLITS = 5
RANDOM_STATE = 42
# ============================================================

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

for folder in ['models', 'results', 'feature_cache', 'extracted_features']:
    os.makedirs(folder, exist_ok=True)


def safe_filename(text):
    return str(text).replace('.', '').replace(' ', '_').replace('/', '_').replace('\\', '_')


def build_sequence_hash(sequences):
    seq_list = [str(seq).strip().upper() for seq in sequences]
    return hashlib.md5("||".join(seq_list).encode('utf-8')).hexdigest()[:16]


def build_cv_pipeline(base_model):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', clone(base_model)),
    ])


# Load local ESM-2 model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Loading ESM-2 model from {ESM_MODEL_PATH} on {device}...")
torch.serialization.add_safe_globals([argparse.Namespace])
model_data = torch.load(ESM_MODEL_PATH, map_location="cpu")
esm_model, alphabet = load_model_and_alphabet_core("esm2_t33_650M_UR50D", model_data)
esm_model = esm_model.to(device).eval()
batch_converter = alphabet.get_batch_converter()


def get_features(sequences):
    """Dipeptide composition (400 dims) + mean-pooled ESM-2 embedding (1280 dims)."""
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    dipeptides = [''.join(p) for p in product(amino_acids, repeat=2)]
    dpc_list, esm_list = [], []

    for idx, seq in enumerate(sequences):
        seq = str(seq).strip().upper()
        L = len(seq)

        counts_dpc = {dp: 0 for dp in dipeptides}
        if L >= 2:
            for i in range(L - 1):
                dp = seq[i:i + 2]
                if dp in counts_dpc:
                    counts_dpc[dp] += 1
            dpc_list.append([counts_dpc[dp] / (L - 1) for dp in dipeptides])
        else:
            dpc_list.append([0] * 400)

        _, _, batch_tokens = batch_converter([(f"seq_{idx}", seq)])
        batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            results = esm_model(batch_tokens, repr_layers=[ESM_LAYER])
        token_reps = results["representations"][ESM_LAYER]
        esm_list.append(token_reps[0, 1:L + 1].mean(dim=0).cpu().numpy())

    df_dpc = pd.DataFrame(dpc_list, columns=[f'DC_{dp}' for dp in dipeptides])
    df_esm = pd.DataFrame(esm_list, columns=[f'ESM2_{i}' for i in range(ESM_HIDDEN_SIZE)])
    return pd.concat([df_dpc, df_esm], axis=1)


def get_features_cached(species, sequences):
    seq_series = pd.Series(sequences).astype(str).str.strip().str.upper()
    seq_hash = build_sequence_hash(seq_series.tolist())
    cache_path = os.path.join(
        'feature_cache', f'{safe_filename(species)}_{len(seq_series)}_{seq_hash}.pkl')

    if os.path.exists(cache_path):
        X = joblib.load(cache_path)
    else:
        X = get_features(seq_series.tolist())
        joblib.dump(X, cache_path, compress=3)
    X.index = seq_series.index
    return X


def save_scatter_plot(y_true, y_pred, p_coeff, out_path):
    plt.figure(figsize=(7, 7))
    plt.scatter(y_pred, y_true, alpha=0.4, color='red', s=40)

    min_val = min(np.min(y_true), np.min(y_pred)) - 0.5
    max_val = max(np.max(y_true), np.max(y_pred)) + 0.5

    plt.plot([min_val, max_val], [min_val, max_val], color='black', linewidth=1.5)
    plt.plot([min_val, max_val], [min_val + 0.3, max_val + 0.3],
             color='black', linestyle='--', linewidth=1)
    plt.plot([min_val, max_val], [min_val - 0.3, max_val - 0.3],
             color='black', linestyle='--', linewidth=1)

    plt.title(f"R = {p_coeff:.4f}", fontsize=16)
    plt.xlabel("Predicted Log(MIC)", fontsize=14)
    plt.ylabel("True Log(MIC)", fontsize=14)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(out_path, format='pdf')
    plt.close()


models = {
    "RandomForest": RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE),
    "XGBoost": xgb.XGBRegressor(n_jobs=-1, random_state=RANDOM_STATE, verbosity=0),
    "LightGBM": lgb.LGBMRegressor(n_jobs=-1, random_state=RANDOM_STATE, verbose=-1),
    "SVR": SVR(kernel='rbf', C=10),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE),
    "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    "AdaBoost": AdaBoostRegressor(random_state=RANDOM_STATE),
    "KNeighbors": KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
    "DecisionTree": DecisionTreeRegressor(random_state=RANDOM_STATE),
}


def main():
    df = pd.read_csv(SUMMARY_CSV)
    df.columns = ['Species', 'Sequence', 'MIC']
    df = df.dropna()

    all_metrics = []
    all_species_data = {}   # species -> (X, y), reused for final training
    species_best_names = []  # best algorithm per species, used to pick the global best

    for sp in df['Species'].unique():
        df_sp = df[df['Species'] == sp].copy()
        if len(df_sp) < 5:
            continue

        df_sp['logMIC'] = np.log10(df_sp['MIC'])
        y = df_sp['logMIC'].values
        X = get_features_cached(sp, df_sp['Sequence'])
        all_species_data[sp] = (X, y)

        # Save the full feature table for this species
        combined_sp = pd.concat(
            [df_sp.reset_index(drop=True), X.reset_index(drop=True)], axis=1)
        combined_sp.to_csv(
            f'extracted_features/Features_{safe_filename(sp)}.csv', index=False)

        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        best_r, best_algo = -1.0, ""

        for name, model in models.items():
            try:
                y_pred = cross_val_predict(
                    build_cv_pipeline(model), X, y, cv=kf, n_jobs=-1)

                p_coeff, _ = pearsonr(y, y_pred)
                r2 = r2_score(y, y_pred)
                mae = mean_absolute_error(y, y_pred)
                mse = mean_squared_error(y, y_pred)

                all_metrics.append({
                    'Species': sp, 'Model': name, 'Pearson': p_coeff,
                    'COD': r2, 'MAE': mae, 'MSE': mse, 'RMSE': np.sqrt(mse),
                })

                safe_sp = safe_filename(sp)
                safe_name = safe_filename(name)
                save_scatter_plot(
                    y, y_pred, p_coeff,
                    f'results/Plot_{safe_sp}_{safe_name}.pdf')
                pd.DataFrame({
                    'Species': sp, 'Model': name,
                    'Sequence': df_sp['Sequence'].values,
                    'True_MIC': df_sp['MIC'].values,
                    'True_Log10MIC': y,
                    'Predicted_Log10MIC': y_pred,
                }).to_csv(f'results/CV_PlotData_{safe_sp}_{safe_name}.csv', index=False)

                if p_coeff > best_r:
                    best_r, best_algo = p_coeff, name
            except Exception as e:
                print(f"Error processing {sp} with {name}: {e}")

        if best_algo:
            species_best_names.append(best_algo)
            print(f"{sp}: best algorithm = {best_algo} (R = {best_r:.4f})")

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv('results/metrics.csv', index=False)

    # Pick the algorithm that wins for the most species
    most_common = Counter(species_best_names).most_common(1)
    if not most_common:
        print("No models were trained successfully.")
        return

    global_best_algo = most_common[0][0]
    print(f"\n{'=' * 30}")
    print(f"Global Best Algorithm: {global_best_algo}")
    print(f"{'=' * 30}")

    with open(BEST_MODEL_TXT, 'w') as f:
        f.write(global_best_algo)

    # Retrain the global best algorithm on full data per species and save
    for sp, (X, y) in all_species_data.items():
        print(f"Final training for {sp} using {global_best_algo}...")

        final_scaler = StandardScaler()
        X_scaled = final_scaler.fit_transform(X)

        final_model = clone(models[global_best_algo])
        final_model.fit(X_scaled, y)

        safe_sp = safe_filename(sp)
        safe_algo = safe_filename(global_best_algo)
        joblib.dump(final_model, f'models/Model_{safe_sp}_{safe_algo}.pkl')
        joblib.dump(final_scaler, f'models/Scaler_{safe_sp}_{safe_algo}.pkl')

    print("All species models saved using the global best algorithm.")


if __name__ == "__main__":
    main()
