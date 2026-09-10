import os
import math
import argparse
import warnings
from itertools import product

import joblib
import numpy as np
import pandas as pd

import torch
import torch.serialization
from esm.pretrained import load_model_and_alphabet_core

# Imported so that joblib can unpickle any of the candidate models
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
# USER CONFIGURATION - edit the items below before running
# ============================================================
FASTA_FILE_PATH = " "               # input FASTA file to predict
ESM_MODEL_PATH = "esm2_t33_650M_UR50D.pt"     # local ESM-2 weights (see README)
BEST_MODEL_TXT = "best_model_name.txt"        # written by train_optimize_mic.py

# Species to predict. Must match the species names used during training
# (i.e. the values in the Species column of summary.csv).
SPECIES_LIST = ["A.fumigatus", "C. albicans", "C.neoform"]

ESM_BATCH_SIZE = 256   # sequences per GPU batch; lower to 64/128 if OOM
CHUNK_SIZE = 10000     # sequences processed and written to disk per chunk
# ============================================================

ESM_LAYER = 33
ESM_HIDDEN_SIZE = 1280


def safe_filename(text):
    return str(text).replace('.', '').replace(' ', '_').replace('/', '_').replace('\\', '_')


def read_fasta(file_path):
    ids, seqs = [], []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                ids.append(line[1:])
            elif line:
                seqs.append(line.upper())
    return ids, seqs


# Load local ESM-2 model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Loading ESM-2 model from {ESM_MODEL_PATH} on {device}...")

torch.serialization.add_safe_globals([argparse.Namespace])
model_data = torch.load(ESM_MODEL_PATH, map_location="cpu")
esm_model, alphabet = load_model_and_alphabet_core("esm2_t33_650M_UR50D", model_data)
esm_model = esm_model.to(device).eval()
batch_converter = alphabet.get_batch_converter()


def extract_features_batch(sequences):
    """Dipeptide composition (400 dims) + mean-pooled ESM-2 embedding (1280 dims)."""
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    dipeptides = [''.join(p) for p in product(amino_acids, repeat=2)]
    dpc_list, esm_list = [], []

    for seq in sequences:
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

    for i in range(0, len(sequences), ESM_BATCH_SIZE):
        batch_seqs = sequences[i:i + ESM_BATCH_SIZE]
        data = [(f"seq_{j}", seq) for j, seq in enumerate(batch_seqs)]

        _, _, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = esm_model(batch_tokens, repr_layers=[ESM_LAYER])
        token_reps = results["representations"][ESM_LAYER]

        # strip padding and CLS/EOS tokens, then mean-pool over residues
        for j, seq in enumerate(batch_seqs):
            L = len(seq)
            esm_list.append(token_reps[j, 1:L + 1].mean(dim=0).cpu().numpy())

    df_dpc = pd.DataFrame(dpc_list, columns=[f'DC_{dp}' for dp in dipeptides])
    df_esm = pd.DataFrame(esm_list, columns=[f'ESM2_{i}' for i in range(ESM_HIDDEN_SIZE)])
    return pd.concat([df_dpc, df_esm], axis=1)


def main():
    if not os.path.exists(FASTA_FILE_PATH):
        raise FileNotFoundError(f"Cannot find FASTA file: {FASTA_FILE_PATH}")

    if not os.path.exists(BEST_MODEL_TXT):
        raise FileNotFoundError(
            f"Cannot find {BEST_MODEL_TXT}. Please run train_optimize_mic.py first.")

    with open(BEST_MODEL_TXT, 'r') as f:
        global_best_algo = f.read().strip()
    print(f"Loaded best algorithm configuration: [{global_best_algo}]")

    # Pre-load per-species models and scalers
    models_dict = {}
    for sp in SPECIES_LIST:
        safe_sp = safe_filename(sp)
        safe_algo = safe_filename(global_best_algo)
        model_path = f"models/Model_{safe_sp}_{safe_algo}.pkl"
        scaler_path = f"models/Scaler_{safe_sp}_{safe_algo}.pkl"

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            print(f"Warning: model/scaler for {sp} not found. Skipping this species.")
            continue

        models_dict[sp] = {
            'scaler': joblib.load(scaler_path),
            'model': joblib.load(model_path),
        }

    if not models_dict:
        raise RuntimeError("No valid models found to run predictions.")

    ids, seqs = read_fasta(FASTA_FILE_PATH)
    total_samples = len(seqs)
    print(f"Successfully loaded {total_samples} sequences from FASTA.")

    output_all = "Predicted_All_MIC.csv"
    output_below_128 = "Predicted_Active_MIC_below_128.csv"
    output_below_32 = "Predicted_Active_MIC_below_32.csv"

    valid_species = list(models_dict.keys())
    header = ["ID", "Seq"] + valid_species

    for file in [output_all, output_below_128, output_below_32]:
        pd.DataFrame(columns=header).to_csv(file, index=False)

    # Process in chunks to bound memory usage on large FASTA files
    num_chunks = math.ceil(total_samples / CHUNK_SIZE)
    print(f"\nStarting prediction in {num_chunks} chunk(s) of {CHUNK_SIZE} sequences...\n")

    for c in range(num_chunks):
        start_idx = c * CHUNK_SIZE
        end_idx = min((c + 1) * CHUNK_SIZE, total_samples)

        chunk_ids = ids[start_idx:end_idx]
        chunk_seqs = seqs[start_idx:end_idx]

        print(f"Processing chunk {c + 1}/{num_chunks} "
              f"(sequences {start_idx + 1} to {end_idx})...")

        X_features = extract_features_batch(chunk_seqs)
        chunk_df = pd.DataFrame({"ID": chunk_ids, "Seq": chunk_seqs})

        for sp in valid_species:
            scaler = models_dict[sp]['scaler']
            model = models_dict[sp]['model']

            X_scaled = scaler.transform(X_features)
            pred_log10_mic = model.predict(X_scaled)
            chunk_df[sp] = np.round(np.power(10, pred_log10_mic), 3)  # back to MIC scale

        chunk_df = chunk_df[header]

        # Keep sequences predicted active (MIC < 128 or < 32) for any species
        df_128 = chunk_df[(chunk_df[valid_species] < 128).any(axis=1)]
        df_32 = chunk_df[(chunk_df[valid_species] < 32).any(axis=1)]

        chunk_df.to_csv(output_all, mode='a', header=False, index=False)
        if not df_128.empty:
            df_128.to_csv(output_below_128, mode='a', header=False, index=False)
        if not df_32.empty:
            df_32.to_csv(output_below_32, mode='a', header=False, index=False)

    print(f"\n[Success] Prediction finished for {total_samples} sequences.")
    print(f" - All results: {output_all}")
    print(f" - MIC < 128:   {output_below_128}")
    print(f" - MIC < 32:    {output_below_32}")


if __name__ == "__main__":
    main()
