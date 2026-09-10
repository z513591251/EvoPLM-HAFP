# EvoPLM-HAFP

**A deep multimodal fusion framework for antifungal peptide prioritization and species-specific activity prediction**

EvoPLM-HAFP (Evolutionary-PLM integrated Hierarchical Antifungal Peptide framework) is a hierarchical discovery framework that integrates evolutionary features and pre-trained language models to prioritize antifungal peptide candidates via sequentially stringent multi-layer analytical screening. The framework combines a transformer-based sequence classifier, multimodal feature fusion and species-specific MIC prediction to enable both AFP recognition and potency-aware nomination against *A. fumigatus*, *C. albicans* and *C. neoformans*.

The repository is organized around the framework's layers:

1. **Sequence classification layer** — a transformer-based multimodal feature-fusion AFP classifier built with **autoBioSeqpy 2.0** (bundled in `autoBioSeqpy-2.0/`). `examples/AFP/generateCMD.py` exhaustively evaluates all combinations of seven feature views (CTriad, DC, modlamp, PAAComp, peptide, peptidy, ESM-2) to select the optimal feature combination, which is then validated on blind-screening data (see [Third-party component](#third-party-component-autobioseqpy-20)).
2. **Species-specific MIC regression layer** — `train_optimize_mic.py` / `predict_mic.py`. Each peptide is encoded by combining **dipeptide composition (DC, 400 dimensions)** with **mean-pooled ESM-2 embeddings (1280 dimensions)** — the feature pairing informed by the feature-combination search in layer 1. Nine regression algorithms are benchmarked per species under 5-fold cross-validation, and the globally best-performing algorithm is retrained on the full data for large-scale FASTA screening.

## Features

- Local ESM-2 (`esm2_t33_650M_UR50D`) feature extraction with GPU batch inference
- Feature caching (MD5-keyed) to avoid recomputation across runs
- Benchmark of 9 regressors: RandomForest, XGBoost, LightGBM, SVR, ExtraTrees, GradientBoosting, AdaBoost, KNeighbors, DecisionTree
- Per-species CV scatter plots (PDF) and raw plot data (CSV)
- Automatic selection of the global best algorithm, saved to `best_model_name.txt`
- Chunked prediction on large FASTA files with activity filtering (MIC < 128 / MIC < 32)

## Installation

```bash
git clone https://github.com/z513591251/EvoPLM-HAFP.git
cd EvoPLM-HAFP
pip install -r requirements.txt
```

You also need the ESM-2 weights file **`esm2_t33_650M_UR50D.pt`** (~2.5 GB) in the project root. It is downloaded automatically by the `fair-esm` tooling on first use, or can be fetched manually from the [facebookresearch/esm](https://github.com/facebookresearch/esm) release assets.

## Data format

Training data (`summary.csv`) must contain three columns:

| Species     | Seq                              | MIC |
|-------------|----------------------------------|-----|
| A.fumigatus | YPPKPESPGEDASPEEMNKYLTALRHYI...  | 80  |

- `Species`: fungal species name (used to group models)
- `Seq`: peptide amino acid sequence
- `MIC`: minimum inhibitory concentration (converted internally to log10)

## Third-party component: autoBioSeqpy 2.0

The first stage of the EvoPLM-HAFP pipeline — the sequence-level AFP classifier — was trained with **autoBioSeqpy 2.0**, a deep learning tool for biological sequence classification that automates sequence encoding, model loading, training and evaluation, and provides ready-to-adapt neural network model templates.

The key entry point is **`autoBioSeqpy-2.0/examples/AFP/generateCMD.py`**. It performs the multimodal feature fusion of this framework:

- Enumerates **all non-empty combinations of seven feature views** (CTriad, DC, modlamp, PAAComp, peptide, peptidy, ESM-2), 5 repeats per combination
- For each combination, generates and runs an autoBioSeqpy command that fuses the selected features through per-feature sub-networks into a **transformer classifier** (`examples/AFP/model/transformer.py`)
- The resulting performances are compared to **select the optimal feature combination**, which is then evaluated on the **blind-screening test set** (`pote*`/`nete*` files)
- The feature configuration of the MIC regression layer (DC + ESM-2) was determined from the outcome of this search

The tool is bundled in `autoBioSeqpy-2.0/` with its original license and documentation (`manual.docx`). If you use this component, please cite:

> Jing, R., Li, Y., Xue, L., Liu, F., Li, M., & Luo, J. (2020). autoBioSeqpy: A Deep Learning Tool for the Classification of Biological Sequences. *Journal of Chemical Information and Modeling*, 60(8), 3755–3764. https://doi.org/10.1021/acs.jcim.0c00409

Upstream: https://github.com/jingry/autoBioSeqpy2.0

Note: autoBioSeqpy has its own dependencies (Keras/TensorFlow, scikit-learn, NumPy) that are separate from `requirements.txt`; see `autoBioSeqpy-2.0/README.md` for its installation instructions.

## Usage

The pipeline runs in order: **sequence classification first, MIC prediction second**.

### 1. AFP sequence classification and feature-combination search (autoBioSeqpy 2.0)

Run the multimodal feature-fusion search from the `autoBioSeqpy-2.0` directory
(paths in the script are relative to it):

```bash
cd autoBioSeqpy-2.0
python examples/AFP/generateCMD.py
cd ..
```

This trains the transformer-based fusion classifier for every feature
combination (7 feature views × all combinations × 5 repeats). Compare the
performances, pick the optimal feature combination, and evaluate it on the
blind-screening test data. Sequences recognized as AFPs by the final classifier
proceed to MIC prediction. See `autoBioSeqpy-2.0/README.md` and
`autoBioSeqpy-2.0/manual.docx` for the full parameter reference.

### 2. Train and benchmark MIC regressors

```bash
python train_optimize_mic.py
```

What it does:

1. Loads `summary.csv` and groups records by species
2. Extracts DC + ESM-2 features (cached in `feature_cache/`)
3. Runs 5-fold cross-validation for all 9 algorithms per species
4. Writes metrics (`results/metrics.csv`), scatter plots (`results/Plot_*.pdf`) and CV plot data (`results/CV_PlotData_*.csv`)
5. Selects the algorithm that wins for the most species, saves its name to `best_model_name.txt`
6. Retrains that algorithm on full data per species and saves `models/Model_*.pkl` + `models/Scaler_*.pkl`

### 3. Predict MIC for the classified candidates

Takes the FASTA of candidates that passed step 1, predicts MIC for every
species, and writes:

```bash
python predict_mic.py
```

- `Predicted_All_MIC.csv` — all predictions
- `Predicted_Active_MIC_below_128.csv` — sequences with predicted MIC < 128 for any species
- `Predicted_Active_MIC_below_32.csv` — sequences with predicted MIC < 32 for any species

## Configuration (things you must replace)

Both scripts have a `USER CONFIGURATION` block at the top. Edit before running:

| Setting | File | Description |
|---|---|---|
| `SUMMARY_CSV` | `train_optimize_mic.py` | Path to your training CSV |
| `ESM_MODEL_PATH` | both | Path to the local ESM-2 `.pt` weights |
| `FASTA_FILE_PATH` | `predict_mic.py` | Input FASTA file to predict |
| `SPECIES_LIST` | `predict_mic.py` | Species to predict; must match the `Species` values in your training CSV |
| `ESM_BATCH_SIZE` | `predict_mic.py` | Sequences per GPU batch (lower to 64/128 if OOM) |
| `CHUNK_SIZE` | `predict_mic.py` | Sequences written to disk per chunk |

## Repository structure

```
EvoPLM-HAFP/
├── autoBioSeqpy-2.0/       # step 1: sequence-level AFP classification
│   └── examples/AFP/generateCMD.py  # multimodal feature-fusion & optimal feature-combination search
├── train_optimize_mic.py   # step 2: MIC model training + benchmarking pipeline
├── predict_mic.py          # step 3: MIC prediction pipeline for classified candidates
├── summary.csv             # example training data (Species, Seq, MIC)
└── requirements.txt
```

Generated at runtime (git-ignored): `models/`, `results/`, `feature_cache/`, `extracted_features/`, `best_model_name.txt`, `Predicted_*.csv`.

## Requirements

- Python >= 3.9
- PyTorch (CUDA recommended for ESM-2 inference; CPU works but is slow)
- See `requirements.txt` for the full list

## Citation

If you use EvoPLM-HAFP in your research, please cite:

<!-- TODO: add the formal citation once the paper is published -->
> *A deep multimodal fusion framework for antifungal peptide prioritization and species-specific activity prediction* (manuscript).

The ESM-2 model is described in:

> Lin, Z., et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science*, 379(6637), 1123–1130. https://doi.org/10.1126/science.ade2574

## License

<!-- TODO: choose a license before publishing -->
The code in this repository is released under the license specified in `LICENSE` (to be added).
Note that `autoBioSeqpy-2.0/` is a third-party component distributed under its
own license (see `autoBioSeqpy-2.0/LICENSE`), which takes precedence for that
directory.
