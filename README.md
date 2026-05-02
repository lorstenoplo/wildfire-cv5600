# WildFire-CV5600

Wildfire ignition probability prediction for Northern California (2012–2023) using spatio-temporal weather, terrain, vegetation, and fire-weather index features.

The pipeline has three stages:

1. **Data collection & preprocessing** — download raw sources, transform into analysis-ready parquet files
2. **Current-day prediction** — PatchTST + DLA model that estimates ignition probability for the current day
3. **Future forecasting** — Monte Carlo simulation that propagates the current-day probability forward over a 7-day horizon

---

## Dataset

The preprocessed datasets are publicly available on Hugging Face — **you do not need to run data collection** unless you want to regenerate from scratch.

| Split | Link |
|-------|------|
| Features (X) | [NagrajMG/WildFire-X](https://huggingface.co/datasets/NagrajMG/WildFire-X) |
| Labels (Y) | [NagrajMG/WildFire-Y](https://huggingface.co/datasets/NagrajMG/WildFire-Y) |

Download them and place the files under `data_processed/`.

---

## Project Structure

```
wildfire_data/
├── data_collection/            # Download & transform scripts
│   ├── run_all.sh              # Entry point — runs all steps
│   ├── config.py               # AOI, date range, paths
│   ├── gee_auth.py             # One-time Google Earth Engine auth
│   ├── gdrive_download_and_merge.py
│   ├── download/               # Per-source download scripts (GridMET, SRTM, VIIRS, GFS, ignition)
│   └── transform/              # Per-source transform scripts (t01–t08)
├── build_features/             # Feature engineering notebooks (imputation, window features)
├── build_labels/               # Label construction
│   ├── build_label_splits_only.ipynb
│   └── pattern_matches/        # Matched fire patterns & combined label CSV/JSON
├── utils/                      # Shared utilities across the pipeline
│   ├── cleaning_missing.py     # Missing value handling
│   ├── feature_grids.py        # Spatial grid utilities
│   ├── fire_patterns.py        # Fire pattern extraction
│   ├── nofire_sampling.py      # Negative sample generation
│   ├── nofire_splitter.py      # Train/val/test splitting for no-fire samples
│   ├── pattern_splits.py       # Pattern-based data splitting
│   ├── replenish_utils.py      # Sample replenishment helpers
│   ├── window_feature_utils.py # Sliding window feature construction
│   └── window_pipeline_io.py   # I/O for window feature pipeline
├── current_day_predictions/    # Stage 2 — current-day model (run on Kaggle)
│   ├── current-probability-pipeline-2.ipynb
│   ├── model_patchtst_dla.py   # PatchTST + DLA architecture
│   ├── trainer.py
│   ├── configs.py
│   ├── data_io.py
│   ├── losses.py
│   ├── metrics.py
│   └── utils.py
├── monte_carlo/                # Stage 3 — future forecasting (run on Kaggle)
│   ├── future-forecasting-monte-carlo.ipynb
│   ├── functions.py
│   ├── main_kaggle.py
│   └── future_mc_forecast/     # MC forecast subpackage
│       ├── forecast_runner.py  # Recursive forecast orchestration
│       ├── monte_carlo_projector.py
│       ├── mlp_model.py        # Future probability MLP
│       ├── training_pipeline.py
│       ├── training_data.py
│       ├── feature_stats.py
│       ├── io_utils.py
│       └── config.py
├── cffdrs/                     # Fire-weather index library (third-party, see credits)
├── metadata/                   # Area of interest geometry
├── research_paper/             # Reference papers
└── notebooks/                  # Exploratory notebooks
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
# .venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

> **Note:** The model training notebooks (`current_day_predictions/`, `monte_carlo/`) are designed to run on Kaggle with GPU. The `requirements.txt` lists those packages for reference, but you only need the data collection packages if running locally.

---

## Running the Pipeline

### Option A — Use the pre-built dataset (recommended)

Skip to [Stage 2](#stage-2--current-day-prediction).

### Option B — Regenerate from scratch

#### Stage 1 — Data Collection

**One-time GEE authentication** (opens browser):

```bash
python data_collection/gee_auth.py
```

**Run all downloads and transforms:**

```bash
bash data_collection/run_all.sh
```

This runs downloads in parallel (GridMET, SRTM, VIIRS spectral, historical ignition) then runs transforms sequentially. Processed files land in `data_processed/`.

Data sources:

| Source | Description |
|--------|-------------|
| GridMET | Daily gridded weather (17 variables) |
| SRTM | Terrain elevation |
| VIIRS VNP09GA | Surface spectral reflectance via GEE |
| MTBS / NIFC | Historical fire ignition points |

Area of interest: Northern California (`[-124.135, 36.993, -118.963, 42.01]`), 2012–2023.

---

### Stage 2 — Current-Day Prediction

Open and run:

```
current_day_predictions/current-probability-pipeline-2.ipynb
```

The model is a **PatchTST + DLA (Deep Layer Aggregation)** hybrid:

- Tokenizes weather/FWI time-series into temporal patches per feature channel
- Stacked transformer encoder with multi-level DLA aggregation
- Auxiliary tabular shortcut branch + gated fusion head
- Outputs per-pixel ignition probability for the current day

Key config is in `current_day_predictions/configs.py` (`ModelConfig`, `TrainConfig`, `DataConfig`).

---

### Stage 3 — Future Forecasting (Monte Carlo)

Open and run after Stage 2:

```
monte_carlo/future-forecasting-monte-carlo.ipynb
```

Takes the current-day probability output from Stage 2 and runs recursive Monte Carlo simulation over a 7-day horizon (`horizon_days=7`, `num_mc_samples=100000`). Outputs day-by-day probability distributions.

---

## Fire-Weather Indices (cffdrs)

The `cffdrs/` directory contains the **Canadian Forest Fire Danger Rating System** implementation used to compute FWI indices (FFMC, DMC, DC, ISI, BUI, FWI).

This code is **not** part of this project. Full credit goes to the original author:

> Greg A. Greene — [github.com/gagreene/cffdrs](https://github.com/gagreene/cffdrs)

---

## Reference Papers

All papers are in `research_paper/`:

- `spatio-temp.pdf` — Spatio-temporal wildfire modeling
- `remotesensing-14-03496-v3.pdf` — Remote sensing for fire risk
- `ts-sat-fire.pdf` — Time-series satellite fire analysis
- `wild-fire-risk-power.pdf` — Wildfire risk and power infrastructure
- `Spatio-Temporal_Agnostic_Deep_Learning_Modeling_of_Forest_Fire_Prediction_Using_Weather_Data.pdf` — Spatio-temporal agnostic deep learning for fire prediction

---

## Area of Interest

The study region geometry is in `metadata/aoi.geojson` and `metadata/aoi_shapefile.zip`.
