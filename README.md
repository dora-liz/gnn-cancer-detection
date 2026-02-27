# GNN Cancer Drug Ranking

A Graph Neural Network (GNN) model for predicting drug effectiveness on cancer cell lines based on gene expression data.

## Overview

This project uses a heterogeneous Graph Neural Network to predict IC50 (drug effectiveness) values for cancer cell lines based on their gene expression profiles. The model can rank drugs for new cell lines to identify potentially effective treatments.

**Architecture:**
- **FeatureProjector**: Projects raw gene expression (17,737 features) to 256-dimensional embeddings
- **GraphSAGE GNN**: 3-layer message-passing network on cell line-drug interaction graph
- **LinkPredictor**: 4-layer MLP decoder predicting IC50 from cell-drug embeddings

**Performance:**
- Spearman correlation: ~0.35
- Top-10 Precision: ~39%
- Trained on GDSC1/GDSC2 data (912 cell lines, 542 drugs, 434K interactions)

## Project Structure

```
├── gnn_model.py            # Model architecture (shared across all scripts)
├── requirements.txt        # Python dependencies
│
├── preprocessing/          # Data preprocessing
│   └── preprocess_gdsc.py  # Convert GDSC Excel → PyTorch format
│
├── training/               # Model training
│   ├── train.py            # Standard GNN training with early stopping
│   └── train_active_learning.py  # Active learning training
│
├── evaluation/             # Model evaluation
│   ├── evaluate_model.py   # Evaluate on test cell lines
│   ├── evaluate_with_explanations.py  # Detailed rankings + explainability
│   └── test_random_cell.py # Quick test with random cell line
│
├── inference/              # Drug ranking for new cell lines
│   ├── predict_drugs.py    # Main prediction script (multiple input formats)
│   ├── rank_drugs.py       # Alternative ranking script with explainability
│   ├── explain_gnn_prediction.py  # Feature importance analysis
│   └── generate_sample_input.py  # Generate sample input files
│
├── data/                   # Data directory (not in repo, generate locally)
│   ├── raw/                # Raw GDSC files (download separately)
│   └── processed/          # Processed PyTorch data (generated)
│
└── models/                 # Trained model weights (generated locally)
```

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install PyTorch (CPU)
pip install torch==2.4.0+cpu torchvision==0.19.0+cpu torchaudio==2.4.0+cpu --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch Geometric
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
pip install torch-geometric

# Install other dependencies
pip install pandas scipy flask openpyxl
```

### 2. Download Data

Download raw data from [GDSC](https://www.cancerrxgene.org/downloads/bulk_download):
- `Cell_line_RMA_proc_basalExp.txt` (gene expression)
- `Cell_Lines_Details.xlsx` (cell line metadata)
- `screened_compounds_rel_8.5.csv` (drug annotations)
- `GDSC1_fitted_dose_response_27Oct23.xlsx` (IC50 data)
- `GDSC2_fitted_dose_response_27Oct23.xlsx` (IC50 data)

Place in `data/raw/` directory.

### 3. Preprocess Data

```bash
cd preprocessing
python preprocess_gdsc.py
```

### 4. Train Model

**Standard Training:**
```bash
cd training
python train.py --epochs 100 --patience 15
```

**Active Learning Training (recommended for limited labeling budget):**
```bash
python train_active_learning.py --strategy hybrid --budget 5000 --batch-size 100
```

### 5. Predict Drugs for New Cell Line

```bash
cd inference
python predict_drugs.py --input cell_features.pt --top-k 20
```

## Input Format

The model expects gene expression values for 17,737 genes. Supported formats:
- `.pt` - PyTorch tensor
- `.json` - JSON array of floats
- `.csv` - Comma-separated values

## Active Learning

Active learning reduces the number of expensive drug-cell experiments needed by intelligently selecting which pairs to test next.

### Why Active Learning?

- Drug screening is expensive (~$100-1000 per experiment)
- Active learning can achieve same accuracy with **20-40% fewer experiments**
- Prioritizes informative samples over random selection

### Query Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `random` | Baseline random sampling | Comparison |
| `uncertainty` | MC Dropout variance | General use |
| `diversity` | Cluster-based coverage | Avoiding redundancy |
| `hybrid` | Uncertainty + diversity | **Recommended** |
| `qbc` | Committee disagreement | Robust uncertainty |

### Usage

```bash
cd training

# Hybrid strategy (recommended)
python train_active_learning.py --strategy hybrid --budget 5000 --batch-size 100

# Uncertainty sampling with more MC samples
python train_active_learning.py --strategy uncertainty --mc-samples 30 --budget 10000

# Compare strategies
python train_active_learning.py --strategy random --budget 5000 --output-prefix al_random
python train_active_learning.py --strategy hybrid --budget 5000 --output-prefix al_hybrid
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--strategy` | hybrid | Query strategy |
| `--budget` | 5000 | Total experiments allowed |
| `--batch-size` | 100 | Samples per AL round |
| `--initial-size` | 1000 | Initial labeled pool |
| `--mc-samples` | 20 | MC Dropout samples for uncertainty |

### Output

The script saves:
- `models/active_learning_*.pt` - Trained model weights
- `models/active_learning_history.json` - Learning curve data

## Demonstration Guide

### Prerequisites After Cloning

After cloning the repository, you need to set up the data and models:

```bash
# 1. Clone the repo
git clone <repository-url>
cd finalproject

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 3. Install dependencies (see Quick Start section above)

# 4. Download GDSC data files to data/raw/ (see "Download Data" section)

# 5. Preprocess data
cd preprocessing
python preprocess_gdsc.py
cd ..

# 6. Train the model (takes ~10-30 minutes depending on hardware)
cd training
python train.py --epochs 100 --patience 15
cd ..
```

### Quick Demo (5 minutes)

**1. Test on a random cell line:**
```bash
cd evaluation
python test_random_cell.py
```

This will:
- Pick a random cell line from test data
- Remove its known drug responses (simulating a "new" cell line)
- Predict drug rankings
- Compare with actual rankings (Spearman correlation)

**2. Evaluate on multiple cell lines:**
```bash
python evaluate_model.py --num-cells-to-test 5
```

### Full Demo Workflow

**Step 1: Show Dataset Statistics**
```bash
python -c "import torch; d=torch.load('data/processed/gdsc_processed_train.pt', weights_only=False); print(f'Cell lines: {d[\"cell_line\"].x.shape[0]}'); print(f'Drugs: {d[\"drug\"].x.shape[0]}'); print(f'Gene features: {d[\"cell_line\"].x.shape[1]}')"
```

**Step 2: Generate Sample Input & Get Drug Rankings**
```bash
cd inference

# Extract a cell line from test data
python generate_sample_input.py --from-existing ../data/processed/gdsc_processed_test.pt --output demo_cell.pt

# Get drug rankings
python predict_drugs.py --input demo_cell.pt --top-k 20 --output demo_results.json
```

**Step 3: Show Explainability (which genes influenced the prediction)**
```bash
python rank_drugs.py --cell-features demo_cell.pt --top-k 10 --explain-top-features 10
```

### Key Metrics to Present

| Metric | Value | Meaning |
|--------|-------|---------|
| **Top-10 Precision** | ~39% | 4 of top 10 predicted drugs are actually in top 10 |
| **Spearman Correlation** | ~0.35 | Moderate ranking agreement with ground truth |
| **Cell Lines** | 912 | Cancer cell lines in training data |
| **Drugs** | 542 | Compounds that can be ranked |
| **Interactions** | 434K | Drug-cell line IC50 measurements |

### What Each Component Does

| Component | Purpose |
|-----------|---------|
| **Gene Expression Input** | 17,737 features from RNA-seq representing cell state |
| **FeatureProjector** | Compresses genes to 256-dim learned representation |
| **GraphSAGE GNN** | Learns patterns from cell-drug interaction graph |
| **LinkPredictor** | Predicts IC50 (lower = more effective drug) |

### Real-World Use Case

> "Given a patient's tumor gene expression profile, the model ranks 542 cancer drugs by predicted effectiveness, helping oncologists prioritize which drugs to test first."

## License

MIT License

## Citation

Data from: Genomics of Drug Sensitivity in Cancer (GDSC)
https://www.cancerrxgene.org/
