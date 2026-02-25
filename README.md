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
│   └── train.py            # Train GNN with early stopping
│
├── evaluation/             # Model evaluation
│   ├── evaluate_model.py   # Evaluate on test cell lines
│   ├── evaluate_with_explanations.py  # Detailed rankings + explainability
│   └── test_random_cell.py # Quick test with random cell line
│
├── inference/              # Drug ranking for new cell lines
│   ├── predict_drugs.py    # Main prediction script (multiple input formats)
│   ├── rank_drugs.py       # Alternative ranking script
│   └── generate_sample_input.py  # Generate sample input files
│
├── api/                    # REST API deployment
│   ├── api_server.py       # Flask API server
│   ├── example_client.py   # API client example
│   ├── Dockerfile          # Docker container
│   └── DEPLOYMENT_README.md
│
├── data/                   # Data directory
│   ├── raw/                # Raw GDSC files (download separately)
│   └── processed/          # Processed PyTorch data (generated)
│
└── models/                 # Trained model weights (generated)
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

```bash
cd training
python train.py --epochs 100 --patience 15
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

## API Usage

```bash
cd api
python api_server.py
```

Then send POST requests:
```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"gene_expression": [0.1, 0.2, ...]}'
```

## License

MIT License

## Citation

Data from: Genomics of Drug Sensitivity in Cancer (GDSC)
https://www.cancerrxgene.org/
