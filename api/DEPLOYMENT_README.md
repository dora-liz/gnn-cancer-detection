# Drug Ranking Prediction - Portable Deployment Guide

This guide explains how to deploy and use the drug ranking model on external systems.

## Required Files for Deployment

Copy these files to your target system:

```
predict_drugs.py                     # Main prediction script
gnn_model.py                         # Model architecture definitions
train_data.pt                        # Base graph structure
training_artifacts.pt                # Model hyperparameters
feature_projector_trained_cpu.pt     # Trained feature projector weights
gnn_model_trained_cpu.pt             # Trained GNN encoder weights
link_predictor_trained_cpu.pt        # Trained link predictor weights
generate_sample_input.py             # (Optional) Helper to create sample inputs
```

## Installation

Install required Python packages:

```bash
# For CPU-only (recommended for deployment):
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric

# Or with GPU support:
pip install torch
pip install torch-geometric
```

## Usage

### Basic Usage

```bash
# Rank drugs for a cell line from a .pt file
python predict_drugs.py --input cell_features.pt

# From JSON file
python predict_drugs.py --input gene_expression.json

# From CSV file
python predict_drugs.py --input expression_data.csv
```

### Options

```
--input, -i       : Path to cell line features file (REQUIRED)
--output, -o      : Output file path (default: drug_ranking_results.json)
--top-k, -k       : Number of top drugs to return (default: 20)
--explain         : Include feature importance explanations
--format          : Output format: json, csv, or text (default: json)
--drug-names      : Optional JSON file with drug name mappings
--quiet, -q       : Suppress console output
```

### Examples

```bash
# Get top 50 drugs with explanations
python predict_drugs.py --input mydata.pt --top-k 50 --explain

# Output as CSV
python predict_drugs.py --input mydata.json --format csv --output results.csv

# Quiet mode (only write to file)
python predict_drugs.py --input mydata.pt --quiet
```

## Input File Formats

### PyTorch Tensor (.pt)
A tensor of shape `[1, 19215]` containing gene expression values.

### JSON (.json)
```json
[0.123, -0.456, 0.789, ...]
```
Or with a wrapper:
```json
{"gene_expression": [0.123, -0.456, 0.789, ...]}
```

### CSV (.csv) or Text (.txt)
Comma-separated values on a single line:
```
0.123,-0.456,0.789,...
```
Or one value per line.

## Output Format

### JSON Output (default)
```json
{
  "num_drugs_scored": 397,
  "ranking": [
    {
      "rank": 1,
      "drug_index": 120,
      "drug_name": "drug_120",
      "predicted_log_ic50": -1.677
    },
    ...
  ],
  "metrics_note": "Predicted values are log10(IC50). Lower values = more effective drug."
}
```

### CSV Output
```
rank,drug_index,drug_name,predicted_log_ic50,predicted_ic50
1,120,drug_120,-1.6774,0.0210
2,64,drug_64,-1.4252,0.0376
...
```

## Creating Sample Input Files

Use the helper script to generate sample input files:

```bash
# Generate random sample (for testing)
python generate_sample_input.py --output sample.json

# Extract from existing training data
python generate_sample_input.py --from-existing train_data.pt --cell-index 0 --output cell0.json
```

## Interpreting Results

- **Lower IC50 = More effective drug**: The IC50 value represents the drug concentration needed to inhibit 50% of the target. Lower values mean the drug is more potent.
- **log IC50**: Predictions are in log10 scale. A value of -1 means IC50 = 0.1, a value of 0 means IC50 = 1.0.
- **Feature importance**: When using `--explain`, the output includes which gene expression features most influenced the prediction for the top-ranked drug.

## Troubleshooting

### "PyTorch Geometric is required but not installed"
Install with:
```bash
pip install torch-geometric
```

### Feature dimension mismatch
Ensure your input has exactly 19,215 features (gene expression values).

### Missing model files
Ensure all required `.pt` files are in the same directory as the script.
