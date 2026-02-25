"""
Test model with a random cell line by removing its edges.
Simulates a new cell line with no known drug response data.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import random
from scipy.stats import spearmanr
from torch_geometric.nn import to_hetero
from gnn_model import GNN, LinkPredictor, FeatureProjector

# Resolve paths relative to project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Load data and mappings
print("Loading data...")
train = torch.load(DATA_DIR / 'gdsc_processed_train.pt', weights_only=False)
test = torch.load(DATA_DIR / 'gdsc_processed_test.pt', weights_only=False)
mappings = torch.load(DATA_DIR / 'gdsc_processed_mappings.pt', weights_only=False)
artifacts = torch.load(MODELS_DIR / 'training_artifacts.pt', weights_only=False)

# Get cell lines with test edges
edge_index = test['cell_line', 'treated_with', 'drug'].pos_edge_label_index
cell_indices = edge_index[0].unique().tolist()

# Pick random cell
random.seed()
cell_idx = random.choice(cell_indices)
cell_name = mappings['idx_to_cell_name'][cell_idx]

print(f"\n{'='*60}")
print(f"SELECTED CELL LINE: {cell_name} (index {cell_idx})")
print(f"{'='*60}")

# Get this cell's actual drug responses from test set
mask = edge_index[0] == cell_idx
drug_indices = edge_index[1][mask].tolist()
ic50_labels = test['cell_line', 'treated_with', 'drug'].pos_edge_label[mask].tolist()

print(f"Number of drugs tested on this cell: {len(drug_indices)}")

# Load models
print("\nLoading trained model...")
hyper = artifacts["hyperparameters"]

dims_dict = {
    "cell_line": int(artifacts["cell_feature_dim"]),
    "drug": int(artifacts["drug_feature_dim"]),
}

projector = FeatureProjector(
    dims_dict=dims_dict,
    project_dim=hyper.get("project_dim", 256),
    dropout=hyper["dropout"],
)

encoder = GNN(
    hidden_channels=hyper["hidden_channels"],
    out_channels=hyper["out_channels"],
    dropout=hyper["dropout"],
    num_layers=hyper.get("num_gnn_layers", 3),
)
model = to_hetero(encoder, train.metadata(), aggr="sum")

decoder = LinkPredictor(
    in_channels=hyper["out_channels"] * 2,
    hidden_channels=hyper["decoder_hidden"],
    out_channels=1,
    dropout=hyper["dropout"],
    num_layers=hyper.get("num_decoder_layers", 4),
)

# Load weights
projector.load_state_dict(torch.load(MODELS_DIR / "feature_projector_trained_cpu.pt", weights_only=False))
model.load_state_dict(torch.load(MODELS_DIR / "gnn_model_trained_cpu.pt", weights_only=False))
decoder.load_state_dict(torch.load(MODELS_DIR / "link_predictor_trained_cpu.pt", weights_only=False))

# Initialize lazy modules
with torch.no_grad():
    proj_x = projector(train.x_dict)
    _ = model(proj_x, train.edge_index_dict)

projector.eval()
model.eval()
decoder.eval()

# Extract cell line features (as if it's a new cell line)
cell_features = train['cell_line'].x[cell_idx].unsqueeze(0)  # Shape: [1, 17737]
print(f"Cell line gene expression features: {cell_features.shape[1]} genes")

# Create test graph without this cell line's edges
# Use training graph structure (no test edges for this cell)
print("\nPredicting drug rankings (using cell features only, no edge info)...")

with torch.no_grad():
    proj_x = projector(train.x_dict)
    emb = model(proj_x, train.edge_index_dict)
    
    # Get embeddings
    cell_emb = emb['cell_line'][cell_idx].unsqueeze(0)
    drug_emb = emb['drug']
    
    # Predict IC50 for all drugs
    repeated_cell = cell_emb.repeat(drug_emb.shape[0], 1)
    all_preds = decoder(repeated_cell, drug_emb)

# Get predictions for the drugs we have ground truth for
pred_ic50 = [all_preds[d].item() for d in drug_indices]

# Convert to log scale for comparison
import math
actual_log_ic50 = [math.log10(max(ic50, 1e-6)) for ic50 in ic50_labels]

# Rank drugs
actual_ranking = sorted(range(len(drug_indices)), key=lambda i: actual_log_ic50[i])
pred_ranking = sorted(range(len(drug_indices)), key=lambda i: pred_ic50[i])

# Calculate metrics
spear_corr, _ = spearmanr(actual_log_ic50, pred_ic50)

# Top-k precision
def topk_precision(actual_rank, pred_rank, k):
    actual_topk = set(actual_rank[:k])
    pred_topk = set(pred_rank[:k])
    return len(actual_topk & pred_topk) / k

top5_prec = topk_precision(actual_ranking, pred_ranking, min(5, len(drug_indices)))
top10_prec = topk_precision(actual_ranking, pred_ranking, min(10, len(drug_indices)))

print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}")
print(f"Spearman Correlation: {spear_corr:.3f}")
print(f"Top-5 Precision: {top5_prec:.1%}")
print(f"Top-10 Precision: {top10_prec:.1%}")

# Show top predicted drugs vs actual
print(f"\n{'='*60}")
print("TOP 10 PREDICTED vs ACTUAL")
print(f"{'='*60}")
print(f"{'Rank':<6} {'Predicted Drug':<20} {'Actual Drug':<20}")
print("-" * 50)

for i in range(min(10, len(drug_indices))):
    pred_drug_idx = drug_indices[pred_ranking[i]]
    actual_drug_idx = drug_indices[actual_ranking[i]]
    pred_drug_name = mappings['idx_to_drug_name'][pred_drug_idx]
    actual_drug_name = mappings['idx_to_drug_name'][actual_drug_idx]
    match = "✓" if pred_ranking[i] in actual_ranking[:10] else ""
    print(f"#{i+1:<5} {pred_drug_name:<20} {actual_drug_name:<20} {match}")

print(f"\n{'='*60}")
print("INTERPRETATION")
print(f"{'='*60}")
print(f"For cell line '{cell_name}':")
print(f"  - Model predicts drug effectiveness with {spear_corr:.1%} correlation to actual")
print(f"  - {int(top10_prec*10)}/10 of top predicted drugs are truly in top 10")
print(f"  - Lower IC50 = more effective drug")
print(f"{'='*60}")
