"""
Model Explainability for GNN Drug Response Model
- Computes feature importance for cell line (gene) features using gradient-based saliency.
- Outputs top genes contributing to drug response prediction for each cell line-drug pair.
"""

import torch
from torch_geometric.nn import to_hetero
from gnn_model import GNN, LinkPredictor, FeatureProjector
import numpy as np
import argparse
from pathlib import Path
import torch_geometric.data.hetero_data
import torch_geometric.data.storage

torch.serialization.add_safe_globals([
    torch_geometric.data.hetero_data.HeteroData,
    torch_geometric.data.storage.BaseStorage,
    torch_geometric.data.storage.NodeStorage,
    torch_geometric.data.storage.EdgeStorage
])

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def load_model(device, data):
    # Load mappings for feature dimensions
    mappings = torch.load(DATA_DIR / "gdsc_processed_mappings.pt", map_location=device)
    cell_dim = mappings['cell_feature_dim']
    drug_dim = mappings['drug_feature_dim']
    dims_dict = {"cell_line": cell_dim, "drug": drug_dim}

    # Load models
    projector = FeatureProjector(dims_dict, project_dim=256)
    gnn = GNN(hidden_channels=128, out_channels=32, dropout=0.2, num_layers=3)
    link_pred = LinkPredictor(in_channels=64, hidden_channels=128, out_channels=1, dropout=0.2, num_layers=4)
    projector.load_state_dict(torch.load(MODELS_DIR / "feature_projector_trained_cpu.pt", map_location=device))
    # Build hetero GNN with metadata from data
    gnn_hetero = to_hetero(gnn, data.metadata(), aggr="sum").to(device)
    gnn_hetero.load_state_dict(torch.load(MODELS_DIR / "gnn_model_trained_cpu.pt", map_location=device))
    link_pred.load_state_dict(torch.load(MODELS_DIR / "link_predictor_trained_cpu.pt", map_location=device))
    projector.eval()
    gnn_hetero.eval()
    link_pred.eval()
    return projector, gnn_hetero, link_pred, mappings


def explain_cell_drug(cell_idx, drug_idx, data, projector, gnn_hetero, link_pred, mappings, device):
    # Prepare input
    cell_x = data['cell_line'].x.clone().detach().to(device)
    drug_x = data['drug'].x.clone().detach().to(device)
    cell_x.requires_grad = True
    x_dict = {"cell_line": cell_x, "drug": drug_x}
    x_proj = projector(x_dict)
    # Build edge index for single pair
    edge_index = torch.tensor([[cell_idx], [drug_idx]], dtype=torch.long, device=device)
    # Forward pass through GNN
    emb_dict = gnn_hetero(x_proj, data.edge_index_dict)
    cell_emb = emb_dict['cell_line'][cell_idx].unsqueeze(0)
    drug_emb = emb_dict['drug'][drug_idx].unsqueeze(0)
    # Predict
    pred = link_pred(cell_emb, drug_emb)
    pred.backward()
    # Print activations and gradients for debugging
    print("cell_emb:", cell_emb)
    print("drug_emb:", drug_emb)
    print("pred:", pred)
    print("cell_x.grad[{}]:".format(cell_idx), cell_x.grad[cell_idx])
    # Alternate attribution: just use gradient (saliency)
    saliency = cell_x.grad[cell_idx].abs().detach().cpu().numpy()
    # Input x gradient attribution
    input_x_grad = (cell_x.grad[cell_idx].detach() * cell_x[cell_idx].detach()).abs().cpu().numpy()
    gene_names = mappings.get('gene_names', [f"Gene_{i}" for i in range(len(saliency))])
    top_idx = np.argsort(-saliency)[:10]
    top_genes = [(gene_names[i], float(saliency[i]), float(input_x_grad[i])) for i in top_idx]
    return top_genes


def main():
    parser = argparse.ArgumentParser(description="Explain GNN drug response predictions")
    parser.add_argument('--cell-idx', type=int, required=True)
    parser.add_argument('--drug-idx', type=int, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    device = torch.device(args.device)
    data = torch.load(DATA_DIR / "gdsc_processed_test.pt", map_location=device)
    projector, gnn_hetero, link_pred, mappings = load_model(device, data)
    # Explain
    top_genes = explain_cell_drug(args.cell_idx, args.drug_idx, data, projector, gnn_hetero, link_pred, mappings, device)
    print("Top genes contributing to prediction:")
    for gene, saliency, input_x_grad in top_genes:
        print(f"{gene}: saliency={saliency:.4f}, input_x_grad={input_x_grad:.4f}")

if __name__ == "__main__":
    main()
