"""
Evaluate the trained GNN model by simulating new cell lines.

Process:
1. Select a cell line from test set
2. Remove all its edges (drug interactions)
3. Use the model to predict drug rankings
4. Compare predicted rankings with actual rankings (from removed edges)

This simulates the real-world scenario where we have a new cell line
with no known drug response data.
"""

import argparse
import random
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch_geometric.data import HeteroData
from torch_geometric.nn import to_hetero

from gnn_model import GNN, LinkPredictor, FeatureProjector

EDGE_TYPE = ("cell_line", "treated_with", "drug")
REV_EDGE_TYPE = ("drug", "rev_treated_with", "cell_line")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate model on held-out cell lines")
    p.add_argument("--test-data", default="test_data.pt")
    p.add_argument("--train-data", default="train_data.pt")
    p.add_argument("--artifacts", default="training_artifacts.pt")
    p.add_argument("--projector-weights", default="feature_projector_trained_cpu.pt")
    p.add_argument("--model-weights", default="gnn_model_trained_cpu.pt")
    p.add_argument("--decoder-weights", default="link_predictor_trained_cpu.pt")
    p.add_argument("--num-cells-to-test", type=int, default=10, help="Number of cell lines to evaluate")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_models(base_graph: HeteroData, artifacts: dict):
    """Load trained models."""
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
    model = to_hetero(encoder, base_graph.metadata(), aggr="sum")
    
    decoder = LinkPredictor(
        in_channels=hyper["out_channels"] * 2,
        hidden_channels=hyper["decoder_hidden"],
        out_channels=1,
        dropout=hyper["dropout"],
        num_layers=hyper.get("num_decoder_layers", 4),
    )
    
    return projector, model, decoder


def get_cell_line_edges(data: HeteroData, cell_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get all drug indices and IC50 labels for a specific cell line."""
    edge_store = data[EDGE_TYPE]
    edge_index = edge_store.pos_edge_label_index
    labels = edge_store.pos_edge_label
    
    # Find edges where source is this cell line
    mask = edge_index[0] == cell_idx
    drug_indices = edge_index[1, mask]
    ic50_values = labels[mask]
    
    return drug_indices, ic50_values


def remove_cell_line_edges(data: HeteroData, cell_idx: int) -> HeteroData:
    """Remove all edges connected to a specific cell line."""
    new_data = data.clone()
    
    # Remove from forward edges
    if EDGE_TYPE in new_data.edge_types:
        edge_store = new_data[EDGE_TYPE]
        if hasattr(edge_store, 'pos_edge_label_index'):
            edge_index = edge_store.pos_edge_label_index
            labels = edge_store.pos_edge_label
            mask = edge_index[0] != cell_idx
            edge_store.pos_edge_label_index = edge_index[:, mask]
            edge_store.pos_edge_label = labels[mask]
        
        if hasattr(edge_store, 'edge_index'):
            edge_index = edge_store.edge_index
            mask = edge_index[0] != cell_idx
            edge_store.edge_index = edge_index[:, mask]
    
    # Remove from reverse edges
    if REV_EDGE_TYPE in new_data.edge_types:
        edge_store = new_data[REV_EDGE_TYPE]
        if hasattr(edge_store, 'edge_index'):
            edge_index = edge_store.edge_index
            mask = edge_index[1] != cell_idx  # In reverse, cell_line is target
            edge_store.edge_index = edge_index[:, mask]
    
    return new_data


def predict_drug_rankings(
    projector: nn.Module,
    model: nn.Module,
    decoder: nn.Module,
    data: HeteroData,
    cell_idx: int,
) -> torch.Tensor:
    """Predict IC50 for all drugs for a given cell line."""
    projector.eval()
    model.eval()
    decoder.eval()
    
    with torch.no_grad():
        proj_x = projector(data.x_dict)
        emb = model(proj_x, data.edge_index_dict)
        
        cell_emb = emb["cell_line"][cell_idx].unsqueeze(0)
        drug_emb = emb["drug"]  # All drugs
        
        # Predict IC50 for all cell-drug pairs
        repeated_cell = cell_emb.repeat(drug_emb.shape[0], 1)
        preds = decoder(repeated_cell, drug_emb)
    
    return preds


def compute_ranking_metrics(
    pred_ic50: torch.Tensor,
    true_drug_indices: torch.Tensor,
    true_ic50: torch.Tensor,
    k_values: List[int] = [5, 10, 20],
) -> Dict[str, float]:
    """
    Compute ranking metrics comparing predicted vs actual.
    
    Lower IC50 = more effective drug, so we sort ascending.
    """
    # Get predictions only for drugs we have ground truth
    pred_for_known = pred_ic50[true_drug_indices]
    
    # Log transform true IC50 for fair comparison (model predicts log IC50)
    true_log_ic50 = torch.log10(true_ic50.clamp(min=1e-6))
    
    # Spearman correlation
    pred_np = pred_for_known.cpu().numpy()
    true_np = true_log_ic50.cpu().numpy()
    
    if len(pred_np) > 1:
        spearman, _ = spearmanr(pred_np, true_np)
        spearman = float(spearman) if not (spearman != spearman) else 0.0  # Handle NaN
    else:
        spearman = 0.0
    
    # Top-K precision: what fraction of predicted top-K are in true top-K?
    num_drugs = len(true_drug_indices)
    metrics = {"spearman": spearman, "num_drugs_tested": num_drugs}
    
    for k in k_values:
        if num_drugs < k:
            continue
        
        # Get drug indices sorted by predicted IC50 (ascending = most effective first)
        pred_ranking = torch.argsort(pred_for_known)[:k]  # Top K by prediction
        true_ranking = torch.argsort(true_log_ic50)[:k]   # Top K by ground truth
        
        # Convert to sets for intersection
        pred_set = set(pred_ranking.tolist())
        true_set = set(true_ranking.tolist())
        
        precision = len(pred_set & true_set) / k
        metrics[f"top_{k}_precision"] = precision
    
    return metrics


def evaluate_single_cell(
    projector: nn.Module,
    model: nn.Module,
    decoder: nn.Module,
    full_data: HeteroData,
    cell_idx: int,
    verbose: bool = False,
) -> Dict[str, float]:
    """Evaluate model on a single cell line by removing its edges."""
    
    # Step 1: Get ground truth edges for this cell
    true_drug_indices, true_ic50 = get_cell_line_edges(full_data, cell_idx)
    
    if len(true_drug_indices) < 5:
        return None  # Skip cells with too few drugs tested
    
    # Step 2: Create graph with this cell's edges removed
    masked_data = remove_cell_line_edges(full_data, cell_idx)
    
    # Step 3: Predict drug rankings (cell still has features, just no edges)
    pred_ic50 = predict_drug_rankings(projector, model, decoder, masked_data, cell_idx)
    
    # Step 4: Compare predictions with ground truth
    metrics = compute_ranking_metrics(pred_ic50, true_drug_indices, true_ic50)
    
    if verbose:
        print(f"  Cell {cell_idx}: {metrics['num_drugs_tested']} drugs | "
              f"Spearman={metrics['spearman']:.3f} | "
              f"Top-10 Prec={metrics.get('top_10_precision', 'N/A')}")
    
    return metrics


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("=" * 70)
    print("EVALUATING MODEL: HELD-OUT CELL LINE TEST")
    print("=" * 70)
    print("Process: Remove all edges for a cell line, predict rankings,")
    print("         compare with actual drug responses.")
    print("=" * 70)
    
    # Load data and models
    print("\nLoading data...")
    test_data = torch.load(args.test_data, map_location="cpu", weights_only=False)
    train_data = torch.load(args.train_data, map_location="cpu", weights_only=False)
    artifacts = torch.load(args.artifacts, map_location="cpu", weights_only=False)
    
    print("Loading models...")
    projector, model, decoder = load_models(train_data, artifacts)
    
    # Load weights
    projector.load_state_dict(torch.load(args.projector_weights, map_location="cpu", weights_only=False))
    model.load_state_dict(torch.load(args.model_weights, map_location="cpu", weights_only=False))
    decoder.load_state_dict(torch.load(args.decoder_weights, map_location="cpu", weights_only=False))
    
    # Initialize lazy modules
    with torch.no_grad():
        proj_x = projector(train_data.x_dict)
        _ = model(proj_x, train_data.edge_index_dict)
    
    # Find cell lines with edges in test data
    test_edge_index = test_data[EDGE_TYPE].pos_edge_label_index
    unique_cells = test_edge_index[0].unique().tolist()
    
    print(f"\nFound {len(unique_cells)} cell lines with test edges")
    
    # Sample cells to evaluate
    if len(unique_cells) > args.num_cells_to_test:
        cells_to_test = random.sample(unique_cells, args.num_cells_to_test)
    else:
        cells_to_test = unique_cells
    
    print(f"Evaluating {len(cells_to_test)} cell lines...\n")
    
    # Evaluate each cell
    all_metrics = []
    for cell_idx in cells_to_test:
        metrics = evaluate_single_cell(
            projector, model, decoder, test_data, cell_idx, verbose=True
        )
        if metrics is not None:
            all_metrics.append(metrics)
    
    # Aggregate results
    if all_metrics:
        print("\n" + "=" * 70)
        print("AGGREGATE RESULTS (across all tested cell lines)")
        print("=" * 70)
        
        avg_spearman = sum(m["spearman"] for m in all_metrics) / len(all_metrics)
        print(f"  Mean Spearman Correlation: {avg_spearman:.4f}")
        
        for k in [5, 10, 20]:
            key = f"top_{k}_precision"
            values = [m[key] for m in all_metrics if key in m]
            if values:
                avg_prec = sum(values) / len(values)
                print(f"  Mean Top-{k} Precision:     {avg_prec:.4f} ({avg_prec*100:.1f}%)")
        
        avg_drugs = sum(m["num_drugs_tested"] for m in all_metrics) / len(all_metrics)
        print(f"  Avg drugs per cell line:   {avg_drugs:.1f}")
        print("=" * 70)
    else:
        print("No valid cell lines to evaluate!")


if __name__ == "__main__":
    main()
