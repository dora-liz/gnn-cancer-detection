"""
Evaluate model with detailed rankings and explanations for each cell line.

For each test cell line:
1. Remove all its edges (simulate new cell line)
2. Predict drug rankings
3. Generate feature attributions (explainability)
4. Compare with ground truth
5. Output detailed, readable results
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch_geometric.data import HeteroData
from torch_geometric.nn import to_hetero

from gnn_model import GNN, LinkPredictor, FeatureProjector

EDGE_TYPE = ("cell_line", "treated_with", "drug")
REV_EDGE_TYPE = ("drug", "rev_treated_with", "cell_line")

# Resolve paths relative to project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate with detailed explanations")
    p.add_argument("--test-data", default=str(DATA_DIR / "gdsc_processed_test.pt"))
    p.add_argument("--train-data", default=str(DATA_DIR / "gdsc_processed_train.pt"))
    p.add_argument("--artifacts", default=str(MODELS_DIR / "training_artifacts.pt"))
    p.add_argument("--projector-weights", default=str(MODELS_DIR / "feature_projector_trained_cpu.pt"))
    p.add_argument("--model-weights", default=str(MODELS_DIR / "gnn_model_trained_cpu.pt"))
    p.add_argument("--decoder-weights", default=str(MODELS_DIR / "link_predictor_trained_cpu.pt"))
    p.add_argument("--num-cells", type=int, default=5, help="Number of cell lines to evaluate")
    p.add_argument("--top-k-drugs", type=int, default=10, help="Show top K drugs")
    p.add_argument("--top-k-features", type=int, default=10, help="Show top K features per drug")
    p.add_argument("--output", default="evaluation_results.json", help="Output JSON file")
    p.add_argument("--gene-names", default="", help="Optional gene names JSON file")
    p.add_argument("--drug-names", default="", help="Optional drug names JSON file")
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


def load_names(path: str, count: int, prefix: str) -> List[str]:
    """Load names from JSON or generate defaults."""
    if path and Path(path).exists():
        names = json.loads(Path(path).read_text())
        if len(names) >= count:
            return names[:count]
    return [f"{prefix}_{i}" for i in range(count)]


def get_cell_line_edges(data: HeteroData, cell_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get drug indices and IC50 labels for a cell line."""
    edge_store = data[EDGE_TYPE]
    edge_index = edge_store.pos_edge_label_index
    labels = edge_store.pos_edge_label
    
    mask = edge_index[0] == cell_idx
    drug_indices = edge_index[1, mask]
    ic50_values = labels[mask]
    
    return drug_indices, ic50_values


def remove_cell_line_edges(data: HeteroData, cell_idx: int) -> HeteroData:
    """Remove all edges for a cell line."""
    new_data = data.clone()
    
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
    
    if REV_EDGE_TYPE in new_data.edge_types:
        edge_store = new_data[REV_EDGE_TYPE]
        if hasattr(edge_store, 'edge_index'):
            edge_index = edge_store.edge_index
            mask = edge_index[1] != cell_idx
            edge_store.edge_index = edge_index[:, mask]
    
    return new_data


def predict_all_drugs(
    projector: nn.Module,
    model: nn.Module,
    decoder: nn.Module,
    data: HeteroData,
    cell_idx: int,
) -> torch.Tensor:
    """Predict IC50 for all drugs."""
    projector.eval()
    model.eval()
    decoder.eval()
    
    with torch.no_grad():
        proj_x = projector(data.x_dict)
        emb = model(proj_x, data.edge_index_dict)
        
        cell_emb = emb["cell_line"][cell_idx].unsqueeze(0)
        drug_emb = emb["drug"]
        
        repeated_cell = cell_emb.repeat(drug_emb.shape[0], 1)
        preds = decoder(repeated_cell, drug_emb)
    
    return preds


def explain_drug_prediction(
    projector: nn.Module,
    model: nn.Module,
    decoder: nn.Module,
    data: HeteroData,
    cell_idx: int,
    drug_idx: int,
) -> Tuple[float, torch.Tensor]:
    """Compute feature attribution for a specific drug prediction."""
    explain_data = data.clone()
    explain_data["cell_line"].x = explain_data["cell_line"].x.clone().detach()
    explain_data["cell_line"].x.requires_grad_(True)
    
    projector.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)
    decoder.zero_grad(set_to_none=True)
    
    proj_x = projector(explain_data.x_dict)
    emb = model(proj_x, explain_data.edge_index_dict)
    
    cell_emb = emb["cell_line"][cell_idx].unsqueeze(0)
    drug_emb = emb["drug"][drug_idx].unsqueeze(0)
    
    pred = decoder(cell_emb, drug_emb).squeeze()
    pred.backward()
    
    grad = explain_data["cell_line"].x.grad[cell_idx]
    inp = data["cell_line"].x[cell_idx]
    attribution = (grad * inp).abs()
    
    return pred.detach().item(), attribution.detach()


def evaluate_cell_line(
    projector: nn.Module,
    model: nn.Module,
    decoder: nn.Module,
    data: HeteroData,
    cell_idx: int,
    drug_names: List[str],
    gene_names: List[str],
    top_k_drugs: int,
    top_k_features: int,
) -> Dict:
    """Full evaluation for one cell line with explanations."""
    
    # Get ground truth
    true_drug_indices, true_ic50 = get_cell_line_edges(data, cell_idx)
    if len(true_drug_indices) < 5:
        return None
    
    true_log_ic50 = torch.log10(true_ic50.clamp(min=1e-6))
    
    # Remove edges and predict
    masked_data = remove_cell_line_edges(data, cell_idx)
    pred_ic50 = predict_all_drugs(projector, model, decoder, masked_data, cell_idx)
    
    # Get predictions for known drugs only
    pred_for_known = pred_ic50[true_drug_indices]
    
    # Compute Spearman
    pred_np = pred_for_known.cpu().numpy()
    true_np = true_log_ic50.cpu().numpy()
    spearman = float(spearmanr(pred_np, true_np)[0]) if len(pred_np) > 1 else 0.0
    
    # Sort predictions for drugs with ground truth (lowest IC50 = most effective)
    pred_sorted_indices = torch.argsort(pred_for_known)
    top_predicted_local = pred_sorted_indices[:top_k_drugs]
    
    # Get actual top-K drugs from ground truth
    true_sorted = torch.argsort(true_log_ic50)
    actual_top_local = true_sorted[:top_k_drugs]
    
    # Top-K precision
    pred_set = set(top_predicted_local.tolist())
    actual_set = set(actual_top_local.tolist())
    top_k_precision = len(pred_set & actual_set) / top_k_drugs
    
    # Build detailed results with explanations
    drug_rankings = []
    for rank, local_idx in enumerate(top_predicted_local.tolist(), 1):
        drug_idx = true_drug_indices[local_idx].item()
        pred_log = pred_for_known[local_idx].item()
        pred_ic50_val = 10 ** pred_log
        
        actual_ic50 = true_ic50[local_idx].item()
        actual_rank = (true_sorted == local_idx).nonzero(as_tuple=True)[0].item() + 1
        
        # Get explanation for this drug
        _, attribution = explain_drug_prediction(
            projector, model, decoder, masked_data, cell_idx, drug_idx
        )
        
        # Top contributing features
        top_feat_indices = torch.argsort(attribution, descending=True)[:top_k_features]
        top_features = []
        for feat_idx in top_feat_indices:
            feat_idx = feat_idx.item()
            top_features.append({
                "feature_index": feat_idx,
                "feature_name": gene_names[feat_idx] if feat_idx < len(gene_names) else f"gene_{feat_idx}",
                "importance": round(attribution[feat_idx].item(), 6),
                "feature_value": round(data["cell_line"].x[cell_idx, feat_idx].item(), 6),
            })
        
        is_correct = actual_rank <= top_k_drugs
        
        drug_rankings.append({
            "predicted_rank": rank,
            "drug_index": drug_idx,
            "drug_name": drug_names[drug_idx] if drug_idx < len(drug_names) else f"drug_{drug_idx}",
            "predicted_log_ic50": round(pred_log, 4),
            "predicted_ic50": round(pred_ic50_val, 4),
            "actual_ic50": round(actual_ic50, 4),
            "actual_rank": actual_rank,
            "is_correct_top_k": is_correct,
            "explanation": {
                "summary": f"Prediction driven by {top_k_features} key gene expression features",
                "top_contributing_features": top_features,
            }
        })
    
    return {
        "cell_line_index": cell_idx,
        "num_drugs_in_ground_truth": len(true_drug_indices),
        "metrics": {
            "spearman_correlation": round(spearman, 4),
            "top_k_precision": round(top_k_precision, 4),
            "interpretation": interpret_spearman(spearman),
        },
        "predicted_drug_rankings": drug_rankings,
    }


def interpret_spearman(s: float) -> str:
    """Human-readable interpretation of Spearman correlation."""
    if s >= 0.9:
        return "Excellent - near-perfect ranking agreement"
    elif s >= 0.8:
        return "Very Good - strong ranking agreement"
    elif s >= 0.7:
        return "Good - moderate ranking agreement"
    elif s >= 0.5:
        return "Fair - some ranking agreement"
    else:
        return "Weak - limited ranking agreement"


def print_cell_result(result: Dict):
    """Pretty print results for one cell line with simple explanations."""
    print("\n" + "=" * 80)
    print(f"PATIENT/CELL LINE: {result['cell_line_index']}")
    print("=" * 80)
    
    # Simple accuracy explanation
    accuracy = result['metrics']['top_k_precision'] * 100
    spearman = result['metrics']['spearman_correlation']
    
    print(f"\nHow good is our prediction?")
    print(f"  - We correctly identified {accuracy:.0f}% of the best drugs")
    print(f"  - Our drug ranking matches {spearman*100:.0f}% with actual results")
    
    print("\n" + "=" * 80)
    print("DRUG RECOMMENDATIONS")
    print("(Lower drug score = drug works better = less drug needed to kill cancer)")
    print("=" * 80)
    
    for drug in result["predicted_drug_rankings"]:
        print(f"\n{'─'*70}")
        
        # Simple rank display
        if drug['is_correct_top_k']:
            print(f"  #{drug['predicted_rank']} {drug['drug_name']} ✓ GOOD PREDICTION!")
        else:
            print(f"  #{drug['predicted_rank']} {drug['drug_name']}")
        
        print(f"{'─'*70}")
        
        # Simple effectiveness explanation
        pred = drug['predicted_ic50']
        actual = drug['actual_ic50']
        
        print(f"\n  What we predicted: Score = {pred:.2f}")
        print(f"  What actually happened: Score = {actual:.2f} (was actually #{drug['actual_rank']} best drug)")
        
        # Plain English interpretation
        if pred < 1:
            print(f"\n  >> We predicted: This drug should work VERY WELL")
        elif pred < 5:
            print(f"\n  >> We predicted: This drug should work MODERATELY")
        else:
            print(f"\n  >> We predicted: This drug may NOT work well")
        
        if actual < 1:
            print(f"  >> Reality: This drug actually works VERY WELL")
        elif actual < 5:
            print(f"  >> Reality: This drug actually works MODERATELY")
        else:
            print(f"  >> Reality: This drug actually does NOT work well")
        
        # Simple gene explanation
        print(f"\n  WHY DID WE RECOMMEND THIS DRUG?")
        print(f"  ─────────────────────────────────")
        
        features = drug['explanation']['top_contributing_features'][:3]
        
        print(f"\n  The model looked at the genes in this cell line and found:")
        
        for i, feat in enumerate(features, 1):
            gene = feat['feature_name']
            expression = feat['feature_value']
            
            # Very simple explanation
            if expression > 9:
                level = "HIGH"
                meaning = "This gene is very active in this cell"
            elif expression > 6:
                level = "MEDIUM" 
                meaning = "This gene is moderately active"
            else:
                level = "LOW"
                meaning = "This gene is not very active"
            
            print(f"\n  Gene {i}: {gene}")
            print(f"    Activity Level: {level} ({expression:.1f})")
            print(f"    What this means: {meaning}")
        
        # Simple summary
        genes = [f['feature_name'] for f in features[:3]]
        print(f"\n  SIMPLE EXPLANATION:")
        print(f"  ───────────────────")
        print(f"  This drug was recommended because the activity patterns of")
        print(f"  {genes[0]}, {genes[1]}, and {genes[2]}")
        print(f"  suggest this cell line will respond to {drug['drug_name']}.")
        
        if drug['is_correct_top_k']:
            print(f"\n  ✓ SUCCESS: Our prediction was correct! This drug IS one of the best.")
        else:
            rank_diff = abs(drug['actual_rank'] - drug['predicted_rank'])
            if rank_diff <= 3:
                print(f"\n  ○ CLOSE: We were close - drug is ranked #{drug['actual_rank']} (we predicted #{drug['predicted_rank']})")
            else:
                print(f"\n  ✗ MISSED: Drug is actually ranked #{drug['actual_rank']} (we predicted #{drug['predicted_rank']})")


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("=" * 80)
    print("DRUG RANKING EVALUATION WITH EXPLANATIONS")
    print("=" * 80)
    
    # Load data
    print("\nLoading data and models...")
    test_data = torch.load(args.test_data, map_location="cpu", weights_only=False)
    train_data = torch.load(args.train_data, map_location="cpu", weights_only=False)
    artifacts = torch.load(args.artifacts, map_location="cpu", weights_only=False)
    
    projector, model, decoder = load_models(train_data, artifacts)
    
    projector.load_state_dict(torch.load(args.projector_weights, map_location="cpu", weights_only=False))
    model.load_state_dict(torch.load(args.model_weights, map_location="cpu", weights_only=False))
    decoder.load_state_dict(torch.load(args.decoder_weights, map_location="cpu", weights_only=False))
    
    # Initialize lazy modules
    with torch.no_grad():
        proj_x = projector(train_data.x_dict)
        _ = model(proj_x, train_data.edge_index_dict)
    
    # Load names
    num_drugs = int(artifacts["num_drugs"])
    num_features = int(artifacts["cell_feature_dim"])
    drug_names = load_names(args.drug_names, num_drugs, "drug")
    gene_names = load_names(args.gene_names, num_features, "gene")
    
    # Find cell lines to test
    test_edge_index = test_data[EDGE_TYPE].pos_edge_label_index
    unique_cells = test_edge_index[0].unique().tolist()
    
    if len(unique_cells) > args.num_cells:
        cells_to_test = random.sample(unique_cells, args.num_cells)
    else:
        cells_to_test = unique_cells
    
    print(f"\nEvaluating {len(cells_to_test)} cell lines with detailed explanations...\n")
    
    # Evaluate each cell
    all_results = []
    for cell_idx in cells_to_test:
        result = evaluate_cell_line(
            projector, model, decoder, test_data, cell_idx,
            drug_names, gene_names, args.top_k_drugs, args.top_k_features
        )
        if result:
            all_results.append(result)
            print_cell_result(result)
    
    # Aggregate summary
    if all_results:
        print("\n" + "=" * 80)
        print("SUMMARY ACROSS ALL CELL LINES")
        print("=" * 80)
        
        avg_spearman = sum(r["metrics"]["spearman_correlation"] for r in all_results) / len(all_results)
        avg_precision = sum(r["metrics"]["top_k_precision"] for r in all_results) / len(all_results)
        
        print(f"  Cell lines evaluated: {len(all_results)}")
        print(f"  Mean Spearman: {avg_spearman:.4f}")
        print(f"  Mean Top-{args.top_k_drugs} Precision: {avg_precision*100:.1f}%")
        print("=" * 80)
    
    # Save to JSON
    output = {
        "summary": {
            "num_cells_evaluated": len(all_results),
            "mean_spearman": round(avg_spearman, 4) if all_results else 0,
            "mean_top_k_precision": round(avg_precision, 4) if all_results else 0,
        },
        "cell_line_results": all_results,
    }
    
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
