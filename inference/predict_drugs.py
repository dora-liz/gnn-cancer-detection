#!/usr/bin/env python3
"""
Drug Ranking Prediction for New Cell Lines

This standalone script ranks drugs for a new cell line based on predicted IC50 values
using a pre-trained GNN model. It's designed to be portable and work on any system
when the required model files are copied together.

REQUIRED FILES (in project structure):
    - ../gnn_model.py                             : Model architecture definitions
    - ../data/processed/gdsc_processed_train.pt   : Base graph with drug/cell line structure
    - ../models/training_artifacts.pt             : Model hyperparameters
    - ../models/feature_projector_trained_cpu.pt  : Trained feature projector weights
    - ../models/gnn_model_trained_cpu.pt          : Trained GNN encoder weights
    - ../models/link_predictor_trained_cpu.pt     : Trained link predictor weights

USAGE EXAMPLES:
    1. From a PyTorch tensor file (.pt):
       python predict_drugs.py --input cell_features.pt

    2. From a JSON file with gene expression array:
       python predict_drugs.py --input cell_expression.json

    3. From CSV file (values on single line or column):
       python predict_drugs.py --input cell_expression.csv
       
    4. With custom output file:
       python predict_drugs.py --input cell_features.pt --output results.json

    5. Get more drugs in ranking:
       python predict_drugs.py --input cell_features.pt --top-k 50

    6. Include feature explanations:
       python predict_drugs.py --input cell_features.pt --explain

INPUT FORMAT:
    The input should contain gene expression values for a single cell line.
    Expected number of features: 17,737 (will be validated against trained model)

    Supported formats:
    - .pt  : PyTorch tensor (1D or 2D with shape [1, num_features])
    - .json: JSON array of floats, e.g., [0.1, 0.2, 0.3, ...]
    - .csv : Comma-separated values (single row or single column)
    - .txt : Same as CSV

OUTPUT:
    JSON file with ranked drugs (lower predicted IC50 = more effective):
    {
        "num_drugs_scored": 345,
        "ranking": [
            {"rank": 1, "drug_index": 42, "drug_name": "drug_42", "predicted_log_ic50": -1.23},
            ...
        ],
        "explainability": {...}  // If --explain flag is used
    }

DEPENDENCIES:
    pip install torch torch-geometric

Author: Generated for portable drug ranking predictions
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

# Try to import torch_geometric, provide helpful error if not available
try:
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import to_hetero
except ImportError:
    print("ERROR: PyTorch Geometric is required but not installed.")
    print("Install with: pip install torch-geometric")
    print("\nFor CPU-only installation:")
    print("  pip install torch-geometric")
    print("\nFor specific PyTorch version compatibility, see:")
    print("  https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html")
    sys.exit(1)

# Set up paths relative to project structure
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Add parent directory to import gnn_model
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from gnn_model import GNN, LinkPredictor, FeatureProjector
except ImportError:
    # Try importing from script directory (portable mode)
    sys.path.insert(0, str(SCRIPT_DIR))
    try:
        from gnn_model import GNN, LinkPredictor, FeatureProjector
    except ImportError:
        print(f"ERROR: Could not import gnn_model.py")
        print(f"Searched in: {PROJECT_ROOT} and {SCRIPT_DIR}")
        sys.exit(1)


# Constants
EDGE_TYPE = ("cell_line", "treated_with", "drug")
REV_EDGE_TYPE = ("drug", "rev_treated_with", "cell_line")


def get_file_paths(model_dir: Optional[Path] = None) -> Dict[str, Path]:
    """Get file paths based on project structure or portable mode."""
    if model_dir:
        # Portable mode: all files in one directory
        return {
            "train_data": model_dir / "gdsc_processed_train.pt",
            "artifacts": model_dir / "training_artifacts.pt",
            "projector": model_dir / "feature_projector_trained_cpu.pt",
            "gnn": model_dir / "gnn_model_trained_cpu.pt",
            "decoder": model_dir / "link_predictor_trained_cpu.pt",
        }
    else:
        # Project structure mode
        return {
            "train_data": DATA_DIR / "gdsc_processed_train.pt",
            "artifacts": MODELS_DIR / "training_artifacts.pt",
            "projector": MODELS_DIR / "feature_projector_trained_cpu.pt",
            "gnn": MODELS_DIR / "gnn_model_trained_cpu.pt",
            "decoder": MODELS_DIR / "link_predictor_trained_cpu.pt",
        }


def check_required_files(paths: Dict[str, Path]) -> bool:
    """Check if all required model files exist."""
    missing = []
    for name, fpath in paths.items():
        if not fpath.exists():
            missing.append(str(fpath))
    
    if missing:
        print("ERROR: Missing required model files:")
        for f in missing:
            print(f"  - {f}")
        return False
    return True


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Rank drugs for a new cell line using trained GNN model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict_drugs.py --input cell_features.pt
  python predict_drugs.py --input expression.json --top-k 50
  python predict_drugs.py --input data.csv --output my_results.json --explain
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to cell line features file (.pt, .json, .csv, or .txt)"
    )
    
    parser.add_argument(
        "--output", "-o",
        default="drug_ranking_results.json",
        help="Output file path for results (default: drug_ranking_results.json)"
    )
    
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=20,
        help="Number of top drugs to include in ranking (default: 20)"
    )
    
    parser.add_argument(
        "--drug-names",
        default="",
        help="Optional JSON file mapping drug indices to names"
    )
    
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Include feature importance explanations for top drug"
    )
    
    parser.add_argument(
        "--explain-features",
        type=int,
        default=15,
        help="Number of top features to show in explanation (default: 15)"
    )
    
    parser.add_argument(
        "--model-dir",
        default="",
        help="Directory containing model files (default: same as script)"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress console output (only write to output file)"
    )
    
    parser.add_argument(
        "--format",
        choices=["json", "csv", "text"],
        default="json",
        help="Output format (default: json)"
    )
    
    return parser.parse_args()


def load_cell_features(path: str) -> torch.Tensor:
    """
    Load cell line features from various file formats.
    
    Args:
        path: Path to the feature file (.pt, .json, .csv, .txt)
        
    Returns:
        Tensor of shape [1, num_features]
    """
    p = Path(path)
    
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    
    suffix = p.suffix.lower()
    
    if suffix == ".pt":
        # PyTorch tensor file
        obj = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(obj, torch.Tensor):
            features = obj.float()
        elif isinstance(obj, dict) and "features" in obj:
            features = torch.tensor(obj["features"], dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported .pt file content type: {type(obj)}")
            
    elif suffix == ".json":
        # JSON file with array
        content = p.read_text(encoding="utf-8")
        data = json.loads(content)
        
        if isinstance(data, list):
            features = torch.tensor(data, dtype=torch.float32)
        elif isinstance(data, dict):
            # Try common key names
            for key in ["features", "gene_expression", "expression", "values", "data"]:
                if key in data:
                    features = torch.tensor(data[key], dtype=torch.float32)
                    break
            else:
                raise ValueError("JSON must be an array or dict with 'features'/'gene_expression' key")
        else:
            raise ValueError(f"Unsupported JSON content type: {type(data)}")
            
    elif suffix in [".csv", ".txt"]:
        # CSV or text file
        content = p.read_text(encoding="utf-8").strip()
        
        # Try parsing as comma-separated values
        if "," in content:
            # Single line or multiple lines with commas
            values = []
            for line in content.split("\n"):
                line = line.strip()
                if line and not line.startswith("#"):
                    values.extend([float(x.strip()) for x in line.split(",") if x.strip()])
        else:
            # One value per line
            values = []
            for line in content.split("\n"):
                line = line.strip()
                if line and not line.startswith("#"):
                    try:
                        values.append(float(line))
                    except ValueError:
                        continue
        
        if not values:
            raise ValueError("Could not parse any numeric values from file")
        
        features = torch.tensor(values, dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .pt, .json, .csv, or .txt")
    
    # Ensure correct shape [1, num_features]
    if features.ndim == 1:
        features = features.unsqueeze(0)
    
    if features.shape[0] != 1:
        raise ValueError(f"Expected single cell line features, got shape {tuple(features.shape)}")
    
    return features


def build_augmented_graph(base_data: HeteroData, new_cell_features: torch.Tensor) -> Tuple[HeteroData, int]:
    """
    Add a new cell line to the graph.
    
    Args:
        base_data: Original graph with existing cell lines and drugs
        new_cell_features: Features for the new cell line [1, num_features]
        
    Returns:
        Tuple of (augmented graph, index of new cell line)
    """
    data = base_data.clone()
    
    old_x = data["cell_line"].x
    new_index = old_x.shape[0]
    
    # Append new cell line features
    data["cell_line"].x = torch.cat([old_x, new_cell_features], dim=0)
    
    # Clone edge indices to avoid modifying original
    if EDGE_TYPE in data.edge_types:
        et_store = data[EDGE_TYPE]
        if hasattr(et_store, "edge_index"):
            et_store.edge_index = et_store.edge_index.clone()
    
    if REV_EDGE_TYPE in data.edge_types:
        rt_store = data[REV_EDGE_TYPE]
        if hasattr(rt_store, "edge_index"):
            rt_store.edge_index = rt_store.edge_index.clone()
    
    return data, new_index


def load_drug_names(num_drugs: int, path: str) -> List[str]:
    """Load drug names from JSON file or generate default names."""
    if path and Path(path).exists():
        try:
            items = json.loads(Path(path).read_text(encoding="utf-8"))
            if isinstance(items, list) and len(items) == num_drugs:
                return [str(x) for x in items]
            elif isinstance(items, dict):
                # Dict mapping index to name
                return [items.get(str(i), f"drug_{i}") for i in range(num_drugs)]
        except Exception:
            pass
    
    return [f"drug_{i}" for i in range(num_drugs)]


def create_models(base_graph: HeteroData, artifacts: dict):
    """
    Create model instances with correct architecture.
    
    Args:
        base_graph: Graph with metadata for to_hetero conversion
        artifacts: Training artifacts with hyperparameters
        
    Returns:
        Tuple of (projector, model, decoder)
    """
    hyper = artifacts["hyperparameters"]
    
    # Feature dimensions
    dims_dict = {
        "cell_line": int(artifacts["cell_feature_dim"]),
        "drug": int(artifacts["drug_feature_dim"]),
    }
    
    # Feature projector
    projector = FeatureProjector(
        dims_dict=dims_dict,
        project_dim=hyper.get("project_dim", 256),
        dropout=hyper["dropout"],
    )
    
    # GNN encoder (converted to heterogeneous)
    encoder = GNN(
        hidden_channels=hyper["hidden_channels"],
        out_channels=hyper["out_channels"],
        dropout=hyper["dropout"],
        num_layers=hyper.get("num_gnn_layers", 3),
    )
    model = to_hetero(encoder, base_graph.metadata(), aggr="sum")
    
    # Link predictor (decoder)
    decoder = LinkPredictor(
        in_channels=hyper["out_channels"] * 2,
        hidden_channels=hyper["decoder_hidden"],
        out_channels=1,
        dropout=hyper["dropout"],
        num_layers=hyper.get("num_decoder_layers", 4),
    )
    
    return projector, model, decoder


def rank_drugs(
    projector,
    model,
    decoder,
    graph: HeteroData,
    new_cell_index: int
) -> torch.Tensor:
    """
    Compute drug rankings for the new cell line.
    
    Args:
        projector: Feature projector model
        model: GNN encoder
        decoder: Link predictor
        graph: Augmented graph with new cell line
        new_cell_index: Index of the new cell line in the graph
        
    Returns:
        Tensor of predicted log IC50 values for each drug
    """
    projector.eval()
    model.eval()
    decoder.eval()
    
    with torch.no_grad():
        # Project features
        proj_x = projector(graph.x_dict)
        
        # Get embeddings
        emb = model(proj_x, graph.edge_index_dict)
        
        # Get new cell line embedding
        new_cell_emb = emb["cell_line"][new_cell_index].unsqueeze(0)
        
        # Get all drug embeddings
        drug_emb = emb["drug"]
        
        # Predict IC50 for all cell-drug pairs
        repeated_cell = new_cell_emb.repeat(drug_emb.shape[0], 1)
        preds = decoder(repeated_cell, drug_emb)
    
    return preds


def explain_prediction(
    projector,
    model,
    decoder,
    graph: HeteroData,
    new_cell_features: torch.Tensor,
    drug_idx: int,
    base_cell_count: int,
) -> Tuple[float, torch.Tensor]:
    """
    Compute feature importance for a specific drug prediction.
    
    Uses input×gradient attribution method.
    
    Args:
        projector, model, decoder: Model components
        graph: Augmented graph
        new_cell_features: Original cell line features
        drug_idx: Index of drug to explain
        base_cell_count: Number of cell lines in original graph
        
    Returns:
        Tuple of (prediction value, attribution scores for each feature)
    """
    # Clone graph for gradient computation
    explain_graph = graph.clone()
    explain_graph["cell_line"].x = explain_graph["cell_line"].x.clone().detach()
    explain_graph["cell_line"].x.requires_grad_(True)
    
    # Clear gradients
    projector.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)
    decoder.zero_grad(set_to_none=True)
    
    # Forward pass
    proj_x = projector(explain_graph.x_dict)
    emb = model(proj_x, explain_graph.edge_index_dict)
    new_cell_emb = emb["cell_line"][base_cell_count].unsqueeze(0)
    drug_emb = emb["drug"][drug_idx].unsqueeze(0)
    pred = decoder(new_cell_emb, drug_emb).squeeze()
    
    # Backward pass
    pred.backward()
    
    # Compute attributions
    grad = explain_graph["cell_line"].x.grad[base_cell_count]
    inp = new_cell_features.squeeze(0)
    attribution = (grad * inp).abs()
    
    return pred.detach().item(), attribution.detach().cpu()


def format_output_json(ranking: List[dict], explanation: Optional[dict], num_drugs: int) -> dict:
    """Format output as JSON structure."""
    output = {
        "num_drugs_scored": num_drugs,
        "ranking": ranking,
        "metrics_note": "Predicted values are log10(IC50). Lower values = more effective drug.",
    }
    
    if explanation:
        output["explainability"] = explanation
    
    return output


def format_output_csv(ranking: List[dict]) -> str:
    """Format output as CSV."""
    lines = ["rank,drug_index,drug_name,predicted_log_ic50,predicted_ic50"]
    for r in ranking:
        ic50 = 10 ** r["predicted_log_ic50"]
        lines.append(f"{r['rank']},{r['drug_index']},{r['drug_name']},{r['predicted_log_ic50']:.4f},{ic50:.4f}")
    return "\n".join(lines)


def format_output_text(ranking: List[dict], explanation: Optional[dict]) -> str:
    """Format output as human-readable text."""
    lines = [
        "=" * 60,
        "DRUG RANKING RESULTS",
        "=" * 60,
        "",
        f"Total drugs scored: {len(ranking) if ranking else 0}",
        "",
        "Top Drugs (ranked by predicted effectiveness):",
        "-" * 60,
        f"{'Rank':<6} {'Drug':<20} {'Log IC50':<12} {'IC50':<12}",
        "-" * 60,
    ]
    
    for r in ranking[:20]:
        ic50 = 10 ** r["predicted_log_ic50"]
        lines.append(f"{r['rank']:<6} {r['drug_name']:<20} {r['predicted_log_ic50']:<12.4f} {ic50:<12.4f}")
    
    if explanation:
        lines.extend([
            "",
            "=" * 60,
            "FEATURE IMPORTANCE (for top drug)",
            "=" * 60,
            f"Top drug: {explanation.get('top_drug_name', 'N/A')}",
            "",
            "Most influential gene expression features:",
        ])
        for attr in explanation.get("top_feature_attributions", [])[:10]:
            lines.append(f"  Feature {attr['feature_index']}: importance = {attr['importance']:.6f}")
    
    lines.extend(["", "=" * 60, "Note: Lower IC50 = more effective drug", "=" * 60])
    
    return "\n".join(lines)


def main():
    """Main entry point."""
    args = parse_args()
    
    # Determine model directory and get file paths
    if args.model_dir:
        model_dir = Path(args.model_dir).resolve()
        paths = get_file_paths(model_dir)
    else:
        model_dir = None
        paths = get_file_paths()
    
    # Check required files
    if not check_required_files(paths):
        sys.exit(1)
    
    # Load cell line features
    if not args.quiet:
        print(f"Loading cell line features from: {args.input}")
    
    try:
        new_cell_features = load_cell_features(args.input)
    except Exception as e:
        print(f"ERROR loading input file: {e}")
        sys.exit(1)
    
    if not args.quiet:
        print(f"  Feature shape: {tuple(new_cell_features.shape)}")
    
    # Load base graph and artifacts
    if not args.quiet:
        print("Loading model files...")
    
    base_graph = torch.load(paths["train_data"], map_location="cpu", weights_only=False)
    artifacts = torch.load(paths["artifacts"], map_location="cpu", weights_only=False)
    
    # Validate feature dimensions
    expected_dim = int(artifacts["cell_feature_dim"])
    if new_cell_features.shape[1] != expected_dim:
        print(f"ERROR: Feature dimension mismatch!")
        print(f"  Expected: {expected_dim} features")
        print(f"  Got: {new_cell_features.shape[1]} features")
        sys.exit(1)
    
    # Build augmented graph
    graph, new_cell_idx = build_augmented_graph(base_graph, new_cell_features)
    
    # Create and load models
    projector, model, decoder = create_models(base_graph, artifacts)
    
    projector.load_state_dict(
        torch.load(paths["projector"], map_location="cpu", weights_only=False)
    )
    model.load_state_dict(
        torch.load(paths["gnn"], map_location="cpu", weights_only=False)
    )
    decoder.load_state_dict(
        torch.load(paths["decoder"], map_location="cpu", weights_only=False)
    )
    
    # Initialize lazy modules with a forward pass
    with torch.no_grad():
        proj_x = projector(graph.x_dict)
        _ = model(proj_x, graph.edge_index_dict)
    
    if not args.quiet:
        print("Ranking drugs...")
    
    # Get predictions
    preds = rank_drugs(projector, model, decoder, graph, new_cell_idx)
    
    # Load drug names
    num_drugs = int(preds.shape[0])
    drug_names = load_drug_names(num_drugs, args.drug_names)
    
    # Sort by predicted IC50 (lower = better)
    top_k = min(args.top_k, num_drugs)
    ranked_idx = torch.argsort(preds, descending=False)[:top_k]
    
    # Build ranking list
    ranking = []
    for i, idx in enumerate(ranked_idx.tolist(), start=1):
        ranking.append({
            "rank": i,
            "drug_index": int(idx),
            "drug_name": drug_names[idx],
            "predicted_log_ic50": float(preds[idx].item()),
        })
    
    # Generate explanations if requested
    explanation = None
    if args.explain:
        if not args.quiet:
            print("Computing feature importance explanations...")
        
        top_drug_idx = int(ranked_idx[0].item())
        top_pred, attributions = explain_prediction(
            projector=projector,
            model=model,
            decoder=decoder,
            graph=graph,
            new_cell_features=new_cell_features,
            drug_idx=top_drug_idx,
            base_cell_count=base_graph["cell_line"].x.shape[0],
        )
        
        # Get top features by attribution
        explain_k = min(args.explain_features, attributions.shape[0])
        feat_idx = torch.argsort(attributions, descending=True)[:explain_k]
        
        explanation = {
            "top_drug_index": top_drug_idx,
            "top_drug_name": drug_names[top_drug_idx],
            "top_drug_predicted_log_ic50": float(top_pred),
            "method": "input_x_gradient_attribution",
            "interpretation": (
                "Lower IC50 = stronger drug effect. "
                "Feature attributions show which cell line gene expression features "
                "most influenced the predicted sensitivity to the top-ranked drug."
            ),
            "top_feature_attributions": [
                {"feature_index": int(fi), "importance": float(attributions[fi].item())}
                for fi in feat_idx.tolist()
            ],
        }
    
    # Format and save output
    output_path = Path(args.output)
    
    if args.format == "json":
        output_data = format_output_json(ranking, explanation, num_drugs)
        output_path.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
    elif args.format == "csv":
        output_text = format_output_csv(ranking)
        output_path.write_text(output_text, encoding="utf-8")
    else:  # text
        output_text = format_output_text(ranking, explanation)
        output_path.write_text(output_text, encoding="utf-8")
    
    # Print results
    if not args.quiet:
        print()
        print("=" * 60)
        print("DRUG RANKING COMPLETE")
        print("=" * 60)
        print(f"Output saved to: {output_path}")
        print(f"\nTop {min(5, top_k)} drugs (lowest predicted IC50 = most effective):")
        print("-" * 60)
        
        for r in ranking[:5]:
            ic50 = 10 ** r["predicted_log_ic50"]
            print(f"  #{r['rank']}: {r['drug_name']:<15} | log IC50: {r['predicted_log_ic50']:>8.4f} | IC50: {ic50:>10.4f}")
        
        if top_k > 5:
            print(f"  ... and {top_k - 5} more in output file")
        
        if explanation:
            print(f"\nTop drug: {explanation['top_drug_name']}")
            print(f"Feature importance for top {args.explain_features} features saved.")
        
        print("=" * 60)


if __name__ == "__main__":
    main()
