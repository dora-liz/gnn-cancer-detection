#!/usr/bin/env python3
"""
Generate sample cell line input files for the drug ranking prediction system.

This script creates example input files in various formats to demonstrate
the expected format for the predict_drugs.py script.

Usage:
    python generate_sample_input.py --output sample_cell.json
    python generate_sample_input.py --output sample_cell.pt
    python generate_sample_input.py --output sample_cell.csv

The generated files will contain realistic-looking gene expression values
that can be used to test the drug ranking system.
"""

import argparse
import json
import sys
from pathlib import Path

# Try to import torch for .pt file generation
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate sample cell line input files for drug ranking",
    )
    parser.add_argument(
        "--output", "-o",
        default="sample_cell_input.json",
        help="Output file path (.json, .csv, .txt, or .pt)"
    )
    parser.add_argument(
        "--num-features", "-n",
        type=int,
        default=19215,
        help="Number of gene expression features (default: 19215)"
    )
    parser.add_argument(
        "--from-existing",
        default="",
        help="Extract features from an existing .pt file (e.g., train_data.pt)"
    )
    parser.add_argument(
        "--cell-index",
        type=int,
        default=0,
        help="Index of cell line to extract when using --from-existing"
    )
    return parser.parse_args()


def generate_random_features(num_features: int) -> list:
    """Generate random gene expression-like values."""
    import random
    random.seed(42)
    # Gene expression values are typically log-normalized, ranging roughly -3 to 3
    return [random.gauss(0, 1) for _ in range(num_features)]


def extract_features_from_graph(path: str, cell_index: int) -> list:
    """Extract cell line features from an existing graph file."""
    if not HAS_TORCH:
        print("ERROR: PyTorch is required to extract from .pt files")
        sys.exit(1)
    
    data = torch.load(path, map_location="cpu", weights_only=False)
    
    if hasattr(data, "x_dict") or isinstance(data, dict):
        # HeteroData object
        if hasattr(data, "__getitem__"):
            cell_features = data["cell_line"].x
        else:
            cell_features = data.get("cell_line", {}).get("x")
    else:
        print("ERROR: Unsupported file format")
        sys.exit(1)
    
    if cell_index >= cell_features.shape[0]:
        print(f"ERROR: Cell index {cell_index} out of range (max: {cell_features.shape[0]-1})")
        sys.exit(1)
    
    return cell_features[cell_index].tolist()


def save_json(features: list, path: str):
    """Save features as JSON array."""
    Path(path).write_text(json.dumps(features, indent=None), encoding="utf-8")


def save_csv(features: list, path: str):
    """Save features as CSV (single row)."""
    csv_content = ",".join(f"{v:.6f}" for v in features)
    Path(path).write_text(csv_content, encoding="utf-8")


def save_pt(features: list, path: str):
    """Save features as PyTorch tensor."""
    if not HAS_TORCH:
        print("ERROR: PyTorch is required to save .pt files")
        sys.exit(1)
    
    tensor = torch.tensor([features], dtype=torch.float32)
    torch.save(tensor, path)


def main():
    args = parse_args()
    
    # Generate or extract features
    if args.from_existing:
        print(f"Extracting features from: {args.from_existing} (cell index: {args.cell_index})")
        features = extract_features_from_graph(args.from_existing, args.cell_index)
    else:
        print(f"Generating random features ({args.num_features} values)...")
        features = generate_random_features(args.num_features)
    
    # Determine output format and save
    output_path = Path(args.output)
    suffix = output_path.suffix.lower()
    
    if suffix == ".json":
        save_json(features, args.output)
    elif suffix in [".csv", ".txt"]:
        save_csv(features, args.output)
    elif suffix == ".pt":
        save_pt(features, args.output)
    else:
        print(f"ERROR: Unsupported output format: {suffix}")
        print("Supported: .json, .csv, .txt, .pt")
        sys.exit(1)
    
    print(f"Saved {len(features)} features to: {args.output}")
    print(f"\nTo use with drug ranking:")
    print(f"  python predict_drugs.py --input {args.output}")


if __name__ == "__main__":
    main()
