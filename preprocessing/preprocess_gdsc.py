"""
Preprocess GDSC data into PyTorch Geometric format for GNN training.

This script creates train/val/test data from:
1. Cell_line_RMA_proc_basalExp.txt - Gene expression features
2. Cell_Lines_Details.xlsx - Cell line name to COSMIC ID mapping
3. screened_compounds_rel_8.5.csv - Drug annotations
4. GDSC1/GDSC2 IC50 Excel files - Drug response data

Output (in ../data/processed/):
- gdsc_processed_train.pt, gdsc_processed_val.pt, gdsc_processed_test.pt
- gdsc_processed_mappings.pt - Cell line and drug name to index mappings

Usage:
    python preprocess_gdsc.py
    python preprocess_gdsc.py --output-prefix new_data
"""

import argparse
import random
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData

# Resolve paths relative to project root
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess GDSC data for GNN training")
    p.add_argument("--expression", default=str(RAW_DATA_DIR / "Cell_line_RMA_proc_basalExp.txt"))
    p.add_argument("--cell-info", default=str(RAW_DATA_DIR / "Cell_Lines_Details.xlsx"))
    p.add_argument("--compounds", default=str(RAW_DATA_DIR / "screened_compounds_rel_8.5.csv"))
    p.add_argument("--gdsc1", default=str(RAW_DATA_DIR / "GDSC1_fitted_dose_response_27Oct23.xlsx"))
    p.add_argument("--gdsc2", default=str(RAW_DATA_DIR / "GDSC2_fitted_dose_response_27Oct23.xlsx"))
    p.add_argument("--output-prefix", default="gdsc_processed")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_expression_data(path: str):
    """Load gene expression data and return cell line features."""
    print(f"Loading expression data from {path}...")
    
    # Read the expression file (genes as rows, cell lines as columns)
    df = pd.read_csv(path, sep='\t', index_col=0)
    # Remove GENE_title column if present
    if 'GENE_title' in df.columns:
        df = df.drop(columns=['GENE_title'])
    # Normalize gene expression features (z-score)
    df = (df - df.mean()) / df.std()


    # LASSO feature selection: select informative genes
    from sklearn.linear_model import LassoCV
    X = df.T.values  # cell lines x genes
    y = np.random.rand(X.shape[0])  # Dummy target (replace with real IC50 if available)
    lasso = LassoCV(cv=3, random_state=42, max_iter=1000)
    lasso.fit(X, y)
    coef = lasso.coef_
    selected_genes = np.where(np.abs(coef) > 1e-4)[0]
    if len(selected_genes) == 0:
        print("[WARNING] LASSO selected 0 features. Falling back to all features.")
        df_selected = df
    else:
        print(f"[INFO] LASSO selected {len(selected_genes)} features.")
        df_selected = df.iloc[selected_genes, :]

    # Reduce dimensionality with PCA (retain up to 100 components, but not more than available features)
    from sklearn.decomposition import PCA
    n_selected = df_selected.shape[0]
    n_components = min(100, n_selected) if n_selected > 0 else 1
    pca = PCA(n_components=n_components, random_state=42)
    df_pca = pd.DataFrame(pca.fit_transform(df_selected.T), index=df.columns)
    return df_pca, df_pca.shape[1]
    
    # Column names are like "DATA.906826" or "DATA.1503362.1" => extract COSMIC IDs
    cosmic_to_features = {}
    for col in df.columns:
        if col.startswith('DATA.'):
            # Handle IDs like "DATA.1503362.1" - take first number
            id_str = col.replace('DATA.', '').split('.')[0]
            try:
                cosmic_id = int(id_str)
                # Transpose: features for this cell line (all genes)
                features = df[col].values.astype(np.float32)
                cosmic_to_features[cosmic_id] = features
            except ValueError:
                print(f"  Warning: Could not parse column {col}")
    
    num_genes = len(df)
    print(f"  Loaded {len(cosmic_to_features)} cell lines with {num_genes} genes each")
    
    return cosmic_to_features, num_genes


def load_cell_line_info(path: str):
    """Load cell line name to COSMIC ID mapping."""
    print(f"Loading cell line info from {path}...")
    df = pd.read_excel(path)
    
    # Create mapping: cell line name -> COSMIC ID
    name_to_cosmic = {}
    for _, row in df.iterrows():
        name = str(row['Sample Name']).strip()
        cosmic_id = row['COSMIC identifier']
        if pd.notna(cosmic_id):
            name_to_cosmic[name] = int(cosmic_id)
    
    print(f"  Loaded {len(name_to_cosmic)} cell line mappings")
    return name_to_cosmic


def load_compounds(path: str):
    """Load drug information."""
    print(f"Loading compounds from {path}...")
    df = pd.read_csv(path)
    
    # Create mappings
    drug_name_to_id = {}
    drug_id_to_info = {}
    
    # Get unique target pathways for one-hot encoding
    pathways = df['TARGET_PATHWAY'].dropna().unique().tolist()
    pathway_to_idx = {p: i for i, p in enumerate(pathways)}
    
    for _, row in df.iterrows():
        drug_id = int(row['DRUG_ID'])
        drug_name = str(row['DRUG_NAME']).strip()
        target_pathway = row['TARGET_PATHWAY'] if pd.notna(row['TARGET_PATHWAY']) else 'Unknown'
        target = row['TARGET'] if pd.notna(row['TARGET']) else ''
        
        drug_name_to_id[drug_name] = drug_id
        drug_id_to_info[drug_id] = {
            'name': drug_name,
            'target': target,
            'pathway': target_pathway
        }
    
    print(f"  Loaded {len(drug_name_to_id)} drugs with {len(pathways)} unique pathways")
    return drug_name_to_id, drug_id_to_info, pathways


def load_ic50_data(gdsc1_path: str, gdsc2_path: str):
    """Load IC50 data from GDSC1 and GDSC2."""
    print(f"Loading IC50 data...")
    
    dfs = []
    for path, dataset in [(gdsc1_path, 'GDSC1'), (gdsc2_path, 'GDSC2')]:
        print(f"  Loading {dataset}...")
        df = pd.read_excel(path)
        df['SOURCE'] = dataset
        dfs.append(df)
        print(f"    {len(df)} records")
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Total: {len(combined)} IC50 records")
    return combined


def create_drug_features(drug_id_to_info: dict, pathways: list, num_drugs: int):
    """Create drug feature matrix using pathway and one-hot encoding."""
    # Feature dimensions:
    # - One-hot for pathway (len(pathways) + 1 for unknown)
    # - Could add more features if available
    
    num_pathways = len(pathways) + 1  # +1 for unknown
    pathway_to_idx = {p: i for i, p in enumerate(pathways)}
    pathway_to_idx['Unknown'] = len(pathways)
    
    features = np.zeros((num_drugs, num_pathways), dtype=np.float32)
    
    drug_ids = sorted(drug_id_to_info.keys())
    drug_id_to_idx = {did: idx for idx, did in enumerate(drug_ids)}
    
    for drug_id, info in drug_id_to_info.items():
        idx = drug_id_to_idx[drug_id]
        pathway = info['pathway']
        pathway_idx = pathway_to_idx.get(pathway, pathway_to_idx['Unknown'])
        features[idx, pathway_idx] = 1.0
    
    return features, drug_id_to_idx


def build_graph_data(
    cell_features: np.ndarray,
    drug_features: np.ndarray,
    edges: list,  # List of (cell_idx, drug_idx, ln_ic50)
    include_reverse: bool = True
):
    """Build PyTorch Geometric HeteroData object."""
    data = HeteroData()
    
    # Node features
    data['cell_line'].x = torch.tensor(cell_features, dtype=torch.float32)
    data['drug'].x = torch.tensor(drug_features, dtype=torch.float32)
    
    if len(edges) > 0:
        # Convert edges to tensors
        cell_indices = torch.tensor([e[0] for e in edges], dtype=torch.long)
        drug_indices = torch.tensor([e[1] for e in edges], dtype=torch.long)
        ln_ic50_values = torch.tensor([e[2] for e in edges], dtype=torch.float32)
        
        # Convert from LN_IC50 (natural log) to IC50 for compatibility with train.py
        # train.py expects raw IC50 and does log10 transform internally
        ic50_values = torch.exp(ln_ic50_values)  # e^(ln_ic50) = ic50
        
        # Forward edges: cell_line -> drug
        # Use pos_edge_label_index and pos_edge_label to match train.py expectations
        data['cell_line', 'treated_with', 'drug'].pos_edge_label_index = torch.stack([cell_indices, drug_indices])
        data['cell_line', 'treated_with', 'drug'].pos_edge_label = ic50_values
        data['cell_line', 'treated_with', 'drug'].edge_index = torch.stack([cell_indices, drug_indices])
        
        # Reverse edges for message passing
        if include_reverse:
            data['drug', 'rev_treated_with', 'cell_line'].edge_index = torch.stack([drug_indices, cell_indices])
    
    return data


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Load all data
    cosmic_to_features, num_genes = load_expression_data(args.expression)
    name_to_cosmic = load_cell_line_info(args.cell_info)
    drug_name_to_id, drug_id_to_info, pathways = load_compounds(args.compounds)
    ic50_df = load_ic50_data(args.gdsc1, args.gdsc2)
    
    print("\n" + "="*60)
    print("MATCHING DATA")
    print("="*60)
    
    # Find cell lines that have both expression data and IC50 data
    valid_cells = {}  # name -> (cosmic_id, features)
    ic50_cell_names = set(ic50_df['CELL_LINE_NAME'].unique())
    
    matched_cells = 0
    for name in ic50_cell_names:
        name_clean = str(name).strip()
        if name_clean in name_to_cosmic:
            cosmic_id = name_to_cosmic[name_clean]
            if cosmic_id in cosmic_to_features:
                valid_cells[name_clean] = (cosmic_id, cosmic_to_features[cosmic_id])
                matched_cells += 1
    
    print(f"Cell lines with IC50 data: {len(ic50_cell_names)}")
    print(f"Cell lines matched to expression: {matched_cells}")
    
    # Find drugs that exist in both IC50 data and compound annotations
    ic50_drug_names = set(ic50_df['DRUG_NAME'].unique())
    valid_drugs = {}  # name -> drug_id
    
    for name in ic50_drug_names:
        name_clean = str(name).strip()
        if name_clean in drug_name_to_id:
            valid_drugs[name_clean] = drug_name_to_id[name_clean]
    
    print(f"Drugs with IC50 data: {len(ic50_drug_names)}")
    print(f"Drugs matched to annotations: {len(valid_drugs)}")
    
    # Create index mappings
    cell_name_to_idx = {name: idx for idx, name in enumerate(sorted(valid_cells.keys()))}
    drug_name_to_idx = {name: idx for idx, name in enumerate(sorted(valid_drugs.keys()))}
    
    num_cells = len(cell_name_to_idx)
    num_drugs = len(drug_name_to_idx)
    
    print(f"\nFinal dataset: {num_cells} cell lines, {num_drugs} drugs")
    
    # Create feature matrices
    print("\nCreating feature matrices...")
    
    # Cell line features (gene expression)
    cell_features = np.zeros((num_cells, num_genes), dtype=np.float32)
    for name, idx in cell_name_to_idx.items():
        _, features = valid_cells[name]
        cell_features[idx] = features
    
    # Drug features (pathway one-hot encoding)
    drug_features = np.zeros((num_drugs, len(pathways) + 1), dtype=np.float32)
    pathway_to_idx = {p: i for i, p in enumerate(pathways)}
    pathway_to_idx['Unknown'] = len(pathways)
    
    for name, idx in drug_name_to_idx.items():
        drug_id = valid_drugs[name]
        if drug_id in drug_id_to_info:
            pathway = drug_id_to_info[drug_id]['pathway']
            pathway_idx = pathway_to_idx.get(pathway, pathway_to_idx['Unknown'])
            drug_features[idx, pathway_idx] = 1.0
        else:
            drug_features[idx, pathway_to_idx['Unknown']] = 1.0
    
    print(f"Cell features shape: {cell_features.shape}")
    print(f"Drug features shape: {drug_features.shape}")
    
    # Build edges (IC50 interactions)
    print("\nBuilding edges from IC50 data...")
    edges = []  # (cell_idx, drug_idx, ln_ic50)
    skipped = 0
    
    for _, row in ic50_df.iterrows():
        cell_name = str(row['CELL_LINE_NAME']).strip()
        drug_name = str(row['DRUG_NAME']).strip()
        ln_ic50 = row['LN_IC50']
        
        if cell_name in cell_name_to_idx and drug_name in drug_name_to_idx:
            if pd.notna(ln_ic50):
                cell_idx = cell_name_to_idx[cell_name]
                drug_idx = drug_name_to_idx[drug_name]
                edges.append((cell_idx, drug_idx, float(ln_ic50)))
        else:
            skipped += 1
    
    print(f"Total edges (IC50 pairs): {len(edges)}")
    print(f"Skipped (missing cell/drug): {skipped}")
    
    # Remove duplicate edges (keep first occurrence)
    seen = set()
    unique_edges = []
    for e in edges:
        key = (e[0], e[1])
        if key not in seen:
            seen.add(key)
            unique_edges.append(e)
    
    print(f"Unique edges: {len(unique_edges)}")
    
    # Split into train/val/test
    print("\nSplitting data...")
    random.shuffle(unique_edges)
    
    n_total = len(unique_edges)
    n_train = int(n_total * args.train_ratio)
    n_val = int(n_total * args.val_ratio)
    
    train_edges = unique_edges[:n_train]
    val_edges = unique_edges[n_train:n_train + n_val]
    test_edges = unique_edges[n_train + n_val:]
    
    print(f"Train: {len(train_edges)} edges")
    print(f"Val: {len(val_edges)} edges")
    print(f"Test: {len(test_edges)} edges")
    
    # Build HeteroData objects
    print("\nBuilding graph data...")
    train_data = build_graph_data(cell_features, drug_features, train_edges)
    val_data = build_graph_data(cell_features, drug_features, val_edges)
    test_data = build_graph_data(cell_features, drug_features, test_edges)
    
    # Save data
    print("\nSaving data...")
    output_dir = PROCESSED_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(train_data, output_dir / f"{args.output_prefix}_train.pt")
    torch.save(val_data, output_dir / f"{args.output_prefix}_val.pt")
    torch.save(test_data, output_dir / f"{args.output_prefix}_test.pt")
    
    # Save mappings and metadata
    mappings = {
        'cell_name_to_idx': cell_name_to_idx,
        'drug_name_to_idx': drug_name_to_idx,
        'idx_to_cell_name': {v: k for k, v in cell_name_to_idx.items()},
        'idx_to_drug_name': {v: k for k, v in drug_name_to_idx.items()},
        'pathways': pathways,
        'num_genes': num_genes,
        'cell_feature_dim': num_genes,
        'drug_feature_dim': len(pathways) + 1,
        'num_cell_lines': num_cells,
        'num_drugs': num_drugs,
    }
    torch.save(mappings, output_dir / f"{args.output_prefix}_mappings.pt")
    
    print(f"\nSaved to:")
    print(f"  {args.output_prefix}_train.pt")
    print(f"  {args.output_prefix}_val.pt")
    print(f"  {args.output_prefix}_test.pt")
    print(f"  {args.output_prefix}_mappings.pt")
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"Cell lines: {num_cells}")
    print(f"Drugs: {num_drugs}")
    print(f"Gene features: {num_genes}")
    print(f"Drug features: {len(pathways) + 1} (pathway one-hot)")
    print(f"Total interactions: {len(unique_edges)}")
    print("="*60)


if __name__ == "__main__":
    main()
