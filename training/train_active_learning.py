"""
Active Learning Training for Drug-Cell Line GNN.

This script implements active learning strategies to efficiently train the GNN
model by selecting the most informative drug-cell line pairs to label.

Strategies implemented:
1. Uncertainty Sampling (MC Dropout)
2. Query-by-Committee (ensemble disagreement)
3. Diversity Sampling (coverage-based)
4. Hybrid (uncertainty + diversity)

Usage:
    python train_active_learning.py --strategy uncertainty --budget 5000 --batch-size 100
    python train_active_learning.py --strategy hybrid --budget 10000 --al-rounds 20

The script simulates active learning by:
1. Starting with a small labeled subset
2. Using the oracle (held-out data) to provide labels for selected pairs
3. Iteratively improving the model with newly labeled data
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch_geometric.data import HeteroData
from torch_geometric.nn import to_hetero
from sklearn.cluster import KMeans

from gnn_model import GNN, LinkPredictor, FeatureProjector

EDGE_TYPE = ("cell_line", "treated_with", "drug")
REV_EDGE_TYPE = ("drug", "rev_treated_with", "cell_line")

# Resolve paths relative to project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Active Learning for GNN Drug Ranking")
    
    # Data paths
    p.add_argument("--train-data", default=str(DATA_DIR / "gdsc_processed_train.pt"))
    p.add_argument("--val-data", default=str(DATA_DIR / "gdsc_processed_val.pt"))
    p.add_argument("--test-data", default=str(DATA_DIR / "gdsc_processed_test.pt"))
    
    # Active learning settings
    p.add_argument("--strategy", choices=["uncertainty", "qbc", "diversity", "hybrid", "random"],
                   default="hybrid", help="Query strategy")
    p.add_argument("--budget", type=int, default=5000, 
                   help="Total labeling budget (number of experiments)")
    p.add_argument("--batch-size", type=int, default=100,
                   help="Number of samples to query per round")
    p.add_argument("--initial-size", type=int, default=1000,
                   help="Initial labeled pool size")
    p.add_argument("--mc-samples", type=int, default=20,
                   help="Number of MC Dropout samples for uncertainty")
    p.add_argument("--n-committees", type=int, default=3,
                   help="Number of models for QBC")
    p.add_argument("--diversity-weight", type=float, default=0.3,
                   help="Weight for diversity in hybrid strategy")
    
    # Model architecture
    p.add_argument("--hidden-channels", type=int, default=128)
    p.add_argument("--out-channels", type=int, default=32)
    p.add_argument("--decoder-hidden", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.3,
                   help="Dropout rate (higher for better uncertainty)")
    p.add_argument("--project-dim", type=int, default=256)
    p.add_argument("--num-gnn-layers", type=int, default=3)
    p.add_argument("--num-decoder-layers", type=int, default=4)
    
    # Training settings
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--epochs-per-round", type=int, default=30,
                   help="Training epochs per AL round")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)
    
    # Output
    p.add_argument("--output-prefix", default="active_learning")
    p.add_argument("--save-history", action="store_true", default=True,
                   help="Save learning curve history")
    
    return p.parse_args()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(arg: str) -> torch.device:
    if arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_data(path: str, device: torch.device):
    if not Path(path).exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location=device, weights_only=False)


def get_all_edges(data: HeteroData) -> List[Tuple[int, int, float]]:
    """Extract all edges with their labels from the dataset."""
    store = data[EDGE_TYPE]
    edge_index = store.pos_edge_label_index
    labels = store.pos_edge_label
    
    edges = []
    for i in range(edge_index.shape[1]):
        cell_idx = edge_index[0, i].item()
        drug_idx = edge_index[1, i].item()
        ic50 = labels[i].item()
        edges.append((cell_idx, drug_idx, ic50))
    
    return edges


def compute_metrics(preds: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """Compute evaluation metrics."""
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    # Log-space MSE
    log_mse = float(((preds - labels) ** 2).mean().item())
    
    # RMSE in IC50 units
    ic50_pred = torch.pow(10.0, preds)
    ic50_true = torch.pow(10.0, labels)
    ic50_rmse = float(torch.sqrt(((ic50_pred - ic50_true) ** 2).mean()).item())
    
    # Spearman correlation
    if len(preds_np) > 1:
        spear, _ = spearmanr(preds_np, labels_np)
        spear = float(spear) if not math.isnan(spear) else 0.0
    else:
        spear = 0.0
    
    return {"log_mse": log_mse, "ic50_rmse": ic50_rmse, "spearman": spear}


def top_k_precision(preds: torch.Tensor, labels: torch.Tensor, k: int = 10) -> float:
    """Precision@k: fraction of top-k predicted in true top-k."""
    if preds.shape[0] < k:
        k = preds.shape[0]
    pred_topk = set(torch.argsort(preds)[:k].tolist())
    true_topk = set(torch.argsort(labels)[:k].tolist())
    return len(pred_topk & true_topk) / k


# ---------------------------------------------------------------------------
# Active Learning Data Manager
# ---------------------------------------------------------------------------


class ActiveLearningDataManager:
    """
    Manages labeled and unlabeled pools for active learning.
    Uses the full dataset as an oracle to provide labels on demand.
    """
    
    def __init__(self, full_data: HeteroData, initial_size: int, seed: int = 42):
        self.full_data = full_data
        self.device = full_data["cell_line"].x.device
        
        # Extract all edges as oracle knowledge
        self.all_edges = get_all_edges(full_data)
        self.edge_to_label = {(c, d): ic50 for c, d, ic50 in self.all_edges}
        
        # Create initial split
        random.seed(seed)
        all_indices = list(range(len(self.all_edges)))
        random.shuffle(all_indices)
        
        self.labeled_indices: Set[int] = set(all_indices[:initial_size])
        self.unlabeled_indices: Set[int] = set(all_indices[initial_size:])
        
        # Store node features (constant throughout)
        self.cell_features = full_data["cell_line"].x.clone()
        self.drug_features = full_data["drug"].x.clone()
        self.num_cells = self.cell_features.shape[0]
        self.num_drugs = self.drug_features.shape[0]
        
        print(f"ActiveLearningDataManager initialized:")
        print(f"  Total edges: {len(self.all_edges)}")
        print(f"  Initial labeled: {len(self.labeled_indices)}")
        print(f"  Unlabeled pool: {len(self.unlabeled_indices)}")
    
    def get_labeled_data(self) -> HeteroData:
        """Build HeteroData from current labeled set."""
        labeled_edges = [self.all_edges[i] for i in self.labeled_indices]
        
        cell_idx = torch.tensor([e[0] for e in labeled_edges], dtype=torch.long)
        drug_idx = torch.tensor([e[1] for e in labeled_edges], dtype=torch.long)
        labels = torch.tensor([e[2] for e in labeled_edges], dtype=torch.float32)
        
        data = HeteroData()
        data["cell_line"].x = self.cell_features.clone()
        data["drug"].x = self.drug_features.clone()
        
        edge_index = torch.stack([cell_idx, drug_idx], dim=0)
        data[EDGE_TYPE].edge_index = edge_index
        data[EDGE_TYPE].pos_edge_label_index = edge_index
        data[EDGE_TYPE].pos_edge_label = labels
        
        # Reverse edges
        rev_edge_index = torch.stack([drug_idx, cell_idx], dim=0)
        data[REV_EDGE_TYPE].edge_index = rev_edge_index
        
        return data.to(self.device)
    
    def get_unlabeled_pairs(self) -> List[Tuple[int, int]]:
        """Get list of unlabeled (cell, drug) pairs."""
        return [(self.all_edges[i][0], self.all_edges[i][1]) 
                for i in self.unlabeled_indices]
    
    def get_unlabeled_indices(self) -> List[int]:
        """Get list of unlabeled edge indices."""
        return list(self.unlabeled_indices)
    
    def query_oracle(self, indices: List[int]) -> List[Tuple[int, int, float]]:
        """
        Query the oracle for labels of selected indices.
        In simulation, we just look up the ground truth.
        In practice, this would trigger real experiments.
        """
        results = []
        for idx in indices:
            if idx in self.unlabeled_indices:
                self.unlabeled_indices.remove(idx)
                self.labeled_indices.add(idx)
                results.append(self.all_edges[idx])
        return results
    
    def get_edge_info(self, idx: int) -> Tuple[int, int, float]:
        """Get (cell_idx, drug_idx, ic50) for an edge index."""
        return self.all_edges[idx]


# ---------------------------------------------------------------------------
# Model Creation and Training
# ---------------------------------------------------------------------------


def create_models(data: HeteroData, args, device: torch.device):
    """Create fresh model instances."""
    dims_dict = {
        "cell_line": int(data["cell_line"].x.shape[1]),
        "drug": int(data["drug"].x.shape[1]),
    }
    
    projector = FeatureProjector(
        dims_dict=dims_dict,
        project_dim=args.project_dim,
        dropout=args.dropout,
    ).to(device)
    
    encoder = GNN(
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
        dropout=args.dropout,
        num_layers=args.num_gnn_layers,
    )
    model = to_hetero(encoder, data.metadata(), aggr="sum").to(device)
    
    decoder = LinkPredictor(
        in_channels=args.out_channels * 2,
        hidden_channels=args.decoder_hidden,
        out_channels=1,
        dropout=args.dropout,
        num_layers=args.num_decoder_layers,
    ).to(device)
    
    # Initialize lazy modules
    with torch.no_grad():
        proj_x = projector(data.x_dict)
        _ = model(proj_x, data.edge_index_dict)
    
    return projector, model, decoder


def train_model(
    projector, model, decoder, 
    train_data: HeteroData, 
    val_data: HeteroData,
    args,
    device: torch.device,
    verbose: bool = False
) -> Dict[str, float]:
    """Train model for a fixed number of epochs."""
    
    all_params = list(projector.parameters()) + list(model.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    
    best_val_loss = float("inf")
    patience_ctr = 0
    best_state = None
    
    for epoch in range(1, args.epochs_per_round + 1):
        # Training step
        projector.train()
        model.train()
        decoder.train()
        
        optimizer.zero_grad()
        
        store = train_data[EDGE_TYPE]
        edge_idx = store.pos_edge_label_index
        labels = torch.log10(store.pos_edge_label.clamp(min=1e-6))
        
        proj_x = projector(train_data.x_dict)
        emb = model(proj_x, train_data.edge_index_dict)
        
        cell_emb = emb["cell_line"][edge_idx[0]]
        drug_emb = emb["drug"][edge_idx[1]]
        preds = decoder(cell_emb, drug_emb)
        
        loss = loss_fn(preds, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()
        
        # Validation
        projector.eval()
        model.eval()
        decoder.eval()
        
        with torch.no_grad():
            val_store = val_data[EDGE_TYPE]
            val_edge_idx = val_store.pos_edge_label_index
            val_labels = torch.log10(val_store.pos_edge_label.clamp(min=1e-6))
            
            val_proj_x = projector(val_data.x_dict)
            val_emb = model(val_proj_x, val_data.edge_index_dict)
            
            val_cell_emb = val_emb["cell_line"][val_edge_idx[0]]
            val_drug_emb = val_emb["drug"][val_edge_idx[1]]
            val_preds = decoder(val_cell_emb, val_drug_emb)
            
            val_loss = loss_fn(val_preds, val_labels).item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr = 0
            best_state = {
                "projector": {k: v.cpu().clone() for k, v in projector.state_dict().items()},
                "model": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "decoder": {k: v.cpu().clone() for k, v in decoder.state_dict().items()},
            }
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                break
    
    # Restore best state
    if best_state:
        projector.load_state_dict(best_state["projector"])
        model.load_state_dict(best_state["model"])
        decoder.load_state_dict(best_state["decoder"])
    
    # Final evaluation
    projector.eval()
    model.eval()
    decoder.eval()
    
    with torch.no_grad():
        val_store = val_data[EDGE_TYPE]
        val_edge_idx = val_store.pos_edge_label_index
        val_labels = torch.log10(val_store.pos_edge_label.clamp(min=1e-6))
        
        val_proj_x = projector(val_data.x_dict)
        val_emb = model(val_proj_x, val_data.edge_index_dict)
        
        val_cell_emb = val_emb["cell_line"][val_edge_idx[0]]
        val_drug_emb = val_emb["drug"][val_edge_idx[1]]
        val_preds = decoder(val_cell_emb, val_drug_emb)
    
    metrics = compute_metrics(val_preds, val_labels)
    return metrics


# ---------------------------------------------------------------------------
# Uncertainty Estimation (MC Dropout)
# ---------------------------------------------------------------------------


def estimate_uncertainty_mc_dropout(
    projector, model, decoder,
    data: HeteroData,
    unlabeled_indices: List[int],
    data_manager: ActiveLearningDataManager,
    n_samples: int = 20,
    batch_size: int = 1000
) -> Dict[int, float]:
    """
    Estimate prediction uncertainty using MC Dropout.
    Returns dict mapping edge index to uncertainty (std of predictions).
    """
    # Enable dropout during inference
    projector.train()
    model.train()
    decoder.train()
    
    # Get embeddings for all nodes (these don't change with dropout on decoder)
    all_predictions = defaultdict(list)
    
    for _ in range(n_samples):
        with torch.no_grad():
            proj_x = projector(data.x_dict)
            emb = model(proj_x, data.edge_index_dict)
        
        # Predict for unlabeled pairs in batches
        for batch_start in range(0, len(unlabeled_indices), batch_size):
            batch_indices = unlabeled_indices[batch_start:batch_start + batch_size]
            
            cell_indices = []
            drug_indices = []
            for idx in batch_indices:
                c, d, _ = data_manager.get_edge_info(idx)
                cell_indices.append(c)
                drug_indices.append(d)
            
            cell_idx = torch.tensor(cell_indices, dtype=torch.long, device=data["cell_line"].x.device)
            drug_idx = torch.tensor(drug_indices, dtype=torch.long, device=data["cell_line"].x.device)
            
            cell_emb = emb["cell_line"][cell_idx]
            drug_emb = emb["drug"][drug_idx]
            
            # This uses dropout (model is in train mode)
            with torch.no_grad():
                preds = decoder(cell_emb, drug_emb)
            
            for i, idx in enumerate(batch_indices):
                all_predictions[idx].append(preds[i].item())
    
    # Compute uncertainty as standard deviation
    uncertainties = {}
    for idx, preds in all_predictions.items():
        uncertainties[idx] = np.std(preds)
    
    return uncertainties


# ---------------------------------------------------------------------------
# Query Strategies
# ---------------------------------------------------------------------------


def query_random(
    unlabeled_indices: List[int],
    k: int,
    **kwargs
) -> List[int]:
    """Random sampling baseline."""
    return random.sample(unlabeled_indices, min(k, len(unlabeled_indices)))


def query_uncertainty(
    unlabeled_indices: List[int],
    k: int,
    uncertainties: Dict[int, float],
    **kwargs
) -> List[int]:
    """Select samples with highest uncertainty."""
    # Sort by uncertainty (descending)
    sorted_indices = sorted(unlabeled_indices, key=lambda x: uncertainties.get(x, 0), reverse=True)
    return sorted_indices[:k]


def query_diversity(
    unlabeled_indices: List[int],
    k: int,
    data_manager: ActiveLearningDataManager,
    n_clusters: int = None,
    **kwargs
) -> List[int]:
    """Select diverse samples using clustering on cell features."""
    if n_clusters is None:
        n_clusters = min(k, 50)
    
    # Get cell features for unlabeled pairs
    cell_indices = []
    for idx in unlabeled_indices:
        c, d, _ = data_manager.get_edge_info(idx)
        cell_indices.append(c)
    
    # Get unique cells and their features
    unique_cells = list(set(cell_indices))
    if len(unique_cells) < n_clusters:
        n_clusters = len(unique_cells)
    
    cell_features = data_manager.cell_features[unique_cells].cpu().numpy()
    
    # Cluster cells
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cell_clusters = kmeans.fit_predict(cell_features)
    cell_to_cluster = {c: cell_clusters[i] for i, c in enumerate(unique_cells)}
    
    # Group unlabeled indices by cluster
    cluster_to_indices = defaultdict(list)
    for idx in unlabeled_indices:
        c, d, _ = data_manager.get_edge_info(idx)
        cluster = cell_to_cluster.get(c, 0)
        cluster_to_indices[cluster].append(idx)
    
    # Round-robin selection from clusters
    selected = []
    cluster_iters = {c: iter(indices) for c, indices in cluster_to_indices.items()}
    
    while len(selected) < k:
        any_added = False
        for cluster in cluster_to_indices:
            if len(selected) >= k:
                break
            try:
                idx = next(cluster_iters[cluster])
                selected.append(idx)
                any_added = True
            except StopIteration:
                continue
        if not any_added:
            break
    
    return selected


def query_hybrid(
    unlabeled_indices: List[int],
    k: int,
    uncertainties: Dict[int, float],
    data_manager: ActiveLearningDataManager,
    diversity_weight: float = 0.3,
    **kwargs
) -> List[int]:
    """
    Hybrid strategy: combine uncertainty and diversity.
    Select uncertain samples while ensuring diversity.
    """
    # Get top uncertain samples (2x needed)
    uncertain_pool = query_uncertainty(unlabeled_indices, k * 2, uncertainties)
    
    if len(uncertain_pool) <= k:
        return uncertain_pool
    
    # Apply diversity selection to the uncertain pool
    return query_diversity(uncertain_pool, k, data_manager)


def query_qbc(
    unlabeled_indices: List[int],
    k: int,
    committee_predictions: List[Dict[int, float]],
    **kwargs
) -> List[int]:
    """
    Query-by-Committee: select samples where models disagree most.
    committee_predictions: list of dicts mapping idx -> prediction for each model
    """
    disagreements = {}
    
    for idx in unlabeled_indices:
        preds = [cp.get(idx, 0) for cp in committee_predictions]
        disagreements[idx] = np.var(preds)
    
    sorted_indices = sorted(unlabeled_indices, key=lambda x: disagreements.get(x, 0), reverse=True)
    return sorted_indices[:k]


# ---------------------------------------------------------------------------
# Query-by-Committee Training
# ---------------------------------------------------------------------------


def train_committee(
    data_manager: ActiveLearningDataManager,
    val_data: HeteroData,
    args,
    device: torch.device,
    n_models: int = 3
) -> List[Tuple]:
    """Train multiple models with different initializations for QBC."""
    train_data = data_manager.get_labeled_data()
    committee = []
    
    for i in range(n_models):
        # Different seed for each model
        set_seed(args.seed + i * 100)
        
        projector, model, decoder = create_models(train_data, args, device)
        train_model(projector, model, decoder, train_data, val_data, args, device)
        
        committee.append((projector, model, decoder))
    
    return committee


def get_committee_predictions(
    committee: List[Tuple],
    data: HeteroData,
    unlabeled_indices: List[int],
    data_manager: ActiveLearningDataManager
) -> List[Dict[int, float]]:
    """Get predictions from each committee member."""
    predictions = []
    
    for projector, model, decoder in committee:
        projector.eval()
        model.eval()
        decoder.eval()
        
        with torch.no_grad():
            proj_x = projector(data.x_dict)
            emb = model(proj_x, data.edge_index_dict)
        
        model_preds = {}
        for idx in unlabeled_indices:
            c, d, _ = data_manager.get_edge_info(idx)
            cell_emb = emb["cell_line"][c:c+1]
            drug_emb = emb["drug"][d:d+1]
            
            with torch.no_grad():
                pred = decoder(cell_emb, drug_emb)
            model_preds[idx] = pred.item()
        
        predictions.append(model_preds)
    
    return predictions


# ---------------------------------------------------------------------------
# Main Active Learning Loop
# ---------------------------------------------------------------------------


def active_learning_loop(args):
    """Main active learning training loop."""
    
    print("=" * 60)
    print("ACTIVE LEARNING FOR GNN DRUG RANKING")
    print("=" * 60)
    print(f"Strategy: {args.strategy}")
    print(f"Budget: {args.budget}")
    print(f"Batch size: {args.batch_size}")
    print(f"Initial labeled: {args.initial_size}")
    print("=" * 60)
    
    set_seed(args.seed)
    device = pick_device(args.device)
    
    # Load data
    print("\nLoading data...")
    full_train_data = load_data(args.train_data, device)
    val_data = load_data(args.val_data, device)
    test_data = load_data(args.test_data, device)
    
    # Initialize data manager
    data_manager = ActiveLearningDataManager(
        full_train_data, 
        initial_size=args.initial_size,
        seed=args.seed
    )
    
    # History tracking
    history = {
        "round": [],
        "labeled_size": [],
        "val_spearman": [],
        "val_log_mse": [],
        "test_spearman": [],
        "test_log_mse": [],
    }
    
    # Calculate number of rounds
    remaining_budget = args.budget - args.initial_size
    n_rounds = max(1, remaining_budget // args.batch_size)
    
    print(f"\nWill run {n_rounds} active learning rounds")
    print("-" * 60)
    
    # Initial training
    print("\n[Round 0] Initial training...")
    train_data = data_manager.get_labeled_data()
    projector, model, decoder = create_models(train_data, args, device)
    val_metrics = train_model(projector, model, decoder, train_data, val_data, args, device)
    
    # Evaluate on test
    projector.eval()
    model.eval()
    decoder.eval()
    
    with torch.no_grad():
        test_store = test_data[EDGE_TYPE]
        test_edge_idx = test_store.pos_edge_label_index
        test_labels = torch.log10(test_store.pos_edge_label.clamp(min=1e-6))
        
        test_proj_x = projector(test_data.x_dict)
        test_emb = model(test_proj_x, test_data.edge_index_dict)
        
        test_cell_emb = test_emb["cell_line"][test_edge_idx[0]]
        test_drug_emb = test_emb["drug"][test_edge_idx[1]]
        test_preds = decoder(test_cell_emb, test_drug_emb)
    
    test_metrics = compute_metrics(test_preds, test_labels)
    
    history["round"].append(0)
    history["labeled_size"].append(len(data_manager.labeled_indices))
    history["val_spearman"].append(val_metrics["spearman"])
    history["val_log_mse"].append(val_metrics["log_mse"])
    history["test_spearman"].append(test_metrics["spearman"])
    history["test_log_mse"].append(test_metrics["log_mse"])
    
    print(f"  Labeled: {len(data_manager.labeled_indices)}")
    print(f"  Val Spearman: {val_metrics['spearman']:.4f}")
    print(f"  Test Spearman: {test_metrics['spearman']:.4f}")
    
    # Active learning rounds
    for round_num in range(1, n_rounds + 1):
        print(f"\n[Round {round_num}/{n_rounds}]")
        
        unlabeled_indices = data_manager.get_unlabeled_indices()
        if len(unlabeled_indices) == 0:
            print("  No more unlabeled samples!")
            break
        
        # Get current training data for uncertainty estimation
        train_data = data_manager.get_labeled_data()
        
        # Select samples based on strategy
        print(f"  Selecting {args.batch_size} samples ({args.strategy})...")
        
        if args.strategy == "random":
            selected = query_random(unlabeled_indices, args.batch_size)
            
        elif args.strategy == "uncertainty":
            uncertainties = estimate_uncertainty_mc_dropout(
                projector, model, decoder,
                train_data, unlabeled_indices, data_manager,
                n_samples=args.mc_samples
            )
            selected = query_uncertainty(unlabeled_indices, args.batch_size, uncertainties)
            
        elif args.strategy == "diversity":
            selected = query_diversity(unlabeled_indices, args.batch_size, data_manager)
            
        elif args.strategy == "hybrid":
            uncertainties = estimate_uncertainty_mc_dropout(
                projector, model, decoder,
                train_data, unlabeled_indices, data_manager,
                n_samples=args.mc_samples
            )
            selected = query_hybrid(
                unlabeled_indices, args.batch_size,
                uncertainties, data_manager,
                diversity_weight=args.diversity_weight
            )
            
        elif args.strategy == "qbc":
            committee = train_committee(data_manager, val_data, args, device, args.n_committees)
            committee_preds = get_committee_predictions(
                committee, train_data, unlabeled_indices, data_manager
            )
            selected = query_qbc(unlabeled_indices, args.batch_size, committee_preds)
        
        # Query oracle for labels
        new_labels = data_manager.query_oracle(selected)
        print(f"  Queried oracle: {len(new_labels)} new labels")
        
        # Retrain model with expanded training set
        print(f"  Retraining model...")
        train_data = data_manager.get_labeled_data()
        
        # Create fresh models for retraining
        projector, model, decoder = create_models(train_data, args, device)
        val_metrics = train_model(projector, model, decoder, train_data, val_data, args, device)
        
        # Evaluate on test
        projector.eval()
        model.eval()
        decoder.eval()
        
        with torch.no_grad():
            test_proj_x = projector(test_data.x_dict)
            test_emb = model(test_proj_x, test_data.edge_index_dict)
            
            test_cell_emb = test_emb["cell_line"][test_edge_idx[0]]
            test_drug_emb = test_emb["drug"][test_edge_idx[1]]
            test_preds = decoder(test_cell_emb, test_drug_emb)
        
        test_metrics = compute_metrics(test_preds, test_labels)
        
        # Record history
        history["round"].append(round_num)
        history["labeled_size"].append(len(data_manager.labeled_indices))
        history["val_spearman"].append(val_metrics["spearman"])
        history["val_log_mse"].append(val_metrics["log_mse"])
        history["test_spearman"].append(test_metrics["spearman"])
        history["test_log_mse"].append(test_metrics["log_mse"])
        
        print(f"  Labeled: {len(data_manager.labeled_indices)}")
        print(f"  Val Spearman: {val_metrics['spearman']:.4f}")
        print(f"  Test Spearman: {test_metrics['spearman']:.4f}")
    
    # Final results
    print("\n" + "=" * 60)
    print("ACTIVE LEARNING COMPLETE")
    print("=" * 60)
    print(f"Final labeled size: {len(data_manager.labeled_indices)}")
    print(f"Final Val Spearman: {val_metrics['spearman']:.4f}")
    print(f"Final Test Spearman: {test_metrics['spearman']:.4f}")
    
    # Save models
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(projector.state_dict(), MODELS_DIR / f"{args.output_prefix}_projector.pt")
    torch.save(model.state_dict(), MODELS_DIR / f"{args.output_prefix}_gnn.pt")
    torch.save(decoder.state_dict(), MODELS_DIR / f"{args.output_prefix}_decoder.pt")
    
    # Save history
    if args.save_history:
        history_path = MODELS_DIR / f"{args.output_prefix}_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"\nSaved history to {history_path}")
    
    # Print learning curve summary
    print("\nLearning Curve Summary:")
    print("-" * 50)
    print(f"{'Labeled':>10} | {'Val Spearman':>12} | {'Test Spearman':>12}")
    print("-" * 50)
    for i in range(len(history["round"])):
        print(f"{history['labeled_size'][i]:>10} | {history['val_spearman'][i]:>12.4f} | {history['test_spearman'][i]:>12.4f}")
    
    return projector, model, decoder, history


if __name__ == "__main__":
    args = parse_args()
    active_learning_loop(args)
