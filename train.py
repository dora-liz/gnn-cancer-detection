"""
Train heterogeneous GNN for cell line → drug ranking with IC50 prediction.

Features:
- GraphSAGE with feature-dominant projection (16-dim default embeddings)
- Metrics: log-space MSE, IC50 RMSE, Spearman correlation, top-k precision
- Optional comparison of embedding dims: 16 vs 32 vs 64

Outputs:
- gnn_model_trained_cpu.pt / link_predictor_trained_cpu.pt
- training_artifacts.pt (hyperparameters, history, ranking metrics)
"""

import argparse
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch_geometric.nn import to_hetero

from gnn_model import GNN, LinkPredictor, FeatureProjector

EDGE_TYPE = ("cell_line", "treated_with", "drug")
REV_EDGE_TYPE = ("drug", "rev_treated_with", "cell_line")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Train GNN drug-effect predictor")
    p.add_argument("--train-data", default="train_data.pt")
    p.add_argument("--val-data", default="val_data.pt")
    p.add_argument("--test-data", default="test_data.pt")

    # Architecture - increased capacity
    p.add_argument("--hidden-channels", type=int, default=128)
    p.add_argument("--out-channels", type=int, default=32, help="Embedding dim (GraphSAGE output)")
    p.add_argument("--decoder-hidden", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--project-dim", type=int, default=256, help="Feature projection dim")
    p.add_argument("--num-gnn-layers", type=int, default=3, help="Number of GNN layers")
    p.add_argument("--num-decoder-layers", type=int, default=4, help="Number of decoder layers")

    # Training - improved settings
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm")
    p.add_argument("--ranking-weight", type=float, default=0.1, help="Weight for ranking loss")
    p.add_argument("--lr-scheduler", action="store_true", default=True, help="Use LR scheduler")

    # Comparison
    p.add_argument("--compare-dims", action="store_true", help="Run comparison of 16 vs 32 vs 64 dims")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def pick_device(arg: str) -> torch.device:
    if arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_data(path: str, device: torch.device):
    if not Path(path).exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location=device, weights_only=False)


def get_targets(data):
    """Get edge indices and labels. Labels are log10-transformed IC50."""
    store = data[EDGE_TYPE]
    labels = store.pos_edge_label
    # Log-transform: clamp to avoid log(0)
    labels_log = torch.log10(labels.clamp(min=1e-6))
    return store.pos_edge_label_index, labels_log


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(preds: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """Compute log-MSE, IC50 RMSE, Spearman.
    
    preds and labels are assumed to be in log10(IC50) space.
    """
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()

    # Log-space MSE
    log_mse = float(((preds - labels) ** 2).mean().item())

    # RMSE in IC50 units: convert back from log-space
    ic50_pred = torch.pow(10.0, preds)
    ic50_true = torch.pow(10.0, labels)
    ic50_rmse = float(torch.sqrt(((ic50_pred - ic50_true) ** 2).mean()).item())

    # Spearman correlation (on log-space, same ranking as raw)
    if len(preds_np) > 1:
        spear, _ = spearmanr(preds_np, labels_np)
        spear = float(spear) if not math.isnan(spear) else 0.0
    else:
        spear = 0.0

    return {"log_mse": log_mse, "ic50_rmse": ic50_rmse, "spearman": spear}


def top_k_precision(preds: torch.Tensor, labels: torch.Tensor, k: int = 10) -> float:
    """Precision@k: fraction of top-k predicted drugs that are in top-k true."""
    if preds.shape[0] < k:
        k = preds.shape[0]
    pred_topk = set(torch.argsort(preds)[:k].tolist())
    true_topk = set(torch.argsort(labels)[:k].tolist())
    return len(pred_topk & true_topk) / k


def ranking_metrics_per_cell(
    projector, model, decoder, data, device
) -> Dict[str, float]:
    """Compute ranking metrics averaged over each cell line."""
    projector.eval()
    model.eval()
    decoder.eval()

    with torch.no_grad():
        proj_x = projector(data.x_dict)
        emb = model(proj_x, data.edge_index_dict)

    edge_idx, labels = get_targets(data)
    cell_idx = edge_idx[0]
    drug_idx = edge_idx[1]

    cell_emb = emb["cell_line"][cell_idx]
    drug_emb = emb["drug"][drug_idx]

    with torch.no_grad():
        preds = decoder(cell_emb, drug_emb)

    # Group by cell line
    unique_cells = cell_idx.unique()
    spearman_list: List[float] = []
    topk_list: List[float] = []

    for c in unique_cells:
        mask = cell_idx == c
        if mask.sum() < 5:
            continue
        p = preds[mask]
        l = labels[mask]
        m = compute_metrics(p, l)
        spearman_list.append(m["spearman"])
        topk_list.append(top_k_precision(p, l, k=10))

    return {
        "mean_spearman": float(sum(spearman_list) / max(len(spearman_list), 1)),
        "mean_topk10_precision": float(sum(topk_list) / max(len(topk_list), 1)),
    }


# ---------------------------------------------------------------------------
# Ranking Loss
# ---------------------------------------------------------------------------


def pairwise_ranking_loss(preds: torch.Tensor, labels: torch.Tensor, num_samples: int = 512, margin: float = 0.1) -> torch.Tensor:
    """Margin ranking loss: encourages correct ordering of drug responses.
    
    For drug ranking, LOWER predicted IC50 should mean MORE effective.
    So we want pred[i] < pred[j] when label[i] < label[j].
    """
    n = preds.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=preds.device)
    
    # Sample pairs efficiently
    num_samples = min(num_samples, n * (n - 1) // 2)
    
    # Random sampling of pairs
    idx_i = torch.randint(0, n, (num_samples,), device=preds.device)
    idx_j = torch.randint(0, n, (num_samples,), device=preds.device)
    
    # Ensure different indices
    same_mask = idx_i == idx_j
    idx_j[same_mask] = (idx_j[same_mask] + 1) % n
    
    pred_i, pred_j = preds[idx_i], preds[idx_j]
    label_i, label_j = labels[idx_i], labels[idx_j]
    
    # Target: -1 if label_i < label_j (i is better), +1 otherwise
    target = torch.sign(label_j - label_i)
    
    # Margin ranking loss
    loss = F.margin_ranking_loss(pred_j, pred_i, target, margin=margin, reduction='mean')
    return loss


class CombinedLoss(nn.Module):
    """MSE + Ranking loss for better ranking accuracy."""
    
    def __init__(self, ranking_weight: float = 0.1, margin: float = 0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.ranking_weight = ranking_weight
        self.margin = margin
    
    def forward(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        mse_loss = self.mse(preds, labels)
        if self.ranking_weight > 0:
            rank_loss = pairwise_ranking_loss(preds, labels, margin=self.margin)
            return mse_loss + self.ranking_weight * rank_loss
        return mse_loss


# ---------------------------------------------------------------------------
# Model init
# ---------------------------------------------------------------------------


def init_models(data, args, device, out_channels: int = None):
    oc = out_channels if out_channels else args.out_channels

    # Feature projector runs before to_hetero
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
        out_channels=oc,
        dropout=args.dropout,
        num_layers=getattr(args, 'num_gnn_layers', 3),
    )
    model = to_hetero(encoder, data.metadata(), aggr="sum").to(device)

    decoder = LinkPredictor(
        in_channels=oc * 2,
        hidden_channels=args.decoder_hidden,
        out_channels=1,
        dropout=args.dropout,
        num_layers=getattr(args, 'num_decoder_layers', 4),
    ).to(device)

    # Lazy init GNN with projected features
    with torch.no_grad():
        proj_x = projector(data.x_dict)
        _ = model(proj_x, data.edge_index_dict)

    return projector, model, decoder


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def forward_pass(projector, model, decoder, data, loss_fn):
    edge_idx, labels = get_targets(data)
    proj_x = projector(data.x_dict)
    emb = model(proj_x, data.edge_index_dict)

    cell_emb = emb["cell_line"][edge_idx[0]]
    drug_emb = emb["drug"][edge_idx[1]]

    preds = decoder(cell_emb, drug_emb)
    loss = loss_fn(preds, labels)
    return loss, preds.detach(), labels.detach()


def evaluate(projector, model, decoder, data, loss_fn):
    projector.eval()
    model.eval()
    decoder.eval()
    with torch.no_grad():
        loss, preds, labels = forward_pass(projector, model, decoder, data, loss_fn)
    m = compute_metrics(preds, labels)
    m["loss"] = loss.item()
    return m


def train_model(
    train_data,
    val_data,
    args,
    device,
    out_channels: int = None,
    verbose: bool = True,
):
    projector, model, decoder = init_models(train_data, args, device, out_channels)
    oc = out_channels if out_channels else args.out_channels

    all_params = list(projector.parameters()) + list(model.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(
        all_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Combined loss with ranking component
    ranking_weight = getattr(args, 'ranking_weight', 0.1)
    loss_fn = CombinedLoss(ranking_weight=ranking_weight)
    eval_loss_fn = nn.MSELoss()  # For evaluation, use pure MSE
    
    # Learning rate scheduler
    use_scheduler = getattr(args, 'lr_scheduler', True)
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-6
        )
    
    grad_clip = getattr(args, 'grad_clip', 1.0)

    best_val_loss = float("inf")
    best_val_spearman = float("-inf")
    patience_ctr = 0
    best_state = None

    history: Dict[str, List[float]] = {
        "train_log_mse": [],
        "train_ic50_rmse": [],
        "train_spearman": [],
        "val_log_mse": [],
        "val_ic50_rmse": [],
        "val_spearman": [],
        "learning_rate": [],
    }

    for epoch in range(1, args.epochs + 1):
        projector.train()
        model.train()
        decoder.train()

        optimizer.zero_grad()
        loss, preds, labels = forward_pass(projector, model, decoder, train_data, loss_fn)
        loss.backward()
        
        # Gradient clipping for stability
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(all_params, grad_clip)
        
        optimizer.step()

        train_m = compute_metrics(preds, labels)
        val_m = evaluate(projector, model, decoder, val_data, eval_loss_fn)
        
        current_lr = optimizer.param_groups[0]['lr']
        history["learning_rate"].append(current_lr)

        for k in ["log_mse", "ic50_rmse", "spearman"]:
            history[f"train_{k}"].append(train_m[k])
            history[f"val_{k}"].append(val_m[k])

        # Use combined criterion: val loss + negative spearman (we want higher spearman)
        val_score = val_m["loss"] - 0.5 * val_m["spearman"]  # Lower is better
        
        if val_score < best_val_loss or val_m["spearman"] > best_val_spearman:
            if val_score < best_val_loss:
                best_val_loss = val_score
            if val_m["spearman"] > best_val_spearman:
                best_val_spearman = val_m["spearman"]
            patience_ctr = 0
            best_state = {
                "projector": {k: v.cpu().clone() for k, v in projector.state_dict().items()},
                "model": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "decoder": {k: v.cpu().clone() for k, v in decoder.state_dict().items()},
                "epoch": epoch,
                "val_metrics": val_m,
            }
        else:
            patience_ctr += 1
        
        # Update LR scheduler
        if use_scheduler:
            scheduler.step(val_m["loss"])

        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(
                f"Epoch {epoch:03d} | LR {current_lr:.2e} | "
                f"Train logMSE {train_m['log_mse']:.4f} Spear {train_m['spearman']:.3f} | "
                f"Val logMSE {val_m['log_mse']:.4f} Spear {val_m['spearman']:.3f}"
            )

        if patience_ctr >= args.patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError("No best state recorded")

    projector.load_state_dict(best_state["projector"])
    model.load_state_dict(best_state["model"])
    decoder.load_state_dict(best_state["decoder"])

    return projector, model, decoder, best_state, history


# ---------------------------------------------------------------------------
# Compare embedding dimensions
# ---------------------------------------------------------------------------


def compare_embedding_dims(train_data, val_data, test_data, args, device):
    dims = [16, 32, 64]
    results = {}

    for d in dims:
        print(f"\n{'='*60}\nTraining with embedding dim = {d}\n{'='*60}")
        projector, model, decoder, best_state, _ = train_model(
            train_data, val_data, args, device, out_channels=d, verbose=True
        )

        # Test metrics
        if test_data is not None:
            test_m = evaluate(projector, model, decoder, test_data, nn.MSELoss())
            rank_m = ranking_metrics_per_cell(projector, model, decoder, test_data, device)
            test_m.update(rank_m)
        else:
            test_m = best_state["val_metrics"]
            rank_m = ranking_metrics_per_cell(projector, model, decoder, val_data, device)
            test_m.update(rank_m)

        results[d] = test_m
        print(f"Dim {d} | Test logMSE {test_m['log_mse']:.4f} RMSE {test_m['ic50_rmse']:.2f} Spear {test_m['spearman']:.3f} "
              f"TopK-10 {test_m['mean_topk10_precision']:.3f}")

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY (embedding dimension)")
    print("=" * 60)
    for d in dims:
        r = results[d]
        print(f"  dim={d:2d} | logMSE={r['log_mse']:.4f} | IC50_RMSE={r['ic50_rmse']:.2f} | "
              f"Spearman={r['spearman']:.3f} | Top-10 Prec={r['mean_topk10_precision']:.3f}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    device = pick_device(args.device)

    print("=" * 60)
    print("TRAINING HETEROGENEOUS GNN FOR DRUG RANKING")
    print(f"Embedding dim: {args.out_channels} | Dropout: {args.dropout}")
    print(f"Feature projection dim: {args.project_dim}")
    print(f"Device: {device}")
    print("=" * 60)

    train_data = load_data(args.train_data, device).to(device)
    val_data = load_data(args.val_data, device).to(device)

    test_data = None
    if Path(args.test_data).exists():
        test_data = load_data(args.test_data, device).to(device)

    if args.compare_dims:
        compare_embedding_dims(train_data, val_data, test_data, args, device)
        return

    projector, model, decoder, best_state, history = train_model(
        train_data, val_data, args, device, verbose=True
    )

    # Final ranking metrics
    eval_data = test_data if test_data is not None else val_data
    final_m = evaluate(projector, model, decoder, eval_data, nn.MSELoss())
    rank_m = ranking_metrics_per_cell(projector, model, decoder, eval_data, device)
    final_m.update(rank_m)

    print("\n" + "=" * 60)
    print("FINAL METRICS")
    print("=" * 60)
    print(f"  log-space MSE: {final_m['log_mse']:.4f}")
    print(f"  IC50 RMSE:     {final_m['ic50_rmse']:.2f}")
    print(f"  Spearman:      {final_m['spearman']:.4f}")
    print(f"  Top-10 Prec:   {final_m['mean_topk10_precision']:.4f}")

    # Save
    torch.save(projector.state_dict(), "feature_projector_trained_cpu.pt")
    torch.save(model.state_dict(), "gnn_model_trained_cpu.pt")
    torch.save(decoder.state_dict(), "link_predictor_trained_cpu.pt")

    artifacts = {
        "edge_type": EDGE_TYPE,
        "metadata": train_data.metadata(),
        "hyperparameters": {
            "hidden_channels": args.hidden_channels,
            "out_channels": args.out_channels,
            "decoder_hidden": args.decoder_hidden,
            "dropout": args.dropout,
            "project_dim": args.project_dim,
            "num_gnn_layers": getattr(args, 'num_gnn_layers', 3),
            "num_decoder_layers": getattr(args, 'num_decoder_layers', 4),
        },
        "best_epoch": best_state["epoch"],
        "final_metrics": final_m,
        "history": history,
        "cell_feature_dim": int(train_data["cell_line"].x.shape[1]),
        "drug_feature_dim": int(train_data["drug"].x.shape[1]),
        "num_cell_lines": int(train_data["cell_line"].x.shape[0]),
        "num_drugs": int(train_data["drug"].x.shape[0]),
    }
    torch.save(artifacts, "training_artifacts.pt")

    print("\nSaved: feature_projector_trained_cpu.pt, gnn_model_trained_cpu.pt, link_predictor_trained_cpu.pt, training_artifacts.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
