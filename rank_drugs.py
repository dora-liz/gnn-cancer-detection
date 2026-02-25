"""
Rank effective drugs for a new cell line using trained GNN + link predictor,
with explainability via input-gradient feature attribution.

Example:
python rank_drugs.py --cell-features new_cell_features.pt --top-k 20
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import to_hetero

from gnn_model import GNN, LinkPredictor, FeatureProjector

EDGE_TYPE = ("cell_line", "treated_with", "drug")
REV_EDGE_TYPE = ("drug", "rev_treated_with", "cell_line")


def parse_args():
    parser = argparse.ArgumentParser(description="Rank drugs for a new cell line")
    parser.add_argument("--base-graph", default="train_data.pt", help="Graph with known drugs/cell lines")
    parser.add_argument("--artifacts", default="training_artifacts.pt")
    parser.add_argument("--projector-weights", default="feature_projector_trained_cpu.pt")
    parser.add_argument("--model-weights", default="gnn_model_trained_cpu.pt")
    parser.add_argument("--decoder-weights", default="link_predictor_trained_cpu.pt")
    parser.add_argument("--cell-features", required=True, help="Path to tensor/list for one new cell line")
    parser.add_argument("--drug-names", default="", help="Optional JSON list of drug names")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--explain-top-features", type=int, default=15)
    parser.add_argument("--output", default="drug_ranking.json")
    return parser.parse_args()


def load_cell_features(path: str) -> torch.Tensor:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Cell feature file not found: {path}")

    if p.suffix == ".pt":
        obj = torch.load(p, map_location="cpu", weights_only=False)
    elif p.suffix == ".json":
        obj = json.loads(p.read_text(encoding="utf-8"))
    else:
        raise ValueError("cell-features must be .pt or .json")

    if isinstance(obj, torch.Tensor):
        features = obj.float()
    else:
        features = torch.tensor(obj, dtype=torch.float32)

    if features.ndim == 1:
        features = features.unsqueeze(0)

    if features.shape[0] != 1:
        raise ValueError(f"Expected exactly one cell line feature row, got shape={tuple(features.shape)}")

    return features


def build_augmented_graph(base_data: HeteroData, new_cell_features: torch.Tensor) -> Tuple[HeteroData, int]:
    data = base_data.clone()

    old_x = data["cell_line"].x
    new_index = old_x.shape[0]

    data["cell_line"].x = torch.cat([old_x, new_cell_features], dim=0)

    if EDGE_TYPE in data.edge_types:
        et_store = data[EDGE_TYPE]
        if hasattr(et_store, "edge_index"):
            et_store.edge_index = et_store.edge_index.clone()
    if REV_EDGE_TYPE in data.edge_types:
        rt_store = data[REV_EDGE_TYPE]
        if hasattr(rt_store, "edge_index"):
            rt_store.edge_index = rt_store.edge_index.clone()

    return data, new_index


def get_drug_names(num_drugs: int, path: str) -> List[str]:
    if path and Path(path).exists():
        items = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(items, list) and len(items) == num_drugs:
            return [str(x) for x in items]
    return [f"drug_{i}" for i in range(num_drugs)]


def load_models(base_graph, artifacts):
    hyper = artifacts["hyperparameters"]

    # FeatureProjector
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


def rank_drugs(projector, model, decoder, graph, new_cell_index: int) -> torch.Tensor:
    projector.eval()
    model.eval()
    decoder.eval()
    with torch.no_grad():
        proj_x = projector(graph.x_dict)
        emb = model(proj_x, graph.edge_index_dict)
        new_cell_emb = emb["cell_line"][new_cell_index].unsqueeze(0)
        drug_emb = emb["drug"]
        repeated_cell = new_cell_emb.repeat(drug_emb.shape[0], 1)
        preds = decoder(repeated_cell, drug_emb)
    return preds


def explain_top_drug(
    projector,
    model,
    decoder,
    graph: HeteroData,
    new_cell_features: torch.Tensor,
    top_drug_idx: int,
    base_cell_count: int,
):
    """Compute input×gradient attribution on the new cell line's raw features."""
    explain_graph = graph.clone()
    explain_graph["cell_line"].x = explain_graph["cell_line"].x.clone().detach()
    explain_graph["cell_line"].x.requires_grad_(True)

    projector.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)
    decoder.zero_grad(set_to_none=True)

    proj_x = projector(explain_graph.x_dict)
    emb = model(proj_x, explain_graph.edge_index_dict)
    new_cell_emb = emb["cell_line"][base_cell_count].unsqueeze(0)
    drug_emb = emb["drug"][top_drug_idx].unsqueeze(0)
    pred = decoder(new_cell_emb, drug_emb).squeeze()
    pred.backward()

    grad = explain_graph["cell_line"].x.grad[base_cell_count]
    inp = new_cell_features.squeeze(0)
    attribution = (grad * inp).abs()

    return pred.detach().item(), attribution.detach().cpu()


def main():
    args = parse_args()

    base_graph = torch.load(args.base_graph, map_location="cpu", weights_only=False)
    artifacts = torch.load(args.artifacts, map_location="cpu", weights_only=False)

    new_cell_features = load_cell_features(args.cell_features)
    expected_dim = int(artifacts["cell_feature_dim"])
    if new_cell_features.shape[1] != expected_dim:
        raise ValueError(
            f"New cell feature dim mismatch. expected={expected_dim}, got={new_cell_features.shape[1]}"
        )

    graph, new_cell_idx = build_augmented_graph(base_graph, new_cell_features)

    projector, model, decoder = load_models(base_graph, artifacts)
    projector.load_state_dict(torch.load(args.projector_weights, map_location="cpu", weights_only=False))
    model.load_state_dict(torch.load(args.model_weights, map_location="cpu", weights_only=False))
    decoder.load_state_dict(torch.load(args.decoder_weights, map_location="cpu", weights_only=False))

    # Lazy init GNN
    with torch.no_grad():
        proj_x = projector(graph.x_dict)
        _ = model(proj_x, graph.edge_index_dict)

    preds = rank_drugs(projector, model, decoder, graph, new_cell_idx)

    top_k = min(args.top_k, preds.shape[0])
    ranked_idx = torch.argsort(preds, descending=False)[:top_k]

    drug_names = get_drug_names(int(preds.shape[0]), args.drug_names)

    top_drug_idx = int(ranked_idx[0].item())
    top_pred, attributions = explain_top_drug(
        projector=projector,
        model=model,
        decoder=decoder,
        graph=graph,
        new_cell_features=new_cell_features,
        top_drug_idx=top_drug_idx,
        base_cell_count=base_graph["cell_line"].x.shape[0],
    )

    explain_k = min(args.explain_top_features, attributions.shape[0])
    feat_idx = torch.argsort(attributions, descending=True)[:explain_k]

    ranking = []
    for i, idx in enumerate(ranked_idx.tolist(), start=1):
        ranking.append(
            {
                "rank": i,
                "drug_index": int(idx),
                "drug_name": drug_names[idx],
                "predicted_effect_ic50": float(preds[idx].item()),
            }
        )

    explanation = {
        "top_drug_index": top_drug_idx,
        "top_drug_name": drug_names[top_drug_idx],
        "top_drug_predicted_effect_ic50": float(top_pred),
        "method": "input_x_gradient_on_new_cell_features",
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

    output = {
        "num_drugs_scored": int(preds.shape[0]),
        "ranking": ranking,
        "metrics_note": "Predicted values are log10(IC50). Lower = more effective.",
        "explainability": explanation,
    }

    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("=" * 60)
    print("DRUG RANKING COMPLETE")
    print("=" * 60)
    print(f"Output saved to: {args.output}")
    print(f"\nTop-{top_k} ranked drugs (lowest predicted IC50):")
    for r in ranking[:5]:
        print(f"  #{r['rank']}: {r['drug_name']} (pred logIC50={r['predicted_effect_ic50']:.3f})")
    if top_k > 5:
        print(f"  ...and {top_k - 5} more in {args.output}")
    print(f"\nTop drug: {drug_names[top_drug_idx]}")
    print(f"Top {args.explain_top_features} influential features saved in explainability section.")
    print("=" * 60)


if __name__ == "__main__":
    main()
