"""
Simple Flask API for drug ranking predictions.

Usage:
    pip install flask
    python api_server.py

Then send POST request:
    curl -X POST http://localhost:5000/predict \
         -H "Content-Type: application/json" \
         -d '{"gene_expression": [0.1, 0.2, ...]}'  # 17,737 values
"""

import json
import sys
from pathlib import Path

# Set up paths relative to project root
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Add parent directory to import gnn_model
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from flask import Flask, request, jsonify
from torch_geometric.nn import to_hetero

from gnn_model import GNN, LinkPredictor, FeatureProjector

app = Flask(__name__)

# Global model objects (loaded once at startup)
projector = None
model = None
decoder = None
base_graph = None
artifacts = None


def load_models():
    """Load trained models at startup."""
    global projector, model, decoder, base_graph, artifacts
    
    print("Loading models...")
    
    base_graph = torch.load(DATA_DIR / "gdsc_processed_train.pt", map_location="cpu", weights_only=False)
    artifacts = torch.load(MODELS_DIR / "training_artifacts.pt", map_location="cpu", weights_only=False)
    hyper = artifacts["hyperparameters"]
    
    # Build model architecture
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
    
    # Load weights
    projector.load_state_dict(torch.load(MODELS_DIR / "feature_projector_trained_cpu.pt", map_location="cpu", weights_only=False))
    model.load_state_dict(torch.load(MODELS_DIR / "gnn_model_trained_cpu.pt", map_location="cpu", weights_only=False))
    decoder.load_state_dict(torch.load(MODELS_DIR / "link_predictor_trained_cpu.pt", map_location="cpu", weights_only=False))
    
    # Initialize lazy modules
    with torch.no_grad():
        proj_x = projector(base_graph.x_dict)
        _ = model(proj_x, base_graph.edge_index_dict)
    
    projector.eval()
    model.eval()
    decoder.eval()
    
    print("Models loaded successfully!")


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict drug rankings for a new cell line.
    
    Input JSON:
        {"gene_expression": [float, float, ...]}  # 19,215 values
        
    Output JSON:
        {
            "ranked_drugs": [
                {"rank": 1, "drug_index": 45, "predicted_ic50": 0.82},
                ...
            ],
            "top_features": [{"index": 123, "importance": 0.95}, ...]
        }
    """
    try:
        data = request.get_json()
        
        if "gene_expression" not in data:
            return jsonify({"error": "Missing 'gene_expression' field"}), 400
        
        gene_expr = data["gene_expression"]
        expected_dim = int(artifacts["cell_feature_dim"])
        
        if len(gene_expr) != expected_dim:
            return jsonify({
                "error": f"Expected {expected_dim} features, got {len(gene_expr)}"
            }), 400
        
        # Create tensor
        new_cell = torch.tensor([gene_expr], dtype=torch.float32)
        
        # Augment graph with new cell line
        graph = base_graph.clone()
        graph["cell_line"].x = torch.cat([graph["cell_line"].x, new_cell], dim=0)
        new_cell_idx = base_graph["cell_line"].x.shape[0]
        
        # Get predictions
        with torch.no_grad():
            proj_x = projector(graph.x_dict)
            emb = model(proj_x, graph.edge_index_dict)
            new_cell_emb = emb["cell_line"][new_cell_idx].unsqueeze(0)
            drug_emb = emb["drug"]
            repeated_cell = new_cell_emb.repeat(drug_emb.shape[0], 1)
            preds = decoder(repeated_cell, drug_emb)
        
        # Convert from log-space to IC50
        ic50_preds = torch.pow(10.0, preds)
        
        # Rank (lower IC50 = better)
        ranked_idx = torch.argsort(preds).tolist()
        
        top_k = min(20, len(ranked_idx))
        ranked_drugs = []
        for i, idx in enumerate(ranked_idx[:top_k], 1):
            ranked_drugs.append({
                "rank": i,
                "drug_index": idx,
                "predicted_log_ic50": float(preds[idx].item()),
                "predicted_ic50": float(ic50_preds[idx].item()),
            })
        
        return jsonify({
            "num_drugs": len(ranked_idx),
            "ranked_drugs": ranked_drugs,
            "note": "Lower IC50 = more effective drug"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/drugs", methods=["GET"])
def list_drugs():
    """List all available drugs."""
    num_drugs = int(artifacts["num_drugs"])
    return jsonify({
        "num_drugs": num_drugs,
        "drug_indices": list(range(num_drugs)),
        "note": "Use drug_names.json to map indices to names if available"
    })


if __name__ == "__main__":
    load_models()
    print("\n" + "=" * 50)
    print("API Server Ready!")
    print("=" * 50)
    print("Endpoints:")
    print("  GET  /health  - Health check")
    print("  GET  /drugs   - List available drugs")
    print("  POST /predict - Get drug rankings for new cell line")
    print("=" * 50 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
