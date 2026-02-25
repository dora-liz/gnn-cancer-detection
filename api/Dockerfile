# Dockerfile for Drug Ranking GNN Model
#
# Build:   docker build -t drug-ranking-api .
# Run:     docker run -p 5000:5000 drug-ranking-api
#
# The API will be available at http://localhost:5000

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
    torch-geometric==2.4.0 \
    flask==3.0.0 \
    scipy==1.11.0

# Copy model files
COPY gnn_model.py .
COPY api_server.py .
COPY train_data.pt .
COPY training_artifacts.pt .
COPY feature_projector_trained_cpu.pt .
COPY gnn_model_trained_cpu.pt .
COPY link_predictor_trained_cpu.pt .

# Optional: drug names mapping
COPY drug_names.json* ./

EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:5000/health || exit 1

CMD ["python", "api_server.py"]
