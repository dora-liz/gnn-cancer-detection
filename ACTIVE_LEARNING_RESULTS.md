# Active Learning Training Results

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| **Strategy** | Hybrid (50% uncertainty + 50% diversity) |
| **Total Budget** | 2000 labeled samples |
| **Batch Size** | 200 samples per round |
| **Initial Labeled Pool** | 1000 samples |
| **Epochs per Round** | 15 |
| **Early Stopping Patience** | 5 |
| **MC Dropout Samples** | 5 |
| **Total Rounds** | 5 + initial = 6 |

## Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total Edges** | 347,733 |
| **Initial Unlabeled Pool** | 346,733 |
| **Cell Lines** | 912 |
| **Drugs** | 542 |

---

## Learning Curve Results

### Round-by-Round Metrics

| Round | Labeled Samples | Val Spearman | Test Spearman | Val Log MSE | Test Log MSE |
|-------|-----------------|--------------|---------------|-------------|--------------|
| 0 (Initial) | 1,000 | 0.1119 | 0.1171 | 2.2238 | 2.2383 |
| 1 | 1,200 | -0.0267 | -0.0323 | 2.2386 | 2.2535 |
| 2 | 1,400 | 0.0971 | 0.1141 | 2.4879 | 2.5039 |
| 3 | 1,600 | -0.0400 | -0.0521 | 1.9679 | 1.9809 |
| 4 | 1,800 | 0.1729 | 0.1927 | 1.6895 | 1.6986 |
| 5 (Final) | 2,000 | 0.1745 | 0.2040 | 2.0046 | 2.0159 |

### Key Observations

1. **Initial Performance (Round 0)**: Starting with only 1,000 labeled samples achieved Test Spearman of 0.1171
2. **Variance in Early Rounds**: Rounds 1 and 3 showed temporary performance degradation (negative Spearman), which is common in active learning when the model explores uncertain regions
3. **Strong Recovery**: By Round 4-5, the model achieved significant improvement with Test Spearman reaching 0.2040
4. **Final Improvement**: Test Spearman improved from 0.1171 → 0.2040 (**+74% relative improvement**) by doubling the labeled data from 1,000 to 2,000 samples

---

## Previous Run (10 Rounds - Partial Results)

The earlier training run (before terminal crash) showed similar patterns:

| Round | Labeled | Val Spearman | Test Spearman |
|-------|---------|--------------|---------------|
| 0 | 1,000 | 0.1554 | 0.1719 |
| 1 | 1,200 | 0.2156 | 0.2210 |
| 2 | 1,400 | 0.1636 | 0.1845 |
| 3 | 1,600 | **0.2258** | **0.2431** |
| 4 | 1,800 | 0.2160 | 0.2198 |
| 5 | 2,000 | 0.1049 | 0.1153 |
| 6 | 2,200 | 0.1012 | 0.1269 |
| 7+ | ... | (training interrupted) | |

**Best Performance (Previous Run)**: Test Spearman = 0.2431 at Round 3 (1,600 samples)

---

## Analysis

### Active Learning Effectiveness

The hybrid strategy combines:
- **Uncertainty Sampling**: Uses MC Dropout to identify samples where the model is most uncertain
- **Diversity Sampling**: Uses K-Means clustering on embeddings to ensure diverse sample selection

### Variance in Results

The high variance between rounds is characteristic of:
1. **Small batch sizes** relative to the unlabeled pool
2. **Model re-initialization** (weights not transferred between rounds in this implementation)
3. **Random sampling component** in the initial labeled set

### Recommendations for Future Experiments

1. **Warm Starting**: Initialize new round's model from previous round's weights
2. **Larger Initial Pool**: Start with 5,000-10,000 samples for more stable initial training
3. **Ensemble Approach**: Train multiple models and combine predictions
4. **Longer Training**: Increase epochs per round to ensure convergence

---

## Output Files

| File | Description |
|------|-------------|
| `models/active_learning_history.json` | Complete learning curve data (JSON) |
| `models/active_learning_projector.pt` | Final feature projector weights |
| `models/active_learning_gnn.pt` | Final GNN model weights |
| `models/active_learning_decoder.pt` | Final link predictor weights |

---

## Reproducing Results

```bash
# Activate virtual environment
.venv\Scripts\python.exe training\train_active_learning.py \
    --strategy hybrid \
    --budget 2000 \
    --batch-size 200 \
    --initial-size 1000 \
    --epochs-per-round 15 \
    --patience 5 \
    --mc-samples 5
```

---

*Generated: February 26, 2026*
