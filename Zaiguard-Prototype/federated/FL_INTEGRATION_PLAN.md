# Integrate Federated Learning into Zaiguard-Prototype
Bring the working FL code from `FL-Project/` into the `Zaiguard-Prototype/federated/` directory, adapted to train on the Prototype's **alert embedding data** (138-d vectors from `ConfirmedAlert`) using Flower's Deployment Runtime. The goal: each camera node trains a lightweight `AlertImportanceClassifier` head locally and sends only weight deltas to a central FedAvg/FedProx server — no raw frames leave the node.
**Constraint:** All new code goes into `federated/` as new files. No modifications to existing Prototype files.
---
## Proposed Changes
### Component 1 — FL Task Module (data + model + train/test)
#### [NEW] `federated/fl_task.py`
Port + adapt from FL-Project's `classification_embeddings_prox.py`. Contains:
- **`AlertImportanceClassifier`** — the same 138→64→16→1 binary MLP with Sigmoid
- **`FocalLoss`** — handles the 99/1 class imbalance from the FedProx variant
- **`load_local_embeddings()`** — reads real alert JSONs from `outputs/alerts/` and constructs 138-d embedding vectors from `AlertMeta` fields (bbox_norm, centroid_norm, area_px, aspect_ratio, spatial, temporal). Falls back to synthetic data when no real alerts exist yet (same pattern as the FL-Project demo).
- **`load_centralized_testset()`** — for server-side global evaluation
- **`train()`** / **`test()`** — standard train/eval loops using Focal Loss + Adam
Key adaptation: instead of always using synthetic `torch.randn(N, 138)`, the task module scans `outputs/alerts/<date>/*.json`, parses the `AlertMeta` and builds real embeddings when available. Labels come from the alert's `severity` field (high/critical → 1, low/medium → 0). This is the bridge between the Prototype's live alert pipeline and the FL training loop.
---
### Component 2 — Flower Client App
#### [NEW] `federated/fl_client.py`
Port from FL-Project's `zaiguard_fedprox_demo/client_app.py`. Uses Flower's Message API (`@app.train()` / `@app.evaluate()` decorators). Each camera node:
1. Receives global model weights via `ArrayRecord`
2. Loads local embedding data (real alerts from disk or synthetic fallback)
3. Trains for `local-epochs` rounds with Focal Loss
4. Returns updated weights + metrics (train_loss, num-examples)
---
### Component 3 — Flower Server App
#### [NEW] `federated/fl_server.py`
Port from FL-Project's `zaiguard_fedprox_demo/server_app.py`. Central aggregation using **FedProx** (with proximal μ=0.1 to handle non-IID data across cameras). After final round, saves aggregated model to `models/fl_alert_model.pt` if `save-model` is true. Includes `global_evaluate()` callback for per-round accuracy tracking.
---
### Component 4 — FL Model Loader (standalone inference)
#### [NEW] `federated/fl_model_loader.py`
A lightweight module that:
- Loads the latest FL-trained model from `models/fl_alert_model.pt` (if it exists)
- Provides a `predict(embedding_vector) → importance_score` function
- Provides `build_embedding_from_alert(alert_dict) → torch.Tensor` to construct the 138-d vector from alert JSON
- Thread-safe (model loaded once, inference is `torch.no_grad()`)
This can be imported by the alerter in the future without modifying alerter.py now.
---
### Component 5 — FL-specific Configuration
#### [NEW] `federated/fl_config.yaml`
Standalone config for the FL subsystem:
```yaml
federated:
  enabled: false
  model_path: "models/fl_alert_model.pt"
  importance_threshold: 0.7
  alerts_dir: "outputs/alerts"
  synthetic_fallback: true
  synthetic_samples: 1000
```
#### [NEW] `federated/pyproject.toml`
Flower FAB configuration for running the FL training separately via `flwr run`:
```toml
[tool.flwr.app.components]
serverapp = "federated.fl_server:app"
clientapp = "federated.fl_client:app"
[tool.flwr.app.config]
num-server-rounds = 5
local-epochs = 10
learning-rate = 0.01
batch-size = 32
save-model = true
```
---
### Component 6 — Documentation
#### [UPDATE] `federated/__init__.py`
Add module docstring explaining the federated package purpose and how to run it.
---
## Open Questions
1. **Real vs. synthetic training data**: Should the task module read real alerts from `outputs/alerts/` immediately, or keep synthetic data for now?
2. **FedAvg vs FedProx**: The FL-Project has both. Plan uses FedProx (better for non-IID camera data). Want both strategies with a config toggle, or just FedProx?
3. **Model save location**: Plan saves to `models/fl_alert_model.pt`. Is this the right spot?
4. **Severity escalation**: Should FL importance scores auto-escalate alert severity, or stay purely informational in `extra["fl_importance"]`?
5. **Dependencies**: `torch` and `flwr` are needed. Should they be added to the main `pyproject.toml`, or kept separate (install manually when using FL)?
---
## Verification Plan
### Manual Verification
- Verify all new files import correctly: `python -c "from federated.fl_task import AlertImportanceClassifier"`
- Run existing Prototype test suite to confirm zero regressions: `uv run pytest tests/ -v`
- Verify FL model loading/inference works standalone with a quick script
- If Flower infra is available (SuperLink + SuperNodes), run a short FL training round to confirm end-to-end