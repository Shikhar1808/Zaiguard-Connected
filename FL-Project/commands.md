# Commands Reference — Flower FL Project

Every command in this document is run from inside `~/Desktop/learning/flower/p2` with the venv activated:
```bash
source .venv/bin/activate
```
Or prefix every command with `uv run` without activating.

---

## Setup Commands

### Initialize the project
```bash
uv init my-mnist-fl --python 3.12
```
Creates the project folder with `pyproject.toml`, `README.md`, `.python-version`, `.gitignore`, and a git repo. The `.gitignore` matters — it excludes `.venv/` from Flower's FAB bundler scan, which prevents it from trying to walk gigabytes of installed packages.

### Add dependencies
```bash
uv add flwr torch torchvision matplotlib numpy
```
Installs packages into `.venv/`, updates `pyproject.toml` dependencies, and writes a locked `uv.lock`. Always use `uv add` rather than `pip install` so the lockfile stays in sync.

### Sync dependencies (after editing `pyproject.toml` manually)
```bash
uv sync
```
Reads `pyproject.toml`, resolves the full dependency tree, installs everything into `.venv/`, and also builds + installs your own `src/` package so Python can import `src.fl.client_app` etc. If you see `+ test==0.1.0 (from file://...)` in the output, your package built correctly.

### Scaffold a Flower app template (reference only — already done)
```bash
flwr new @flwrlabs/quickstart-pytorch
```
Downloads the official PyTorch quickstart template into a new folder. Used as a reference for file structure and the Message API pattern. The generated `pyproject.toml` was manually merged into ours, dropping `flwr[simulation]` and `flwr-datasets` since we don't use either.

---

## Flower Config Commands

### List all SuperLink connections
```bash
flwr config list
```

**Example output:**
```
Flower Config file: /home/gigantichiha/.flwr/config.toml
SuperLink connections:
  supergrid
  local
  local-simulation (default)
  local-deployment
```

What each connection means:

| Name | Address | What it does |
|---|---|---|
| `supergrid` | `supergrid.flower.ai` | Flower's hosted cloud — not used |
| `local` | `:local:` | Auto-starts a managed local SuperLink in simulation mode |
| `local-simulation` | `.local.` (typo in your config — should be `:local:`) | Broken — ignore |
| `local-deployment` | `127.0.0.1:9093` | Your real, manually-started SuperLink — use this |

The `(default)` marker on `local-simulation` means bare `flwr run .` (with no connection name) will try to use that connection — which tries simulation, which needs Ray, which isn't installed. **Always pass `local-deployment` explicitly.**

### View the raw config file
```bash
cat ~/.flwr/config.toml
```
This file is machine-wide, lives outside your project directory, and is not committed to git. Each developer/machine gets their own.

---

## Running the Federation — Full Sequence

Open **5 separate terminal tabs**, all from `~/Desktop/learning/flower/p2`, all with the venv activated.

---

### Terminal 1 — Start the SuperLink

```bash
flower-superlink --insecure
```

**What it does:** Starts the central coordinator process. Opens three network ports simultaneously:

| Port | API | Who talks to it |
|---|---|---|
| **9091** | ServerAppIo API | The ServerApp subprocess that runs your `server_app.py` |
| **9092** | Fleet API | Each SuperNode — this is where clients register and receive work |
| **9093** | Control API | `flwr run`, `flwr list`, `flwr log`, `flwr stop` CLI commands |

**Expected output:**
```
INFO :      Starting Flower SuperLink
WARNING :   Option `--insecure` was set. Starting insecure HTTP server with
            unencrypted communication (TLS disabled). Proceed only if you
            understand the risks.
WARNING :   SuperNode authentication is disabled. The SuperLink will accept
            connections from any SuperNode.
INFO :      Flower Deployment Runtime: Starting Control API on 0.0.0.0:9093
INFO :      Flower Deployment Runtime: Starting ServerAppIo API on 0.0.0.0:9091
INFO :      Flower Deployment Runtime: Starting Fleet API (gRPC-rere) on 0.0.0.0:9092
INFO :      Starting Flower SuperExec
```

`0.0.0.0` means it's listening on all network interfaces — important later when SuperNodes on different machines need to reach it. `--insecure` disables TLS, fine for local testing. Leave this terminal running for the entire session.

---

### Terminal 2 — Start SuperNode 0 (Client 0)

```bash
flower-supernode --insecure --superlink 127.0.0.1:9092 --clientappio-api-address 127.0.0.1:9094 --node-config "partition-id=0 num-partitions=3"
```

**Flag breakdown:**

| Flag | Value | Meaning |
|---|---|---|
| `--insecure` | — | No TLS. Must match SuperLink's `--insecure` |
| `--superlink` | `127.0.0.1:9092` | Fleet API address — port 9092, NOT 9093 |
| `--clientappio-api-address` | `127.0.0.1:9094` | Local port this SuperNode opens for its own ClientApp subprocess |
| `--node-config` | `partition-id=0 num-partitions=3` | Key-value pairs read inside `client_app.py` via `context.node_config` |

**What `partition-id=0 num-partitions=3` does in your code:**
Inside `client_app.py`, this becomes:
```python
partition_id = context.node_config["partition-id"]   # → 0
num_partitions = context.node_config["num-partitions"] # → 3
trainloader, testloader = load_data(partition_id, num_partitions, batch_size)
```
Which calls `get_label_window(0, 3)` → `[0, 1, 2, 3]`. So this SuperNode's ClientApp will only train on MNIST digits 0, 1, 2, 3.

**Expected output:**
```
INFO :      Starting Flower SuperNode
WARNING :   Option `--insecure` was set. Starting insecure HTTP channel to 127.0.0.1:9092.
INFO :      Flower Deployment Runtime: Starting ClientAppIo API on 127.0.0.1:9094
```
Then it waits silently until a run is submitted. Leave this terminal running.

---

### Terminal 3 — Start SuperNode 1 (Client 1)

```bash
flower-supernode \
  --insecure \
  --superlink 127.0.0.1:9092 \
  --clientappio-api-address 127.0.0.1:9095 \
  --node-config "partition-id=1 num-partitions=3"
```

**Key differences from Terminal 2:**
- `--clientappio-api-address` is `9095` not `9094` — both SuperNodes are on the same machine so they need different local ports. On different physical machines this port can be the same (e.g. both use 9094) since they're on separate hosts.
- `partition-id=1` → `get_label_window(1, 3)` → labels `[4, 5, 6]`. This client trains only on digits 4, 5, 6.

---

### Terminal 4 — Start SuperNode 2 (Client 2)

```bash
flower-supernode \
  --insecure \
  --superlink 127.0.0.1:9092 \
  --clientappio-api-address 127.0.0.1:9096 \
  --node-config "partition-id=2 num-partitions=3"
```

- `--clientappio-api-address` is `9096`.
- `partition-id=2` → `get_label_window(2, 3)` → labels `[7, 8, 9]`. This client trains only on digits 7, 8, 9.

---

### Terminal 5 — Launch the Run

```bash
flwr run . local-deployment --stream
```

**Flag breakdown:**

| Part | Meaning |
|---|---|
| `flwr run` | Submit a Flower app run |
| `.` | Use the app in the current directory (reads `pyproject.toml`) |
| `local-deployment` | Use this named SuperLink connection from `~/.flwr/config.toml` — points at `127.0.0.1:9093` (Control API) |
| `--stream` | Print live logs to this terminal instead of just printing a run ID and returning |

**What happens when you run this:**

1. `flwr run` talks to the SuperLink's Control API (port 9093) and submits your app.
2. The SuperLink bundles your project into a FAB (Flower App Bundle) — a package containing `src/`, `pyproject.toml`, and `LICENSE`.
3. SuperExec installs the FAB into `~/.flwr/apps/` and creates an isolated runtime environment in `~/.flwr/runtime-envs/<run_id>/`.
4. SuperExec installs your app's dependencies into that runtime env via `uv sync`.
5. The ServerApp (`src/fl/server_app.py`) starts. It initializes a global `Net()`, wraps weights into an `ArrayRecord`, and starts the FedAvg strategy.
6. For each of `num-server-rounds` (default 3) rounds:
   a. Server sends current global weights to all connected SuperNodes via the Fleet API (port 9092).
   b. Each SuperNode receives the weights, spawns a ClientApp subprocess, which unpacks the `ArrayRecord` into a `state_dict`, calls `load_data()` and `train()`, repacks updated weights into a new `ArrayRecord`, and returns it.
   c. Server aggregates all client updates using FedAvg (weighted average by number of training examples).
   d. If `fraction-evaluate > 0`, server sends the updated global model to clients for evaluation. Each client calls `test()` and returns loss/accuracy.
   e. Server's `global_evaluate` function also runs the global model on the full centralized test set.
7. After all rounds, if `save-model = true`, the final model is saved to disk.

**Expected output (partial):**
```
Using SuperLink: local-deployment (127.0.0.1:9093)
Successfully started run 293193206568696263
INFO :      Starting logstream for run_id `293193206568696263`
INFO :      Start `flwr-serverapp` process
Successfully installed test to /home/.../.flwr/apps/shikhar.test.0.1.0.xxxxxxxx.
INFO :      Installing application dependencies...
INFO :      App dependencies installed successfully via uv sync.
INFO :      [ROUND 1] Starting training...
INFO :      [ROUND 1] Loss: 1.832  Accuracy: 0.421
INFO :      [ROUND 2] Starting training...
...
```

---

## Config Override Commands

### Run with more rounds (without editing `pyproject.toml`)
```bash
flwr run . local-deployment --stream --run-config "num-server-rounds=10"
```

### Run with more local epochs per round
```bash
flwr run . local-deployment --stream --run-config "local-epochs=5"
```

### Run with both overridden at once
```bash
flwr run . local-deployment --stream --run-config "num-server-rounds=10 local-epochs=5"
```

### What these knobs do and when to use each

| Config key | Default | Effect |
|---|---|---|
| `num-server-rounds` | 3 | How many full communication rounds (server → clients → server). More rounds = more federation, more communication. Start with 3 to verify correctness, increase to 10–20 to see convergence. |
| `local-epochs` | 1 | How many gradient steps each client does locally before returning weights. Increasing this makes clients diverge more before aggregation (client drift). Start with 1, try 3–5 once basic training works. |
| `learning-rate` | 0.1 | Passed to SGD optimizer in `train()`. If loss is exploding, lower this. |
| `batch-size` | 32 | DataLoader batch size. Increase if you have GPU memory to spare. |
| `fraction-evaluate` | 1.0 | Fraction of clients used for federated evaluation each round. 1.0 = all clients evaluate. |
| `save-model` | false | If true, saves the final global model to `final_model.pt` after all rounds finish. |

---

## Monitoring Commands

### List all past and current runs
```bash
flwr list local-deployment
```
Shows run IDs, statuses (running/finished/failed), and timing.

### Stream logs from a specific run
```bash
flwr log local-deployment --run-id <run_id>
```

### Stop a running run
```bash
flwr stop local-deployment --run-id <run_id>
```

---

## Port Map — Full Picture

```
┌─────────────────────────────────────────────────────────────┐
│                    Your Machine                              │
│                                                              │
│  Terminal 5         Terminal 1                               │
│  (flwr run)  ──9093──▶ SuperLink ◀──9092── SuperNode 0 (T2) │
│                         (server)  ◀──9092── SuperNode 1 (T3) │
│                                   ◀──9092── SuperNode 2 (T4) │
│                                                              │
│  SuperNode 0 opens: 9094  (its own ClientAppIo)              │
│  SuperNode 1 opens: 9095  (its own ClientAppIo)              │
│  SuperNode 2 opens: 9096  (its own ClientAppIo)              │
│                                                              │
│  SuperLink also opens: 9091 (ServerAppIo, internal use)      │
└─────────────────────────────────────────────────────────────┘
```

**When moving to different physical machines:**
- SuperLink machine: same commands, same ports.
- Each SuperNode machine: replace `127.0.0.1` with the SuperLink machine's IP address in `--superlink`. The `--clientappio-api-address` can stay as `127.0.0.1:9094` on every machine since it's only used locally.
- `flwr run` machine (yours): replace `127.0.0.1:9093` in `~/.flwr/config.toml`'s `[superlink.local-deployment]` with the SuperLink machine's IP.
- Drop `--insecure` and set up TLS once running over a real network.

---

## Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| `command not found: flower-superlink` | venv not activated | Run `source .venv/bin/activate` or prefix with `uv run` |
| `ModuleNotFoundError: No module named 'pytorchexample'` | server_app.py / client_app.py still have template import paths | Change `from pytorchexample.task import ...` to `from src.fl.tasks.classification import ...` |
| `ImportError: cannot import name 'load_centralized_dataset'` | Wrong function name from template | Change to `load_centralized_testset` (the name in your actual file) |
| `Property "publisher" missing in [tool.flwr.app]` | `pyproject.toml` missing `[tool.flwr.app]` table | Add `[tool.flwr.app]` with `publisher = "shikhar"` |
| `fab-format-version = 1 requires license` | Template's `fab-format-version = 1` requires a `LICENSE` file | Either `echo "MIT License" > LICENSE` and add `license = { file = "LICENSE" }` to `[project]`, or remove `fab-format-version = 1` entirely |
| `Simulation raised an exception: ray is not available` | `flwr run .` defaulted to simulation runtime | Always run `flwr run . local-deployment --stream`, not bare `flwr run .` |
| SuperNode retrying connection repeatedly | SuperLink not started yet, or wrong port | Start `flower-superlink --insecure` first; SuperNodes connect to port **9092**, not 9093 |
