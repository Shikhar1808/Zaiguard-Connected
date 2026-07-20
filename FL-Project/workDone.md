# Federated Learning Project — Work Done So Far

## Project Overview

A federated learning system built with **Flower (flwr) v1.32.0** and **PyTorch**, using the **Deployment Runtime** (real SuperLink + real SuperNodes in separate terminals) rather than the Simulation Runtime. Designed to run locally across multiple terminals first, then move to separate physical machines with zero code changes.

Two datasets are supported: **MNIST** (digit classification) and **CIFAR-10** (image classification). A YOLO-style object detection pipeline is planned as a future third task and has been structurally accounted for in the folder layout.

---

## Project Structure

```
p2/
├── pyproject.toml                        # Project metadata, dependencies, Flower config
├── README.md
├── .python-version                       # Pins Python 3.12
├── .gitignore                            # Excludes .venv, __pycache__, data/
├── uv.lock                               # Locked dependency tree managed by uv
├── LICENSE                               # Required by Flower's FAB format
├── data/                                 # MNIST / CIFAR-10 downloaded here at runtime
└── src/
    ├── __init__.py
    └── fl/
        ├── __init__.py
        ├── client_app.py                 # Flower ClientApp — FL glue, calls into tasks/
        ├── server_app.py                 # Flower ServerApp — FedAvg strategy, round loop
        └── tasks/
            ├── __init__.py
            ├── classification.py         # MNIST task — model, train, test, data loading
            └── classification_cifar10.py # CIFAR-10 task — same interface, different model
```

The `tasks/` subfolder is the intentional seam for future expansion. When object detection is added, it will live at `src/fl/tasks/detection.py` alongside the two classification modules. Switching `client_app.py` and `server_app.py` between tasks is one import line change.

---

## Dependencies (`pyproject.toml`)

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "test"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "flwr>=1.32.0",
    "torch>=2.12.1",
    "torchvision>=0.27.1",
    "matplotlib>=3.11.0",
    "numpy>=2.5.0",
]

[tool.hatch.build.targets.wheel]
packages = ["src"]

[tool.flwr.app]
publisher = "shikhar"

[tool.flwr.app.components]
serverapp = "src.fl.server_app:app"
clientapp = "src.fl.client_app:app"

[tool.flwr.app.config]
num-server-rounds = 3
fraction-evaluate = 1.0
local-epochs = 1
learning-rate = 0.1
batch-size = 32
save-model = false
```

**Key decisions:**
- `flwr` only — no `flwr[simulation]` extra. Ray is not installed. The simulation runtime is intentionally excluded since we run real SuperNodes in separate terminals.
- `flwr-datasets` is not used. Data loading is handled entirely by our own non-IID partition logic using `torchvision.datasets` directly.
- `hatchling` is the build backend (not uv's default) because Flower's FAB bundler requires the project to be a proper installable package that `flwr run` can import.

---

## Flower API Version: Message API (not legacy NumPyClient)

This project uses the **current Flower Message API** introduced in Flower 1.20+, not the older `NumPyClient.fit()` / `NumPyClient.evaluate()` style seen in older tutorials. Key differences:

| Legacy API | Current Message API (this project) |
|---|---|
| Subclass `NumPyClient` | Decorate functions with `@app.train()`, `@app.evaluate()` |
| `fit(parameters, config)` | `train(msg: Message, context: Context)` |
| Returns list of numpy arrays | Returns `Message` containing `ArrayRecord` |
| `get_parameters` / `set_parameters` helpers | `ArrayRecord(state_dict)` and `.to_torch_state_dict()` built-in |

Model weights travel as `ArrayRecord` objects inside `Message` envelopes. The conversion to/from PyTorch `state_dict` is done via `ArrayRecord(model.state_dict())` (pack) and `msg.content["arrays"].to_torch_state_dict()` (unpack).

---

## Task Modules

### `src/fl/tasks/classification.py` — MNIST

**Dataset:** MNIST, 60,000 training / 10,000 test, grayscale 28×28, 10 digit classes (0–9).

**Model (`Net`):** Simple CNN.
- `conv1`: 1 → 6 channels, 5×5 kernel
- `conv2`: 6 → 16 channels, 5×5 kernel
- `MaxPool2d(2, 2)` after each conv
- FC layers: 256 → 120 → 84 → 10
- Output: 10-class logits

**Non-IID data split:** Each client only sees a subset of digit labels, determined by its `partition_id`. Two schemes are implemented:

| partition_id | Labels seen (3 clients) |
|---|---|
| 0 | 0, 1, 2, 3 |
| 1 | 4, 5, 6 |
| 2 | 7, 8, 9 |

| partition_id | Labels seen (8 clients) |
|---|---|
| 0 | 0, 1, 2, 3 |
| 1 | 1, 2, 3, 4 |
| 2 | 2, 3, 4, 5 |
| 3 | 3, 4, 5, 6 |
| 4 | 5, 6, 7, 8 |
| 5 | 6, 7, 8, 9 |
| 6 | 7, 8, 9, 0 |
| 7 | 8, 9, 0, 1 |

The 8-client scheme uses overlapping label windows by design — each client sees 4 labels with a 1-label overlap with its neighbours, simulating realistic non-IID heterogeneity.

**Normalization:** `mean=(0.1307,)`, `std=(0.3081,)` — standard MNIST single-channel values.

**Caching:** `_train_dataset`, `_test_dataset`, and `_partition_indices_cache` are module-level globals. MNIST is loaded from disk once per process; the 60,000-label index scan runs once per `(partition_id, num_partitions)` pair. This avoids redundant disk I/O and scanning across FL rounds.

**Public functions:**

| Function | Returns | Notes |
|---|---|---|
| `get_label_window(partition_id, num_partitions)` | `list[int]` | Which labels this client sees |
| `load_data(partition_id, num_partitions, batch_size)` | `(trainloader, testloader)` | trainloader is non-IID subset; testloader is FULL test set |
| `load_centralized_testset(batch_size)` | `DataLoader` | Used by server for round-by-round global evaluation |
| `train(net, trainloader, epochs, lr, device)` | `float` (avg loss) | SGD with momentum 0.9 |
| `test(net, testloader, device)` | `(float, float)` (loss, accuracy) | Runs in `torch.no_grad()` |

---

### `src/fl/tasks/classification_cifar10.py` — CIFAR-10

**Dataset:** CIFAR-10, 50,000 training / 10,000 test, RGB 32×32, 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).

**Model (`Net`):** Same CNN architecture as MNIST version, resized for 3-channel 32×32 input.
- `conv1`: **3** → 6 channels, 5×5 kernel (3 not 1 — RGB not grayscale)
- `conv2`: 6 → 16 channels, 5×5 kernel
- FC layers: **400** → 120 → 84 → 10 (400 = 16 × 5 × 5, not 16 × 4 × 4 like MNIST)
- Output: 10-class logits

**Non-IID split:** Identical label-window scheme to MNIST. Label indices 0–9 now refer to CIFAR-10 classes instead of digits, but the partition logic is unchanged.

**Normalization:** `mean=(0.4914, 0.4822, 0.4465)`, `std=(0.2470, 0.2435, 0.2616)` — standard CIFAR-10 per-channel RGB values.

**Key difference from MNIST:** `CIFAR10.targets` is a plain Python `list` of `int`, unlike `MNIST.targets` which is a `torch.Tensor`. The filtering logic (`enumerate(trainset.targets)`) handles both without any code change.

**Same caching pattern, same public function signatures** as `classification.py` — intentional so `client_app.py` can import either module identically.

---

## Flower Runtime Components

### SuperLink
The central coordinator. Runs one instance for the whole federation. Exposes three APIs on separate ports:

| API | Port | Used by |
|---|---|---|
| Fleet API | 9092 | SuperNodes — this is where clients connect |
| ServerAppIo API | 9091 | The ServerApp process |
| Control API | 9093 | `flwr run`, `flwr list`, `flwr log` CLI commands |

### SuperNode
One per client. Each SuperNode connects to the SuperLink's Fleet API (port 9092), receives work, spawns a `ClientApp` subprocess, runs local training/evaluation, and returns results. Each SuperNode also exposes its own `ClientAppIo` API locally so the ClientApp subprocess can communicate with it.

### SuperExec
Automatically spawned as a subprocess by the SuperLink (and by each SuperNode) in default subprocess isolation mode. Responsible for installing app dependencies, loading the FAB, and launching ServerApp / ClientApp processes. Not started manually.

---

## Flower Configuration

Stored at `~/.flwr/config.toml` (machine-wide, not per-project). Named connection profiles:

```toml
[superlink]
default = "local-simulation"          # ← default, but we DON'T use this

[superlink.local-deployment]
address = "127.0.0.1:9093"            # Control API port of the real SuperLink
insecure = true                        # No TLS — local testing only
```

The `local-simulation` default uses `address = ":local:"` which auto-starts a Simulation Runtime. Since we explicitly use `flwr run . local-deployment`, we always target the real, manually-started SuperLink instead.

---

## What's NOT Done Yet

- `client_app.py` and `server_app.py` — written (you have them) but import names need to match the `classification.py` function names exactly (`load_centralized_testset`, not `load_centralized_dataset`).
- Object detection task (`src/fl/tasks/detection.py`) — folder structure is ready, implementation not started.
- TLS / SuperNode authentication — currently running `--insecure` for local development. Needed before moving to different physical machines over a real network.
- Results logging — metrics are printed to terminal. No CSV/JSON export or matplotlib plots yet.
- Model poisoning experiments — intentionally deferred. The `poison_dataset` function from the original code was stripped. Will be added back as a separate, clearly-flagged experiment module.
