"""src.fl.tasks.classification_cifar10

Model, training, evaluation, and non-IID data loading for CIFAR-10 image
classification. Structured the same way as classification.py (MNIST) so
client_app.py / server_app.py can switch between them by changing one
import line. CIFAR-10 has 3-channel 32x32 color images across 10 classes
(airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck),
which is why the model's first conv layer and FC input size differ from
the MNIST version.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

DATA_DIR = "./data"

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Same non-IID label-window scheme as the MNIST task module. cid is
# 1-indexed to match the original scheme; partition-id from Flower's
# node-config is 0-indexed, so callers are responsible for the +1 shift
# (see load_data below). Label indices here refer to CIFAR-10 class
# indices (0=airplane ... 9=truck), not digits.
LABEL_MAP_3_CLIENTS = {
    1: [0, 1, 2, 3],
    2: [4, 5, 6],
    3: [7, 8, 9],
}

LABEL_MAP_8_CLIENTS = {
    1: [0, 1, 2, 3],
    2: [1, 2, 3, 4],
    3: [2, 3, 4, 5],
    4: [3, 4, 5, 6],
    5: [5, 6, 7, 8],
    6: [6, 7, 8, 9],
    7: [7, 8, 9, 0],
    8: [8, 9, 0, 1],
}


class Net(nn.Module):
    """Simple CNN for CIFAR-10 (3-channel, 32x32 input)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def get_label_window(partition_id: int, num_partitions: int) -> list[int]:
    """Return the list of CIFAR-10 class indices this client should see.

    partition_id is 0-indexed (as Flower's node-config provides it).
    Internally we shift to the 1-indexed cid scheme used by the label maps.
    """
    cid = partition_id + 1

    if num_partitions == 3:
        return LABEL_MAP_3_CLIENTS.get(cid, list(range(10)))
    if num_partitions == 8:
        return LABEL_MAP_8_CLIENTS.get(cid, list(range(10)))

    # Fallback for any other client count: give everyone everything.
    return list(range(10))


# Cache the raw CIFAR-10 datasets so they're only loaded/scanned from disk
# once per process, no matter how many times load_data() is called across
# rounds.
_train_dataset = None
_test_dataset = None

# Cache each partition's row indices too -- the label window for a given
# (partition_id, num_partitions) pair never changes, so there is no reason
# to rescan all 50,000 training labels on every call.
_partition_indices_cache: dict[tuple[int, int], list[int]] = {}


def _get_raw_datasets():
    """Load (and cache) the full CIFAR-10 train/test datasets."""
    global _train_dataset, _test_dataset
    if _train_dataset is None or _test_dataset is None:
        # Standard CIFAR-10 per-channel mean/std (RGB).
        transform = Compose(
            [ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]
        )
        _train_dataset = CIFAR10(DATA_DIR, train=True, download=True, transform=transform)
        _test_dataset = CIFAR10(DATA_DIR, train=False, download=True, transform=transform)
    return _train_dataset, _test_dataset


def _get_partition_indices(partition_id: int, num_partitions: int, trainset) -> list[int]:
    """Return (and cache) the training-set row indices for this client's
    non-IID label window."""
    cache_key = (partition_id, num_partitions)
    if cache_key not in _partition_indices_cache:
        labels_to_keep = get_label_window(partition_id, num_partitions)
        # trainset.targets is a plain Python list for CIFAR10 (unlike
        # MNIST, where .targets is a tensor) -- enumerate works the same
        # way either way.
        _partition_indices_cache[cache_key] = [
            i for i, label in enumerate(trainset.targets) if label in labels_to_keep
        ]
    return _partition_indices_cache[cache_key]


def load_data(partition_id: int, num_partitions: int, batch_size: int = 32):
    """Build a non-IID train/test split for one client.

    Each client's trainloader only contains the CIFAR-10 classes in its
    label window. The testloader is always the FULL CIFAR-10 test set, so
    you can see how well a client's locally biased model generalizes.
    """
    trainset, testset = _get_raw_datasets()

    indices = _get_partition_indices(partition_id, num_partitions, trainset)
    train_subset = Subset(trainset, indices)

    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size)
    return trainloader, testloader


def load_centralized_testset(batch_size: int = 128) -> DataLoader:
    """Full CIFAR-10 test set, used by the server for centralized
    evaluation after each round. Reuses the same cached dataset as
    load_data() rather than rebuilding it.
    """
    _, testset = _get_raw_datasets()
    return DataLoader(testset, batch_size=batch_size)


def train(net: nn.Module, trainloader: DataLoader, epochs: int, lr: float, device) -> float:
    """Train for `epochs` local epochs. Returns average training loss."""
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()

    running_loss = 0.0
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    return running_loss / (epochs * len(trainloader))


def test(net: nn.Module, testloader: DataLoader, device) -> tuple[float, float]:
    """Evaluate on testloader. Returns (avg_loss, accuracy)."""
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if total == 0:
        return 0.0, 0.0
    return loss / len(testloader), correct / total