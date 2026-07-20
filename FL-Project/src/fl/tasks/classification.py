"""src.fl.tasks.classification

Model, training, evaluation, and non-IID data loading for MNIST digit
classification. This is one task module among possibly several (a future
detection.py would live alongside this one in src/fl/tasks/). client_app.py
and server_app.py import from here; they should not contain any
classification-specific logic themselves.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor

DATA_DIR = "./data"

# Same non-IID label windows you had before. cid is 1-indexed to match
# the original scheme; partition-id from Flower's node-config is 0-indexed,
# so callers are responsible for the +1 shift (see load_data below).
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
    """Simple CNN used as the common FL model for MNIST classification."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def get_label_window(partition_id: int, num_partitions: int) -> list[int]:
    """Return the list of MNIST digit labels this client should see.

    partition_id is 0-indexed (as Flower's node-config provides it).
    Internally we shift to the 1-indexed cid scheme used by the label maps.
    """
    cid = partition_id + 1

    if num_partitions == 3:
        return LABEL_MAP_3_CLIENTS.get(cid, list(range(10)))
    if num_partitions == 8:
        return LABEL_MAP_8_CLIENTS.get(cid, list(range(10)))

    # Fallback for any other client count: give everyone everything.
    # Extend this with your own scheme if you need non-IID splits for
    # other client counts.
    return list(range(10))


def load_data(partition_id: int, num_partitions: int, batch_size: int = 32):
    """Build a non-IID train/test split for one client.

    Each client's trainloader only contains the digits in its label
    window. The testloader is always the FULL MNIST test set, so you can
    see how well a client's locally biased model generalizes -- this
    matches what your original load_data did.
    """
    transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    trainset = MNIST(DATA_DIR, train=True, download=True, transform=transform)
    testset = MNIST(DATA_DIR, train=False, download=True, transform=transform)

    labels_to_keep = get_label_window(partition_id, num_partitions)

    indices = [i for i, label in enumerate(trainset.targets) if label in labels_to_keep]
    train_subset = Subset(trainset, indices)

    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size)
    return trainloader, testloader


def load_centralized_testset(batch_size: int = 128) -> DataLoader:
    """Full MNIST test set, used by the server for centralized evaluation
    after each round (independent of any single client's local test split).
    """
    transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    testset = MNIST(DATA_DIR, train=False, download=True, transform=transform)
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