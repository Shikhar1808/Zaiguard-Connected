"""classification_embeddings: FL Task for Zaiguard Human-in-the-Loop Alert System."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class AlertImportanceClassifier(nn.Module):
    """
    Binary Classifier that takes a 138-d embedding (appearance, spatial, temporal)
    and predicts if the alert is Important (1) or Normal (0).
    """
    def __init__(self, input_dim=138):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def load_embedding_data(partition_id: int, num_partitions: int, batch_size: int, num_samples: int = 1000):
    """
    Simulates loading the local SQLite database of observer-labeled embeddings.
    Since we don't have the real Zaiguard database here, we generate synthetic data.
    """
    # Set seed so each partition (client) gets different but consistent data
    torch.manual_seed(partition_id + 42)
    
    # Simulate labels: 20% Important (1), 80% Normal (0)
    labels = torch.randint(0, 100, (num_samples, 1)).float()
    labels = (labels < 20).float()
    
    # Create fake embeddings (random normal distribution)
    embeddings = torch.randn(num_samples, 138)
    
    # Add a slight pattern so the model can actually "learn" something in the demo
    # If it's an important alert, shift the embedding values slightly
    embeddings += labels * 0.5 
    
    # Split into train/val (80/20)
    train_size = int(0.8 * num_samples)
    
    train_dataset = TensorDataset(embeddings[:train_size], labels[:train_size])
    val_dataset = TensorDataset(embeddings[train_size:], labels[train_size:])
    
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=batch_size)
    
    return trainloader, valloader


def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    criterion = nn.BCELoss() # Binary Cross Entropy for True/False
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    
    net.train()
    total_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            embeddings, labels = batch[0].to(device), batch[1].to(device)
            
            optimizer.zero_grad()
            outputs = net(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
    return total_loss / (len(trainloader) * epochs)


def test(net, testloader, device):
    """Evaluate the model on the test set."""
    criterion = nn.BCELoss()
    correct, loss = 0, 0.0
    
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            embeddings, labels = batch[0].to(device), batch[1].to(device)
            outputs = net(embeddings)
            loss += criterion(outputs, labels).item()
            
            # Predict 1 if probability > 0.5, else 0
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            
    accuracy = correct / len(testloader.dataset)
    return loss / len(testloader), accuracy
