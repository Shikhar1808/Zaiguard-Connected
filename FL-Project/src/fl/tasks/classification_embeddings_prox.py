"""classification_embeddings_prox: FL Task for Zaiguard with FedProx and Focal Loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification. 
    Penalizes the model heavily when it gets the rare 'Important' (1) class wrong,
    and reduces the loss for the common 'Normal' (0) class once it's learned well.
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # inputs are probabilities (from Sigmoid)
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class AlertImportanceClassifier(nn.Module):
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


def load_imbalanced_embedding_data(partition_id: int, num_partitions: int, batch_size: int, num_samples: int = 1000):
    """
    Simulates loading highly imbalanced data (Non-IID).
    e.g., 99% Normal alerts (0), 1% Important alerts (1).
    """
    torch.manual_seed(partition_id + 42)
    
    # Simulate extreme imbalance: 1% Important (1), 99% Normal (0)
    labels = torch.randint(0, 1000, (num_samples, 1)).float()
    labels = (labels < 10).float() # Only <10 out of 1000 will be 1
    
    embeddings = torch.randn(num_samples, 138)
    
    # Add a slight pattern for the rare Important class
    embeddings += labels * 0.5 
    
    train_size = int(0.8 * num_samples)
    
    train_dataset = TensorDataset(embeddings[:train_size], labels[:train_size])
    val_dataset = TensorDataset(embeddings[train_size:], labels[train_size:])
    
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=batch_size)
    
    return trainloader, valloader


def train(net, trainloader, epochs, lr, device):
    """Train the model using Focal Loss."""
    criterion = FocalLoss()
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
    """Evaluate the model using Focal Loss."""
    criterion = FocalLoss()
    correct, loss = 0, 0.0
    
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            embeddings, labels = batch[0].to(device), batch[1].to(device)
            outputs = net(embeddings)
            loss += criterion(outputs, labels).item()
            
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            
    accuracy = correct / len(testloader.dataset)
    return loss / len(testloader), accuracy
