import torch
import torch.nn.functional as F
from torchvision.datasets import MNIST, ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

from sbnn import BayesianNet

# ---------------------------------------------
# Helper: compute entropy of softmax predictions
# ---------------------------------------------
def compute_entropy(probs):
    return -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

# ---------------------------------------------
# Load MNIST
# ---------------------------------------------
def load_mnist(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten
    ])

    train_data = MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = MNIST(root='./data', train=False, download=True, transform=transform)

    return DataLoader(train_data, batch_size=batch_size, shuffle=True), DataLoader(test_data, batch_size=batch_size, shuffle=False)

# ---------------------------------------------
# Load notMNIST (for OOD detection)
# ---------------------------------------------
def load_notmnist(batch_size=128):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    # Ensure notMNIST dataset exists locally
    data_dir = "./data/notMNIST"
    if not os.path.exists(data_dir):
        raise RuntimeError("Download notMNIST_small manually and unzip it under ./data/notMNIST_small")

    dataset = ImageFolder(root=data_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

# ---------------------------------------------
# Train Model
# ---------------------------------------------
def train_model(model, train_loader, epochs=5, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for x, y in train_loader:
            optimizer.zero_grad()
            preds, _, _ = model(x)

            loss = F.cross_entropy(preds, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y)
            total_correct += (preds.argmax(1) == y).sum().item()
            total_samples += len(y)

        acc = total_correct / total_samples
        print(f"Epoch {epoch}: Loss={total_loss/total_samples:.4f}, Acc={acc:.2%}")

# ---------------------------------------------
# Evaluate Model
# ---------------------------------------------
def evaluate_uncertainty(model, loader, label="MNIST"):
    model.eval()
    all_preds = []
    all_targets = []
    all_entropies = []

    with torch.no_grad():
        for x, y in loader:
            probs, _, _ = model(x)
            entropy = compute_entropy(probs)

            all_preds.append(probs)
            all_targets.append(y)
            all_entropies.append(entropy)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_entropies = torch.cat(all_entropies)

    acc = (all_preds.argmax(1) == all_targets).float().mean().item()
    print(f"[{label}] Accuracy: {acc:.4f}")

    return all_entropies.numpy()

# ---------------------------------------------
# Main Experiment
# ---------------------------------------------
def main():
    train_loader, test_loader = load_mnist()
    notmnist_loader = load_notmnist()

    model = BayesianNet(input_dim=784, hidden_dim=256, output_dim=10)

    print("Training on MNIST...")
    train_model(model, train_loader, epochs=10)

    print("\nEvaluating on MNIST test set...")
    entropy_mnist = evaluate_uncertainty(model, test_loader, label="MNIST")

    print("\nEvaluating on notMNIST (OOD)...")
    entropy_notmnist = evaluate_uncertainty(model, notmnist_loader, label="notMNIST")

    # Plot histogram of entropies
    plt.hist(entropy_mnist, bins=50, alpha=0.6, label="MNIST")
    plt.hist(entropy_notmnist, bins=50, alpha=0.6, label="notMNIST")
    plt.title("Predictive Entropy Distribution")
    plt.xlabel("Entropy")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
