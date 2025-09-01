# Bayesian Bernoulli Neural Network with Beta-distributed Probabilities
# ---------------------------------------------------------------
# From scratch implementation with reparameterized Beta using Kumaraswamy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Uniform
from sl import MultinomialOpinion

# digamma is available as torch.special.digamma (newer) or torch.digamma (older)
try:
    from torch.special import digamma
except ImportError:
    from torch import digamma  # fallback

# betaln(a, b) = lgamma(a) + lgamma(b) - lgamma(a + b)
# use torch.special.gammaln if present; else fallback to torch.lgamma
try:
    from torch.special import gammaln as _gammaln
except Exception:
    _gammaln = torch.lgamma

def betaln(a, b):
    return _gammaln(a) + _gammaln(b) - _gammaln(a + b)

# ---------------------------------------------------------------
# Kumaraswamy Sampler: Approximate Beta(alpha, beta)
# ---------------------------------------------------------------
def sample_kumaraswamy(alpha, beta, eps=1e-7):
    u = torch.rand_like(alpha).clamp(eps, 1 - eps)
    return (1 - u.pow(1 / beta)).pow(1 / alpha)

# ---------------------------------------------------------------
# KL divergence between two Beta distributions
# ---------------------------------------------------------------
def kl_beta(alpha_q, beta_q, alpha_p=1.0, beta_p=1.0):
    term1 = betaln(torch.tensor(alpha_p), torch.tensor(beta_p)) - betaln(alpha_q, beta_q)
    term2 = (alpha_q - alpha_p) * digamma(alpha_q)
    term3 = (beta_q - beta_p) * digamma(beta_q)
    term4 = (alpha_p + beta_p - alpha_q - beta_q) * digamma(alpha_q + beta_q)
    return term1 + term2 + term3 + term4

# ---------------------------------------------------------------
# Concrete / Gumbel-Sigmoid Relaxation for Bernoulli Sampling
# ---------------------------------------------------------------
def sample_concrete(p, temperature):
    u = torch.rand_like(p)
    gumbel = -torch.log(-torch.log(u + 1e-7) + 1e-7)
    logit = torch.log(p + 1e-7) - torch.log(1 - p + 1e-7)
    return torch.sigmoid((logit + gumbel) / temperature)

# ---------------------------------------------------------------
# Simple Bayesian Linear Layer
# ---------------------------------------------------------------
class BayesianBinaryLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Parameters of the Beta distribution for each weight (encourage initial uncertainty)
        self.alpha = nn.Parameter(torch.full((out_features, in_features), -1.0))
        self.beta = nn.Parameter(torch.full((out_features, in_features), -1.0))

        # Parameters for weights (magnitude) with larger init
        self.weight_value = nn.Parameter(torch.randn(out_features, in_features) * 0.5)

        # Slightly randomized biases
        self.bias = nn.Parameter(torch.randn(out_features) * 0.1)

    def forward(self, x, temperature=0.05):
        # Sample probabilities from Beta approx (Kumaraswamy)
        p = sample_kumaraswamy(F.softplus(self.alpha), F.softplus(self.beta))

        # Sample relaxed binary mask
        mask = sample_concrete(p, temperature)

        # Effective weight: binary mask * weight value
        effective_weight = mask * self.weight_value

        return F.linear(x, effective_weight, self.bias), p, mask

    def kl_loss(self):
        return kl_beta(F.softplus(self.alpha), F.softplus(self.beta)).sum()
    
    

# ---------------------------------------------------------------
# Example Network with 2 Bayesian Linear Layers
# ---------------------------------------------------------------
class BayesianNet(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=4, output_dim=1):
        super().__init__()
        self.layer1 = BayesianBinaryLinear(input_dim, hidden_dim)
        self.layer2 = BayesianBinaryLinear(hidden_dim, output_dim)
        self.multiclass = output_dim>1 

    def forward(self, x, temperature=0.05):
        h, p1, m1 = self.layer1(x, temperature)
        h = F.relu(h)
        out, p2, m2 = self.layer2(h, temperature)
        if out.shape[1] == 1:
            out = torch.sigmoid(out)
        else:
            out = F.softmax(out, dim=1)
        return out, (p1, p2), (m1, m2)
    
    def kl(self):
        return self.layer1.kl_loss() + self.layer2.kl_loss()

    def show_model(self):
        print("BayesianNet Architecture")
        print("-------------------------")
        print(self)
        print("\nParameters and Values:")
        for name, param in self.named_parameters():
            print(f"{name} (shape {param.shape}):\n{param.data}\n")
        print("\nSoftplus(alpha) layer1:\n", F.softplus(self.layer1.alpha).detach())
        print("Softplus(beta) layer1:\n", F.softplus(self.layer1.beta).detach())
        print("Softplus(alpha) layer2:\n", F.softplus(self.layer2.alpha).detach())
        print("Softplus(beta) layer2:\n", F.softplus(self.layer2.beta).detach())

    def predict_multiple(model, x, n_samples=100, temperature=0.05):
        model.eval()
        preds = []

        with torch.no_grad():
            for _ in range(n_samples):
                y_hat, _, _ = model(x, temperature)
                preds.append(y_hat)

        preds = torch.stack(preds)  # shape: [n_samples, batch_size, 1]
        mean = preds.mean(dim=0)
        std = preds.std(dim=0)
        
        return mean, std
    
    def predict_nested_samples_with_raw(model, x, n_p=10, n_mask=10, temperature=0.05):
        """
        For each input in the batch, return:
        - averaged probabilities per p (shape [n_p])
        - full list of raw probabilities (shape [n_p * n_mask])

        Returns:
            List of length batch_size, where each element is a tuple:
            (avg_probs: Tensor[n_p], all_preds: Tensor[n_p * n_mask])
        """
        model.eval()
        batch_size = x.size(0)

        # Will contain one (avg_probs, all_preds) per sample
        all_results = []

        # Initialize per-sample containers
        for _ in range(batch_size):
            all_results.append(([], []))  # (avg_probs_list, raw_preds_list)

        with torch.no_grad():
            for _ in range(n_p):
                # Sample once from Beta (p)
                p1 = sample_kumaraswamy(F.softplus(model.layer1.alpha), F.softplus(model.layer1.beta))
                p2 = sample_kumaraswamy(F.softplus(model.layer2.alpha), F.softplus(model.layer2.beta))

                preds_mask = []

                for _ in range(n_mask):
                    # Sample masks from fixed p
                    mask1 = sample_concrete(p1, temperature)
                    mask2 = sample_concrete(p2, temperature)

                    # Build effective weights
                    w1 = mask1 * model.layer1.weight_value
                    w2 = mask2 * model.layer2.weight_value

                    # Forward pass
                    h = F.linear(x, w1, model.layer1.bias)
                    h = F.relu(h)
                    out = F.linear(h, w2, model.layer2.bias)
                    if out.shape[1] == 1:
                        y_hat = torch.sigmoid(out).squeeze(1)  # Binary case
                    else:
                        y_hat = F.softmax(out, dim=1)          # Multiclass case

                    preds_mask.append(y_hat)

                # shape: [n_mask, batch]
                preds_mask = torch.stack(preds_mask)

                # Average over masks: shape [batch]
                avg_pred = preds_mask.mean(dim=0)

                for i in range(batch_size):
                    avg_list, raw_list = all_results[i]
                    avg_list.append(avg_pred[i])  # Store full vector
                    raw_list.extend(preds_mask[:, i].tolist())
        # Convert to tensors
        final_output = []
        for avg_list, raw_list in all_results:
            avg_tensor = torch.stack(avg_list)       # shape [n_p, num_classes]
            raw_tensor = torch.tensor(raw_list)      # shape [n_p * n_mask, num_classes]
            final_output.append((avg_tensor, raw_tensor))

        return final_output



# ---------------------------------------------------------------
# Training Loop Skeleton
# ---------------------------------------------------------------
def train(model, data, targets, epochs=1000, lr=1e-2):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        preds, ps, masks = model(data, temperature=0.05)
        if preds.shape[1] == 1:
            # Binary classification
            likelihood = F.binary_cross_entropy(preds, targets)
        else:
            # Multiclass classification
            if targets.ndim == 2:
                if targets.shape[1] == 1:
                    targets = targets.squeeze(1)
                else:
                    targets = targets.argmax(dim=1)

            targets = targets.long()  
            likelihood = F.cross_entropy(preds, targets)


        # likelihood = F.binary_cross_entropy(preds, targets)
        kl = model.kl()

        # ELBO = likelihood + KL
        loss = likelihood + kl * 1e-3
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            with torch.no_grad():
                if(model.multiclass):
                    # Make sure targets are class indices, not one-hot
                    if targets.ndim == 2 and targets.shape[1] > 1:
                        targets = targets.argmax(dim=1)

                    # Get predicted class labels from softmax output
                    pred_labels = preds.argmax(dim=1)
                else:
                    pred_labels = (preds > 0.5).float()
                acc_train = (pred_labels == targets).float().mean().item()
            print(f"Epoch {epoch}: Loss={loss:.4f}, Acc={acc_train:.2%}, KL={kl:.4f}, CE={likelihood:.4f}")
# ---------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------
            
def test_xor():
    # Toy data: binary classification
    torch.manual_seed(42)
    x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], dtype=torch.float32)
    y = torch.tensor([[0.], [1.], [1.], [0.]], dtype=torch.float32)  # XOR task

    model = BayesianNet()
    # model.show_model()
    train(model, x, y)
    # model.show_model()

    print(model.predict_multiple(torch.tensor([[0., 0.]])))
    print(model.predict_multiple(torch.tensor([[0., 1.]])))
    print(model.predict_multiple(torch.tensor([[1., 0.]])))
    print(model.predict_multiple(torch.tensor([[1., 1.]])))

def test_make_moons():
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    # Generate dataset
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = BayesianNet()
    model.show_model()
    train(model, X_train, y_train, epochs=1000)

    mean_preds, std_preds = model.predict_multiple(X_test, n_samples=100)

    # Binarize predictions (threshold at 0.5)
    y_pred = (mean_preds > 0.5).float()

    # Accuracy
    acc = (y_pred == y_test).float().mean()
    print(f"Test Accuracy: {acc.item():.4f}")

    # std_preds: uncertainty over multiple samples

    if std_preds.ndim == 2 and std_preds.shape[1] > 1:
        # Multiclass: use std across classes for each sample
        uncertainty = std_preds.std(dim=1)  # shape: [N]
    else:
        # Binary: already shape [N]
        uncertainty = std_preds.squeeze()  # or leave as is

    # Apply threshold to filter confident predictions
    conf_threshold = 0.1
    confident_mask = (uncertainty < conf_threshold)

    # Filter to confident only
    y_confident = y_test[confident_mask]
    y_pred_confident = y_pred[confident_mask]

    if len(y_confident) > 0:
        acc_confident = (y_confident == y_pred_confident).float().mean()
        coverage = len(y_confident) / len(y_test)
        print(len(y_test))
        print(f"Accuracy on confident predictions: {acc_confident:.4f}")
        print(f"Coverage (fraction of confident samples): {coverage:.2%}")
    else:
        print("No predictions met the confidence threshold.")
    # Optional: visualize uncertainty
    plt.scatter(X_test[:, 0], X_test[:, 1], c=std_preds.squeeze(), cmap='viridis', s=40)
    plt.colorbar(label='Predictive Uncertainty (std)')
    plt.title('Uncertainty over Test Data')
    plt.show()

def test_with_dataset(X, y, title="", hidden_dim=4, epoch=1000):
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = BayesianNet(input_dim=X.shape[1], hidden_dim=hidden_dim)
    train(model, X_train, y_train, epochs=epoch)

    mean_preds, std_preds = model.predict_multiple(X_test, n_samples=100)
    y_pred = (mean_preds > 0.5).float()
    acc = (y_pred == y_test).float().mean()
    print(f"[{title}] Test Accuracy: {acc.item():.4f}")

    conf_threshold = 0.1
    confident_mask = (std_preds.squeeze() < conf_threshold)
    y_confident = y_test[confident_mask]
    y_pred_confident = y_pred[confident_mask]

    if len(y_confident) > 0:
        acc_confident = (y_confident == y_pred_confident).float().mean()
        coverage = len(y_confident) / len(y_test)
        print(f"[{title}] Accuracy (confident): {acc_confident:.4f} | Coverage: {coverage:.2%}")
    else:
        print(f"[{title}] No confident predictions.")

    # Plot (only works for 2D inputs)
    if X_test.shape[1] == 2:
        plt.scatter(X_test[:, 0], X_test[:, 1], c=std_preds.squeeze(), cmap='viridis', s=40)
        plt.colorbar(label='Uncertainty (std)')
        plt.title(f'Uncertainty on {title}')
        plt.show()
def test_breast_cancer():
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X, y = data.data, data.target
    test_with_dataset(X, y, title="Breast Cancer", hidden_dim = 18, epoch=10000)

def test_mnist():
    from torchvision.datasets import MNIST
    from torchvision import transforms
    from torch.utils.data import DataLoader, random_split
    import matplotlib.pyplot as plt

    # Binary MNIST (e.g., "3 vs 8")
    # classes = (3, 8)
    classes = list(range(10))

    # Load and filter dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    dataset = MNIST(root='./data', train=True, download=True, transform=transform)

    # Filter for chosen classes
    data = []
    targets = []
    for x, y in dataset:
        if y in classes:
            data.append(x)
            targets.append(y)
    X = torch.stack(data)
    y = torch.tensor(targets).float().view(-1, 1)

    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Model setup
    input_dim = X.shape[1]
    model = BayesianNet(input_dim=input_dim, hidden_dim=32, output_dim=len(classes))  # ← Multiclass setup

    # Train
    train(model, X_train, y_train, epochs=100)

    # Predict
    mean_preds, std_preds = model.predict_multiple(X_test, n_samples=100)

    # --- Dynamically convert predictions and labels ---
    if model.multiclass:
        y_pred = mean_preds.argmax(dim=1)
    else:
        y_pred = (mean_preds > 0.5).float()

    if y_test.ndim == 2 and y_test.shape[1] == 1:
        y_test = y_test.squeeze(1)

    # Accuracy
    acc = (y_pred == y_test).float().mean()
    print(f"[MNIST {classes}] Test Accuracy: {acc.item():.4f}")

    # Confidence filtering
    if model.multiclass:
        std_conf = std_preds.std(dim=1)  # Use std across classes as uncertainty
    else:
        std_conf = std_preds.squeeze()

    conf_threshold = 0.1
    confident_mask = (std_conf < conf_threshold)
    unconfident_mask = (std_conf > 0.2)
    y_conf = y_test[confident_mask]
    y_pred_conf = y_pred[confident_mask]

    if len(y_conf) > 0:
        acc_conf = (y_conf == y_pred_conf).float().mean()
        coverage = len(y_conf) / len(y_test)
        print(f"[MNIST {classes}] Accuracy (confident): {acc_conf:.4f} | Coverage: {coverage:.2%}")
    else:
        print("No confident predictions.")
    print(len(unconfident_mask))
    # Raw prediction samples for one instance
    raw_pred = model.predict_nested_samples_with_raw(X_test[unconfident_mask][0].unsqueeze(0), n_p=10, n_mask=10)
    probs = raw_pred[0][0]
    print(MultinomialOpinion.from_list_prob(probs))
    print()

    # Optional: plot histogram of uncertainty
    plt.hist(std_conf.numpy(), bins=30, color='skyblue')
    plt.title('Predictive Uncertainty on MNIST')
    plt.xlabel('Std Dev')
    plt.ylabel('Frequency')
    plt.show()

    

    image = X_test[unconfident_mask][0].reshape(28, 28)
    plt.imshow(image, cmap="gray")
    plt.title("MNIST Sample")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    # test_make_moons()
    # test_breast_cancer()
    test_mnist()