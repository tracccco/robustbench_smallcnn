import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from models.smallcnn import SmallCNN


# =========================
# PGD attack (Linf)
# =========================
def pgd_attack(model, x, y,
               eps=8/255, alpha=2/255, steps=10):
    """
    Standard PGD attack used in adversarial training
    """
    model.eval()  # important: avoid BN / Dropout randomness

    x_adv = x.detach() + torch.empty_like(x).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    for _ in range(steps):
        x_adv.requires_grad_()
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)

        grad = torch.autograd.grad(loss, x_adv)[0]

        x_adv = x_adv + alpha * grad.sign()
        eta = torch.clamp(x_adv - x, min=-eps, max=eps)
        x_adv = torch.clamp(x + eta, 0.0, 1.0).detach()

    return x_adv


# =========================
# Evaluation (clean)
# =========================
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


# =========================
# Training (PGD-AT)
# =========================
def train():
    # CPU
    device = torch.device("cpu")
    print("Using device:", device)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    trainset = CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )
    testset = CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
    testloader  = DataLoader(testset, batch_size=256, shuffle=False)

    model = SmallCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    epochs = 5

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for x, y in trainloader:
            x, y = x.to(device), y.to(device)

            # PGD adversarial training
            x_adv = pgd_attack(model, x, y)

            optimizer.zero_grad()
            logits = model(x_adv)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)

        avg_loss = running_loss / len(trainset)
        acc = evaluate(model, testloader, device)

        print(f"Epoch {epoch}/{epochs} - loss: {avg_loss:.4f}, clean acc: {acc:.4f}")

    torch.save(model.state_dict(), "results/smallcnn_pgd.pth")
    print("Saved PGD-trained model to results/smallcnn_pgd.pth")


if __name__ == "__main__":
    train()
