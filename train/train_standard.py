import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from models.smallcnn import SmallCNN

def train():
    device = torch.device("cpu")

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    trainset = CIFAR10(root="./data", train=True,
                       download=True, transform=transform)
    testset  = CIFAR10(root="./data", train=False,
                       download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
    testloader  = DataLoader(testset, batch_size=256)

    model = SmallCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(5):
        model.train()
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        acc = evaluate(model, testloader, device)
        print(f"Epoch {epoch+1}, clean acc = {acc:.4f}")

    torch.save(model.state_dict(), "results/smallcnn_clean.pth")

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

if __name__ == "__main__":
    train()
