import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn

from models.smallcnn import SmallCNN
from autoattack import AutoAttack 

def main():
    device = torch.device("cpu")

    transform = transforms.Compose([transforms.ToTensor()])
    testset = CIFAR10(root="./data", train=False,
                      download=True, transform=transform)
    loader = DataLoader(testset, batch_size=128, shuffle=False)

    model = SmallCNN().to(device)
    #    model.load_state_dict(torch.load("results/smallcnn_clean.pth"))
    model.load_state_dict(torch.load("results/smallcnn_pgd.pth"))
    model.eval()

    adversary = AutoAttack(
        model,
        norm='Linf',
        eps=8/255,
        version='standard',
	device='cpu'
    )

    x_all, y_all = [], []
    for x, y in loader:
        x_all.append(x)
        y_all.append(y)

    x_all = torch.cat(x_all).to(device)
    y_all = torch.cat(y_all).to(device)

    x_adv = adversary.run_standard_evaluation(x_all, y_all)

    with torch.no_grad():
        acc = (model(x_adv).argmax(1) == y_all).float().mean()

    print("AutoAttack robust accuracy:", acc.item())

if __name__ == "__main__":
    main()
