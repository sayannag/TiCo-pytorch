import numpy as np

import torch
from torch import nn
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10
import copy
from torch.autograd import Variable

from lightly.models.utils import deactivate_requires_grad
from lightly.models.utils import update_momentum
import argparse
import sys
from PIL import Image

class CifarPairTransform:
    def __init__(self, train_transform = True, pair_transform = True):
        if train_transform is True:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
                # TODO: Gaussian Blurring and Solarization
        else:
            self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
        self.pair_transform = pair_transform
    def __call__(self, x):
        if self.pair_transform is True:
            y1 = self.transform(x)
            y2 = self.transform(x)
            return y1, y2
        else:
            return self.transform(x)

class TiCo(nn.Module):
    def __init__(self, backbone, final_dim=256):
        super().__init__()

        self.backbone = backbone
        self.projection_head = nn.Sequential(nn.Linear(2048, 4096, bias=False), nn.BatchNorm1d(4096),
                               nn.ReLU(inplace=True), nn.Linear(4096, final_dim, bias=True))

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward(self, x):
        query = self.backbone(x).flatten(start_dim=1)
        query = self.projection_head(query)
        return query

    def forward_momentum(self, x):
        key = self.backbone_momentum(x).flatten(start_dim=1)
        key = self.projection_head_momentum(key).detach()
        return key

class TiCo_Loss(nn.Module):
    def __init__(self, beta, rho):
        super().__init__()

        self.beta = beta
        self.rho = rho

    def forward(self, C, x_1, x_2):
        z_1 = torch.nn.functional.normalize(x_1, dim = -1)
        z_2 = torch.nn.functional.normalize(x_2, dim = -1)
        B = torch.mm(z_1.T, z_1)/z_1.shape[0]
        C = self.beta * C + (1 - self.beta) * B
        loss = - (z_1 * z_2).sum(dim=1).mean() + self.rho * (torch.mm(z_1, C) * z_1).sum(dim=1).mean()
        return loss, C

def schedule_momentum(iter, max_iter, m = 0.99):
    return m + (1 - m)*np.sin((np.pi/2)*iter/(max_iter-1))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train TiCo')
    parser.add_argument('--final_dim', default=256, type=int, help='Final dimension of the projection head (Feature dimension)')
    parser.add_argument('--beta', default=0.9, type=float, help='Hyperparameter for weighting C')
    parser.add_argument('--rho', default=20.0, type=float, help='Hyperparameter for weighting Loss (Recommended value in the paper = 8.0)')
    parser.add_argument('--LR', default=0.06, type=float, help='Learning rate')
    parser.add_argument('--start_momentum', default=0.99, type=float, help='Learning rate')
    parser.add_argument('--epochs', default=100, type=int, help='Total number of epochs')
    parser.add_argument('--batch_size', default=4096, type=int, help='Batch Size')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of workers')

    args = parser.parse_args()

    resnet = torchvision.models.resnet50()
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    model = TiCo(backbone, final_dim=args.final_dim)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    train_data = torchvision.datasets.CIFAR10("datasets/cifar10", train=True, transform=CifarPairTransform(train_transform = True, pair_transform=True), download=True)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.LR)
    criterion = TiCo_Loss(beta = args.beta, rho = args.rho)

    C_prev = Variable(torch.zeros(args.final_dim, args.final_dim), requires_grad=True).to(device)
    C_prev = C_prev.detach()

    print("Starting Training")
    for epoch in range(args.epochs):
        total_loss = 0
        momentum_val = schedule_momentum(epoch, args.epochs, m = args.start_momentum)
        for (x_query, x_key), _ in train_loader:
            update_momentum(model.backbone, model.backbone_momentum, m=momentum_val)
            update_momentum(model.projection_head, model.projection_head_momentum, m=momentum_val)
            x_query = x_query.to(device)
            x_key = x_key.to(device)
            query = model(x_query)
            key = model.forward_momentum(x_key)
            loss, C = criterion(C_prev, query, key)
            C_prev = C.detach()
            total_loss += loss.detach()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        avg_loss = total_loss / len(train_loader)
        print("epoch: ", epoch + 1, ", loss: ", avg_loss.item(), ", momentum: ", momentum_val)