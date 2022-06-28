import torch
from torch import nn
import torchvision
import copy
from torch.autograd import Variable

from lightly.data import LightlyDataset
from lightly.data import ImageCollateFunction
from lightly.models.modules import MoCoProjectionHead
from lightly.models.utils import deactivate_requires_grad
from lightly.models.utils import update_momentum
import argparse
import sys

class TiCo(nn.Module):
    def __init__(self, backbone, final_dim=128):
        super().__init__()

        self.backbone = backbone
        self.projection_head = MoCoProjectionHead(512, 512, final_dim)

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
        z_1 = torch.nn.functional.normalize(x_1, dim = 1)
        z_2 = torch.nn.functional.normalize(x_2, dim = 1)
        B = torch.mm(z_1.T, z_1)/z_1.shape[0]
        C = self.beta * C + (1 - self.beta) * B
        loss = - (z_1 * z_2).sum(dim=1).mean() + self.rho * (torch.mm(z_1, C) * z_1).sum(dim=1).mean()
        return loss, C

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train TiCo')
    parser.add_argument('--final_dim', default=128, type=int, help='Final dimension of the projection head (Feature dimension)')
    parser.add_argument('--beta', default=0.9, type=float, help='Hyperparameter for weighting C')
    parser.add_argument('--rho', default=8.0, type=float, help='Hyperparameter for weighting Loss')
    parser.add_argument('--LR', default=1e-2, type=float, help='Learning rate')
    parser.add_argument('--epochs', default=100, type=int, help='Total number of epochs')
    parser.add_argument('--batch_size', default=4096, type=int, help='Batch Size')

    args = parser.parse_args()

    resnet = torchvision.models.resnet18()
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    model = TiCo(backbone, final_dim=args.final_dim)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    cifar10 = torchvision.datasets.CIFAR10("datasets/cifar10", download=True)
    dataset = LightlyDataset.from_torch_dataset(cifar10)
    # or create a dataset from a folder containing images or videos:
    # dataset = LightlyDataset("path/to/folder")

    collate_fn = ImageCollateFunction(input_size=32)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=8,
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=args.LR)
    criterion = TiCo_Loss(beta = args.beta, rho = args.rho)

    C_prev = Variable(torch.zeros(args.final_dim, args.final_dim), requires_grad=True).to(device)
    C_prev = C_prev.detach()

    print("Starting Training")
    for epoch in range(args.epochs):
        total_loss = 0
        for (x_query, x_key), _, _ in dataloader:
            update_momentum(model.backbone, model.backbone_momentum, m=0.99)
            update_momentum(model.projection_head, model.projection_head_momentum, m=0.99)
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
        avg_loss = total_loss / len(dataloader)
        print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")