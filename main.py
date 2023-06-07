import torch
import wandb
import numpy as np

from argparse import ArgumentParser
from torchvision.models import resnet34
from torch import nn

from layers import *
from datasets import get_cifar10_dataloader
from train import train


def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


if __name__ == 'main':
    parser = ArgumentParser("Tensor Layer Model")

    parser.add_argument("--layer", required=True, type=str, help=f"which layer type to use")

    parser.add_argument("--path_to_save", type=str, required=False,
                        help="path to save trained model")

    parser.add_argument("--rank", type=int, required=False, default=4, help="rank of the factorisation")
    parser.add_argument("--seed", type=int, required=False, default=42, help="random seed to use during training")
    parser.add_argument("--batch_size", type=int, required=False, default=256, help="batch size to use during training")

    args = parser.parse_args()

    model = resnet34()
    layer: nn.Module
    rank: int = args.rank
    match args.layer:
        case "TT":
            layer = TTLayer(9, 4, rank=rank)
        case "TR":
            layer = TRLayer(9, 4, rank=rank)
        case "TW":
            layer = TWLayer(9, 4, rank=rank)
        case "MERA":
            layer = MERALayer(rank=rank)
        case "MERA_1":
            layer = MERASecondOnlyLayer(rank=rank)
        case "MERA_2":
            layer = MERAFirstOnlyLayer(rank=rank)
        case "MERA_12":
            layer = TreeLayer(rank=rank)
        case _:
            raise ValueError("Unknown tensor layer type")

    train_loader = get_cifar10_dataloader(args.batch_size, train=True)
    test_loader = get_cifar10_dataloader(args.batch_size, train=False)

    model.fc = nn.Sequential(
        layer,
        nn.ReLU(),
        nn.Linear(16, 10)
    )
    n_epochs = 100
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9, nesterov=False, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    loss_function = nn.CrossEntropyLoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    wandb.login()
    wandb.init(wandb.init(
        project="Tensor Layers",
        config={
            "layer type": args.layer,
            "rank": rank,
            "random seed": args.seed
        }
    ))
    set_seed(args.seed)
    train(model, train_loader, test_loader, loss_function, optimizer, scheduler, device, n_epochs=n_epochs)
    torch.save(model.state_dict(), args.path_to_save)
