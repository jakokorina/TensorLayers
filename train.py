import torch
import wandb

from torch.nn.utils import clip_grad_norm_
from tqdm.notebook import tqdm


def train_epoch(model, loader, loss_function, optimizer, device, epoch, max_grad_norm=10.0, log_steps=10):
    cum_loss = 0

    model.train()
    for step, (x, y) in enumerate(tqdm(loader)):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_function(pred, y)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        cum_loss += loss.detach().item()

        if step % log_steps == log_steps - 1:
            metrics = {"train/loss": cum_loss / log_steps,
                       "train/epoch": epoch + step / len(loader)}
            cum_loss = 0
            wandb.log(metrics)


def validate(model, loader, loss_function, device, epoch):
    cum_loss = 0
    cum_acc = 0
    n = 0
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            pred = model(images)
            loss = loss_function(pred, labels)
            cum_loss += loss.detach().item() * labels.size(0)
            cum_acc += (pred.argmax(axis=-1) == labels).sum()
            n += labels.size(0)

    metrics = {
        "test/loss": cum_loss / n,
        "test/accuracy": cum_acc / n,
        "test/epoch": epoch
    }
    wandb.log(metrics)


def train(model, train_loader, test_loader, loss_function, optimizer, scheduler, device, n_epochs=10,
          max_grad_norm=10.0, log_steps=10):

    model.to(device)
    for epoch in range(n_epochs):
        train_epoch(model, train_loader, loss_function, optimizer, device, epoch, max_grad_norm, log_steps)
        scheduler.step()
        validate(model, test_loader, loss_function, device, epoch)
