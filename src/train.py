"""Fine-tune MobileNetV2 on CIFAR-10 and register the result in MLflow.

Pipeline:
  1. Download CIFAR-10 (cached after first run).
  2. Load MobileNetV2 with ImageNet weights; replace the classifier head for 10 classes.
  3. Train for N epochs, logging metrics to MLflow after each epoch.
  4. Measure single-image inference latency.
  5. Register the trained model in the MLflow model registry as 'cifar10-mobilenet'.

The registered model is later promoted to Production by scripts/promote.py,
then loaded by src/serve.py at container startup.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import click
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
MODEL_NAME = "cifar10-mobilenet"
DATA_DIR = Path(os.environ.get("CIFAR_DATA_DIR", "./data"))


def build_model(num_classes: int = 10) -> nn.Module:
    # Start from ImageNet-pretrained weights — fine-tuning converges much faster
    # than training from scratch, which matters given the homelab time budget.
    weights = MobileNet_V2_Weights.IMAGENET1K_V2
    model = mobilenet_v2(weights=weights)
    # Replace only the final classification layer; all other weights are reused.
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def build_loaders(batch_size: int, subset: int | None, num_workers: int):
    # Training transform includes random horizontal flip for basic augmentation.
    # Eval transform is deterministic — no augmentation during validation.
    train_tf = transforms.Compose([
        transforms.Resize(96),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(96),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_set = datasets.CIFAR10(str(DATA_DIR), train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(str(DATA_DIR), train=False, download=True, transform=eval_tf)
    if subset is not None:
        # --subset limits training to the first N samples and validation to N/5
        # (min 200) so CI runs finish in under a minute on the homelab HDD.
        train_set = Subset(train_set, range(min(subset, len(train_set))))
        test_set = Subset(test_set, range(min(max(subset // 5, 200), len(test_set))))
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )
    return train_loader, test_loader


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module):
    model.eval()
    total, correct, loss_sum, latency_sum, batches = 0, 0, 0.0, 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            t0 = time.perf_counter()
            logits = model(x)
            latency_sum += (time.perf_counter() - t0) * 1000.0
            batches += 1
            loss_sum += criterion(logits, y).item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    avg_latency_ms = latency_sum / max(batches, 1)
    return {
        "loss": loss_sum / max(total, 1),
        "accuracy": correct / max(total, 1),
        "latency_ms_per_batch": avg_latency_ms,
    }


@click.command()
@click.option("--epochs", default=1, show_default=True, type=int)
@click.option("--subset", default=None, type=int, help="Use first N training samples (for fast tests).")
@click.option("--batch-size", default=64, show_default=True, type=int)
@click.option("--lr", default=1e-3, show_default=True, type=float)
@click.option("--num-workers", default=2, show_default=True, type=int)
@click.option("--tracking-uri", default=lambda: os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"), show_default=True)
@click.option("--experiment", default="cifar10-mobilenet", show_default=True)
@click.option("--register/--no-register", default=True, show_default=True)
def main(epochs, subset, batch_size, lr, num_workers, tracking_uri, experiment, register):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    click.echo(f"[train] device={device} tracking_uri={tracking_uri}")

    train_loader, test_loader = build_loaders(batch_size, subset, num_workers)
    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    with mlflow.start_run() as run:
        # Log all hyperparameters upfront so each run is fully reproducible.
        mlflow.log_params({
            "epochs": epochs,
            "subset": subset if subset is not None else "full",
            "batch_size": batch_size,
            "lr": lr,
            "model": "mobilenet_v2",
            "device": str(device),
            "input_size": 96,
        })

        for epoch in range(epochs):
            # --- training loop ---
            model.train()
            running_loss, running_correct, running_total = 0.0, 0, 0
            for step, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * x.size(0)
                running_correct += (logits.argmax(dim=1) == y).sum().item()
                running_total += x.size(0)

            train_loss = running_loss / max(running_total, 1)
            train_acc = running_correct / max(running_total, 1)
            eval_metrics = evaluate(model, test_loader, device, criterion)

            # val_accuracy is read by scripts/promote.py for the quality gate check.
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            mlflow.log_metric("val_loss", eval_metrics["loss"], step=epoch)
            mlflow.log_metric("val_accuracy", eval_metrics["accuracy"], step=epoch)
            mlflow.log_metric("val_latency_ms_per_batch", eval_metrics["latency_ms_per_batch"], step=epoch)
            click.echo(
                f"[epoch {epoch+1}/{epochs}] "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={eval_metrics['loss']:.4f} val_acc={eval_metrics['accuracy']:.4f} "
                f"latency={eval_metrics['latency_ms_per_batch']:.1f}ms/batch"
            )

        # --- single-image latency benchmark ---
        # Measured separately from batch evaluation because serve.py processes
        # one image at a time. 3 warmup runs discard JIT compilation overhead.
        model.eval()
        sample = next(iter(test_loader))[0][:1].to(device)
        with torch.no_grad():
            for _ in range(3):  # warmup
                model(sample)
            t0 = time.perf_counter()
            for _ in range(20):
                model(sample)
            single_latency_ms = (time.perf_counter() - t0) * 1000.0 / 20
        mlflow.log_metric("inference_latency_ms", single_latency_ms)
        click.echo(f"[train] single-image latency: {single_latency_ms:.2f} ms")

        mlflow.set_tag("classes", ",".join(CIFAR10_CLASSES))

        if register:
            # Move model to CPU before saving — the serving container may not have a GPU.
            mlflow.pytorch.log_model(
                pytorch_model=model.cpu(),
                artifact_path="model",
                registered_model_name=MODEL_NAME,
                input_example=sample.cpu().numpy(),
            )
            click.echo(f"[train] registered model '{MODEL_NAME}' from run {run.info.run_id}")
        else:
            mlflow.pytorch.log_model(model.cpu(), artifact_path="model",
                                     input_example=sample.cpu().numpy())


if __name__ == "__main__":
    main()
