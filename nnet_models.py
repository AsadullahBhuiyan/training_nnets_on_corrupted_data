import math
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm


@dataclass
class TrainConfig:
    activation: str
    model_type: str
    p: float
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    seed: int
    num_workers: int
    cpu_threads: int = 1
    max_train_samples: int | None = None
    use_cuda: bool = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(use_cuda: bool) -> torch.device:
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.1)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


class MLP(nn.Module):
    def __init__(self, activation: str, hidden_sizes: Iterable[int] = (256, 128)) -> None:
        super().__init__()
        layers = [nn.Flatten()]
        in_dim = 28 * 28
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(get_activation(activation))
            in_dim = h
        layers.append(nn.Linear(in_dim, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleCNN(nn.Module):
    def __init__(self, activation: str) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            get_activation(activation),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            get_activation(activation),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            get_activation(activation),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def build_model(model_type: str, activation: str) -> nn.Module:
    model_type = model_type.lower()
    if model_type == "mlp":
        return MLP(activation)
    if model_type == "cnn":
        return SimpleCNN(activation)
    raise ValueError(f"Unsupported model_type: {model_type}")


def corrupt_pixels(x: torch.Tensor, p: float) -> torch.Tensor:
    if p <= 0:
        return x
    mask = torch.rand_like(x) < p
    replacement = torch.rand_like(x)
    return torch.where(mask, replacement, x)


class CorruptedDataset(Dataset):
    def __init__(self, base: Dataset, p: float) -> None:
        self.base = base
        self.p = p

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        x, y = self.base[idx]
        x = corrupt_pixels(x, self.p)
        return x, y


def _limit_dataset(base: Dataset, max_samples: int | None) -> Dataset:
    if max_samples is None:
        return base
    max_samples = max(1, min(len(base), max_samples))
    return torch.utils.data.Subset(base, list(range(max_samples)))


def get_dataloaders(
    batch_size: int,
    p: float,
    num_workers: int,
    max_train_samples: int | None = None,
) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.ToTensor()
    train_base = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_base = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_base = _limit_dataset(train_base, max_train_samples)
    train_set = CorruptedDataset(train_base, p)
    test_set = test_base

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return train_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy


def train_and_evaluate(cfg: TrainConfig) -> dict:
    torch.set_num_threads(max(1, cfg.cpu_threads))
    set_seed(cfg.seed)
    device = get_device(cfg.use_cuda)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"[{timestamp}] start: model={cfg.model_type} act={cfg.activation} "
        f"p={cfg.p:.3f} seed={cfg.seed}"
    )

    train_loader, test_loader = get_dataloaders(
        batch_size=cfg.batch_size,
        p=cfg.p,
        num_workers=cfg.num_workers,
        max_train_samples=cfg.max_train_samples,
    )
    model = build_model(cfg.model_type, cfg.activation).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    last_train_loss = math.nan
    for _ in tqdm(
        range(cfg.epochs),
        desc="epochs",
        unit="epoch",
        leave=False,
        disable=cfg.epochs <= 1,
    ):
        last_train_loss = train_one_epoch(model, train_loader, optimizer, device)

    test_loss, test_acc = evaluate(model, test_loader, device)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"[{timestamp}] done: model={cfg.model_type} act={cfg.activation} "
        f"p={cfg.p:.3f} acc={test_acc:.4f}"
    )
    return {
        "activation": cfg.activation,
        "model_type": cfg.model_type,
        "p": cfg.p,
        "seed": cfg.seed,
        "train_loss": last_train_loss,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
    }
