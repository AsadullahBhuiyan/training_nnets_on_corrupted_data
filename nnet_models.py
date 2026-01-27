import math
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss


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
    corruption_mode: str = "replacement"
    sigma: float = 0.0
    mlp_hidden_sizes: list[int] | None = None
    custom_split: bool = False
    test_fraction: float = 0.2
    split_seed: int = 1234
    split_source: str = "train"  # options: "train", "full"
    cpu_threads: int = 1
    brightness_scale: float = 1.0
    rbm_components: int = 256
    rbm_learning_rate: float = 0.01
    rbm_batch_size: int = 128
    rbm_n_iter: int = 10
    rbm_classifier_max_iter: int = 1000
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


MLP_HIDDEN_SIZES = (256, 256)
CNN_CHANNELS = (32, 64)
CNN_FC_SIZE = 128


class MLP(nn.Module):
    def __init__(self, activation: str, hidden_sizes: Iterable[int] = MLP_HIDDEN_SIZES) -> None:
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
            nn.Conv2d(1, CNN_CHANNELS[0], kernel_size=3, padding=1),
            get_activation(activation),
            nn.MaxPool2d(2),
            nn.Conv2d(CNN_CHANNELS[0], CNN_CHANNELS[1], kernel_size=3, padding=1),
            get_activation(activation),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(CNN_CHANNELS[1] * 7 * 7, CNN_FC_SIZE),
            get_activation(activation),
            nn.Linear(CNN_FC_SIZE, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def build_model(
    model_type: str,
    activation: str,
    mlp_hidden_sizes: Iterable[int] | None = None,
) -> nn.Module:
    model_type = model_type.lower()
    if model_type == "mlp":
        hidden_sizes = mlp_hidden_sizes if mlp_hidden_sizes is not None else MLP_HIDDEN_SIZES
        return MLP(activation, hidden_sizes=hidden_sizes)
    if model_type == "cnn":
        return SimpleCNN(activation)
    raise ValueError(f"Unsupported model_type: {model_type}")


def get_model_hparams(model_type: str) -> dict:
    model_type = model_type.lower()
    if model_type == "mlp":
        return {
            "hidden_layer_count": len(MLP_HIDDEN_SIZES),
            "hidden_sizes": list(MLP_HIDDEN_SIZES),
        }
    if model_type == "cnn":
        return {
            "conv_layer_count": len(CNN_CHANNELS),
            "conv_channels": list(CNN_CHANNELS),
            "fc_hidden_size": CNN_FC_SIZE,
        }
    raise ValueError(f"Unsupported model_type: {model_type}")


def corrupt_pixels(x: torch.Tensor, p: float) -> torch.Tensor:
    if p <= 0:
        return x
    mask = torch.rand_like(x) < p
    replacement = torch.rand_like(x)
    return torch.where(mask, replacement, x)


def add_additive_uniform_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x
    noise = (torch.rand_like(x) - 0.5) * sigma
    return torch.clamp(x + noise, 0.0, 1.0)


def apply_corruption(x: torch.Tensor, mode: str, p: float, sigma: float) -> torch.Tensor:
    mode = mode.lower()
    if mode == "replacement":
        return corrupt_pixels(x, p)
    if mode == "additive":
        return add_additive_uniform_noise(x, sigma)
    raise ValueError(f"Unsupported corruption_mode: {mode}")


def apply_corruption_numpy(x: np.ndarray, mode: str, p: float, sigma: float) -> np.ndarray:
    mode = mode.lower()
    if mode == "replacement":
        if p <= 0:
            return x
        mask = np.random.rand(*x.shape) < p
        replacement = np.random.rand(*x.shape)
        return np.where(mask, replacement, x)
    if mode == "additive":
        if sigma <= 0:
            return x
        noise = (np.random.rand(*x.shape) - 0.5) * sigma
        return np.clip(x + noise, 0.0, 1.0)
    raise ValueError(f"Unsupported corruption_mode: {mode}")


class CorruptedDataset(Dataset):
    def __init__(self, base: Dataset, p: float, sigma: float, mode: str) -> None:
        self.base = base
        self.p = p
        self.sigma = sigma
        self.mode = mode

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        x, y = self.base[idx]
        x = apply_corruption(x, self.mode, self.p, self.sigma)
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
    brightness_scale: float = 1.0,
    corruption_mode: str = "replacement",
    sigma: float = 0.0,
    custom_split: bool = False,
    test_fraction: float = 0.2,
    split_seed: int = 1234,
    split_source: str = "train",
) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.clamp(x * brightness_scale, 0.0, 1.0)),
        ]
    )
    train_base = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_base = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    if custom_split:
        if split_source == "full":
            full = torch.utils.data.ConcatDataset([train_base, test_base])
        else:
            full = train_base
        test_size = max(1, int(len(full) * test_fraction))
        train_size = len(full) - test_size
        generator = torch.Generator().manual_seed(split_seed)
        train_base, test_base = torch.utils.data.random_split(full, [train_size, test_size], generator=generator)

    train_base = _limit_dataset(train_base, max_train_samples)
    train_set = CorruptedDataset(train_base, p, sigma, corruption_mode)
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


def load_mnist_numpy(
    brightness_scale: float = 1.0,
    max_train_samples: int | None = None,
    custom_split: bool = False,
    test_fraction: float = 0.2,
    split_seed: int = 1234,
    split_source: str = "train",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.clamp(x * brightness_scale, 0.0, 1.0)),
        ]
    )
    train_base = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_base = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    if custom_split:
        if split_source == "full":
            full = torch.utils.data.ConcatDataset([train_base, test_base])
        else:
            full = train_base
        test_size = max(1, int(len(full) * test_fraction))
        train_size = len(full) - test_size
        generator = torch.Generator().manual_seed(split_seed)
        train_base, test_base = torch.utils.data.random_split(full, [train_size, test_size], generator=generator)

    if max_train_samples is not None:
        max_train_samples = max(1, min(len(train_base), max_train_samples))
        train_base = torch.utils.data.Subset(train_base, list(range(max_train_samples)))

    def _to_numpy(ds: Dataset) -> Tuple[np.ndarray, np.ndarray]:
        xs = []
        ys = []
        for x, y in ds:
            xs.append(x.view(-1).numpy())
            ys.append(y)
        return np.stack(xs, axis=0), np.array(ys)

    x_train, y_train = _to_numpy(train_base)
    x_test, y_test = _to_numpy(test_base)
    return x_train, y_train, x_test, y_test


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
        f"p={cfg.p:.3f} sigma={cfg.sigma:.3f} seed={cfg.seed}"
    )

    if cfg.model_type.lower() == "rbm":
        x_train, y_train, x_test, y_test = load_mnist_numpy(
            brightness_scale=cfg.brightness_scale,
            max_train_samples=cfg.max_train_samples,
            custom_split=cfg.custom_split,
            test_fraction=cfg.test_fraction,
            split_seed=cfg.split_seed,
            split_source=cfg.split_source,
        )
        x_train = apply_corruption_numpy(
            x_train, cfg.corruption_mode, cfg.p, cfg.sigma
        )

        rbm = BernoulliRBM(
            n_components=cfg.rbm_components,
            learning_rate=cfg.rbm_learning_rate,
            batch_size=cfg.rbm_batch_size,
            n_iter=cfg.rbm_n_iter,
            random_state=cfg.seed,
            verbose=False,
        )
        clf = LogisticRegression(
            max_iter=cfg.rbm_classifier_max_iter,
            n_jobs=1,
            multi_class="auto",
        )
        model = Pipeline([("rbm", rbm), ("logreg", clf)])
        model.fit(x_train, y_train)
        probs = model.predict_proba(x_test)
        preds = probs.argmax(axis=1)
        test_acc = float((preds == y_test).mean())
        test_loss = float(log_loss(y_test, probs))

        return {
            "activation": cfg.activation,
            "model_type": cfg.model_type,
            "corruption_mode": cfg.corruption_mode,
            "p": cfg.p,
            "sigma": cfg.sigma,
            "seed": cfg.seed,
            "train_loss": math.nan,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "brightness_scale": cfg.brightness_scale,
            "mlp_hidden_sizes": cfg.mlp_hidden_sizes,
            "rbm_components": cfg.rbm_components,
        }

    train_loader, test_loader = get_dataloaders(
        batch_size=cfg.batch_size,
        p=cfg.p,
        num_workers=cfg.num_workers,
        max_train_samples=cfg.max_train_samples,
        brightness_scale=cfg.brightness_scale,
        corruption_mode=cfg.corruption_mode,
        sigma=cfg.sigma,
        custom_split=cfg.custom_split,
        test_fraction=cfg.test_fraction,
        split_seed=cfg.split_seed,
        split_source=cfg.split_source,
    )
    model = build_model(
        cfg.model_type,
        cfg.activation,
        mlp_hidden_sizes=cfg.mlp_hidden_sizes,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    last_train_loss = math.nan
    width_label = cfg.mlp_hidden_sizes[0] if cfg.mlp_hidden_sizes else "n/a"
    strength_label = cfg.p if cfg.corruption_mode == "replacement" else cfg.sigma
    strength_name = "p" if cfg.corruption_mode == "replacement" else "sigma"
    for _ in tqdm(
        range(cfg.epochs),
        desc=f"epochs (w={width_label}, {strength_name}={strength_label:.3f})",
        unit="epoch",
        leave=False,
        disable=cfg.epochs <= 1,
    ):
        last_train_loss = train_one_epoch(model, train_loader, optimizer, device)

    test_loss, test_acc = evaluate(model, test_loader, device)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"[{timestamp}] done: model={cfg.model_type} act={cfg.activation} "
        f"p={cfg.p:.3f} sigma={cfg.sigma:.3f} acc={test_acc:.4f}"
    )
    return {
        "activation": cfg.activation,
        "model_type": cfg.model_type,
        "corruption_mode": cfg.corruption_mode,
        "p": cfg.p,
        "sigma": cfg.sigma,
        "seed": cfg.seed,
        "train_loss": last_train_loss,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "brightness_scale": cfg.brightness_scale,
        "mlp_hidden_sizes": cfg.mlp_hidden_sizes,
    }
