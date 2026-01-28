import os
max_cpu = 20
os.environ.setdefault("OMP_NUM_THREADS", str(max_cpu))
os.environ.setdefault("MKL_NUM_THREADS", str(max_cpu))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(max_cpu))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(max_cpu))

import statistics
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm

from nnet_models import build_model, evaluate, compute_loss


def apply_cpu_affinity(cores: list[int] | None) -> None:
    if not cores:
        return
    try:
        os.sched_setaffinity(0, set(cores))
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Pinned process to CPU cores: {sorted(set(cores))}")
    except AttributeError:
        print("CPU affinity is not supported on this platform.")
    except OSError as exc:
        print(f"Failed to set CPU affinity to {cores}: {exc}")


def _init_worker(cores: list[int] | None) -> None:
    apply_cpu_affinity(cores)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def binarize_tensor(x: torch.Tensor) -> torch.Tensor:
    return (x >= 0.5).float()


def apply_binary_replacement(x: torch.Tensor, p: float) -> torch.Tensor:
    mask = torch.rand_like(x) < p
    replacement = (torch.rand_like(x) < 0.5).float()
    return torch.where(mask, replacement, x)


def _limit_dataset(base: Dataset, max_samples: int | None) -> Dataset:
    if max_samples is None:
        return base
    max_samples = max(1, min(len(base), max_samples))
    return torch.utils.data.Subset(base, list(range(max_samples)))


def get_binary_dataloaders(
    batch_size: int,
    num_workers: int,
    max_train_samples: int | None = None,
    custom_split: bool = False,
    test_fraction: float = 0.2,
    split_seed: int = 1234,
    split_source: str = "train",
) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: binarize_tensor(x)),
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
    test_base = _limit_dataset(test_base, None)

    train_loader = DataLoader(
        train_base,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_base,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return train_loader, test_loader


def _worker_corruption_trials(
    model_state: dict,
    model_type: str,
    activation: str,
    mlp_hidden_sizes: list[int] | None,
    batch_size: int,
    num_workers: int,
    cpu_threads: int,
    max_train_samples: int | None,
    p: float,
    trials: int,
    custom_split: bool,
    test_fraction: float,
    split_seed: int,
    split_source: str,
) -> list[float]:
    torch.set_num_threads(max(1, cpu_threads))
    device = torch.device("cpu")
    model = build_model(model_type, activation, mlp_hidden_sizes=mlp_hidden_sizes).to(device)
    model.load_state_dict(model_state)
    _, test_loader = get_binary_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        max_train_samples=max_train_samples,
        custom_split=custom_split,
        test_fraction=test_fraction,
        split_seed=split_seed,
        split_source=split_source,
    )
    accuracies: list[float] = []
    model.eval()
    with torch.no_grad():
        for _ in range(trials):
            correct = 0
            total = 0
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)
                x = apply_binary_replacement(x, p)
                logits = model(x)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
            accuracies.append(correct / total)
    return accuracies


def evaluate_with_corruption_parallel(
    model: torch.nn.Module,
    model_type: str,
    activation: str,
    mlp_hidden_sizes: list[int] | None,
    batch_size: int,
    num_workers: int,
    cpu_threads: int,
    max_train_samples: int | None,
    p: float,
    trials: int,
    max_workers: int,
    cpu_cores: list[int] | None,
    custom_split: bool,
    test_fraction: float,
    split_seed: int,
    split_source: str,
) -> tuple[float, float, float]:
    if max_workers <= 1 or trials <= 1:
        _, test_loader = get_binary_dataloaders(
            batch_size=batch_size,
            num_workers=num_workers,
            max_train_samples=max_train_samples,
            custom_split=custom_split,
            test_fraction=test_fraction,
            split_seed=split_seed,
            split_source=split_source,
        )
        accuracies = []
        model.eval()
        with torch.no_grad():
            for _ in range(trials):
                correct = 0
                total = 0
                for x, y in test_loader:
                    x = x.to("cpu")
                    y = y.to("cpu")
                    x = apply_binary_replacement(x, p)
                    logits = model(x)
                    preds = logits.argmax(dim=1)
                    correct += (preds == y).sum().item()
                    total += y.size(0)
                accuracies.append(correct / total)
        mean_acc = statistics.mean(accuracies)
        std_acc = statistics.pstdev(accuracies) if len(accuracies) > 1 else 0.0
        stderr_acc = std_acc / (len(accuracies) ** 0.5) if len(accuracies) > 1 else 0.0
        return mean_acc, std_acc, stderr_acc

    model_state = {k: v.cpu() for k, v in model.state_dict().items()}
    per_worker = max(1, trials // max_workers)
    splits = [per_worker] * max_workers
    remainder = trials - per_worker * max_workers
    for i in range(remainder):
        splits[i] += 1
    splits = [s for s in splits if s > 0]

    all_accs: list[float] = []
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(cpu_cores,),
    ) as executor:
        futures = [
            executor.submit(
                _worker_corruption_trials,
                model_state,
                model_type,
                activation,
                mlp_hidden_sizes,
                batch_size,
                num_workers,
                cpu_threads,
                max_train_samples,
                p,
                trials_i,
                custom_split,
                test_fraction,
                split_seed,
                split_source,
            )
            for trials_i in splits
        ]
        for future in as_completed(futures):
            all_accs.extend(future.result())

    mean_acc = statistics.mean(all_accs)
    std_acc = statistics.pstdev(all_accs) if len(all_accs) > 1 else 0.0
    stderr_acc = std_acc / (len(all_accs) ** 0.5) if len(all_accs) > 1 else 0.0
    return mean_acc, std_acc, stderr_acc


def build_cache_key(
    corruption_trials: int,
    epochs: int,
    test_fraction: float,
    ps: list[float],
    mlp_width: int,
    mlp_depth: int,
) -> str:
    p_min, p_max = min(ps), max(ps)
    strength_tag = f"p{p_min:.2f}-{p_max:.2f}"
    return (
        f"binarytest_r{corruption_trials}_e{epochs}_tf{test_fraction:.3f}_"
        f"{strength_tag}_w{mlp_width}_d{mlp_depth}"
    )


def main() -> None:
    # Manual configuration (edit these values directly).
    activations = ["relu", "tanh", "sigmoid", "gelu"]
    model_type = "mlp"  # options: "mlp", "cnn"
    ps = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    corruption_trials = 100
    epochs = 20
    batch_size = 128
    learning_rate = 1e-3
    weight_decay = 0.0
    loss_type = "cross_entropy"  # options: "cross_entropy", "quadratic"
    mlp_width = 256
    mlp_depth = 1
    data_workers = 1
    max_workers = max_cpu
    cpu_threads_per_worker = 1
    cpu_cores = list(range(max_cpu)) # Example: [0, 1, 2, 3] to pin processes.
    max_train_samples = None
    use_cuda = False
    custom_split = False
    test_fraction = 0.2
    split_seed = 1234
    split_source = "train"
    output_dir = "results_noisy_test"
    suffix = "binary_mnist_data"

    if use_cuda and max_workers > 1:
        raise ValueError("use_cuda=True only supports max_workers=1 for now.")

    apply_cpu_affinity(cpu_cores)
    torch.set_num_threads(max(1, cpu_threads_per_worker))
    torch.set_num_interop_threads(1)

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

    sweep_values = ps
    sweep_label = "p"
    suffix_tag = f"_{suffix}" if suffix else ""
    suffix_note = f" (suffix={suffix})" if suffix else ""

    summary_rows = []
    for activation in activations:
        mlp_hidden_sizes = [mlp_width] * mlp_depth if model_type == "mlp" else None
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"[{timestamp}] training model once for binary noisy-test evaluation..."
            f"{suffix_note} (activation={activation})"
        )

        train_loader, test_loader = get_binary_dataloaders(
            batch_size=batch_size,
            num_workers=data_workers,
            max_train_samples=max_train_samples,
            custom_split=custom_split,
            test_fraction=test_fraction,
            split_seed=split_seed,
            split_source=split_source,
        )
        model = build_model(model_type, activation, mlp_hidden_sizes=mlp_hidden_sizes).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        for _ in tqdm(range(epochs), desc=f"epochs ({activation})", unit="epoch"):
            model.train()
            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = compute_loss(logits, y, loss_type)
                loss.backward()
                optimizer.step()

        clean_loss, clean_acc = evaluate(model, test_loader, device, loss_type)
        print(f"[{activation}] clean test accuracy: {clean_acc:.4f} (loss={clean_loss:.4f})")

        for value in tqdm(sweep_values, desc=f"{sweep_label} sweep{suffix_note} ({activation})", unit="s"):
            mean_acc, std_acc, stderr_acc = evaluate_with_corruption_parallel(
                model=model,
                model_type=model_type,
                activation=activation,
                mlp_hidden_sizes=mlp_hidden_sizes,
                batch_size=batch_size,
                num_workers=data_workers,
                cpu_threads=cpu_threads_per_worker,
                max_train_samples=max_train_samples,
                p=value,
                trials=corruption_trials,
                max_workers=max_workers,
                cpu_cores=cpu_cores,
                custom_split=custom_split,
                test_fraction=test_fraction,
                split_seed=split_seed,
                split_source=split_source,
            )
            summary_rows.append(
                {
                    "activation": activation,
                    "model_type": model_type,
                    "corruption_mode": "binary_replacement",
                    "p": value,
                    "corruption_trials": corruption_trials,
                    "mean_test_accuracy": mean_acc,
                    "std_test_accuracy": std_acc,
                    "stderr_test_accuracy": stderr_acc,
                }
            )
            print(
                f"[{activation}] p={value:.3f} mean_acc={mean_acc:.4f} "
                f"stderr={stderr_acc:.4f}"
            )

    cache_key = build_cache_key(
        corruption_trials=corruption_trials,
        epochs=epochs,
        test_fraction=test_fraction,
        ps=ps,
        mlp_width=mlp_width,
        mlp_depth=mlp_depth,
    )
    pdf_path = os.path.join(output_dir, f"noisy_test_summary_{cache_key}{suffix_tag}.pdf")
    with PdfPages(pdf_path) as pdf:
        n = len(activations)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 3), sharey=True)
        if n == 1:
            axes = [axes]
        for ax, act in zip(axes, activations):
            sub = [row for row in summary_rows if row["activation"] == act]
            sub = sorted(sub, key=lambda r: r[sweep_label])
            ps_sorted = [row[sweep_label] for row in sub]
            means = [row["mean_test_accuracy"] for row in sub]
            stderrs = [row["stderr_test_accuracy"] for row in sub]
            ax.errorbar(ps_sorted, means, yerr=stderrs, marker="o")
            ax.set_title(f"{model_type} / {act} (binary noisy test)")
            ax.set_xlabel(sweep_label)
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel("mean test accuracy")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(1, n, figsize=(4 * n, 3), sharey=True)
        if n == 1:
            axes = [axes]
        for ax, act in zip(axes, activations):
            sub = [row for row in summary_rows if row["activation"] == act]
            sub = sorted(sub, key=lambda r: r[sweep_label])
            ps_sorted = [row[sweep_label] for row in sub]
            stds = [row["std_test_accuracy"] for row in sub]
            ax.plot(ps_sorted, stds, marker="o")
            ax.set_title(f"{model_type} / {act} std (binary)")
            ax.set_xlabel(sweep_label)
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel("std test accuracy")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        metadata = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "activations": activations,
            "model_type": model_type,
            "corruption_mode": "binary_replacement",
            "ps": ps,
            "corruption_trials": corruption_trials,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "loss_type": loss_type,
            "mlp_width": mlp_width,
            "mlp_depth": mlp_depth,
            "data_workers": data_workers,
            "max_workers": max_workers,
            "cpu_threads_per_worker": cpu_threads_per_worker,
            "cpu_cores": cpu_cores,
            "custom_split": custom_split,
            "test_fraction": test_fraction,
            "split_seed": split_seed,
            "split_source": split_source,
            "output_dir": output_dir,
            "suffix": suffix,
        }
        meta_lines = [f"{key}: {value}" for key, value in metadata.items()]
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.05, 0.95, "Run Metadata", fontsize=14, fontweight="bold", va="top")
        fig.text(0.05, 0.9, "\n".join(meta_lines), fontsize=10, va="top")
        plt.axis("off")
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Saved PDF: {pdf_path}")


if __name__ == "__main__":
    main()
