import os
max_cpu = 20
os.environ.setdefault("OMP_NUM_THREADS", str(max_cpu))
os.environ.setdefault("MKL_NUM_THREADS", str(max_cpu))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(max_cpu))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(max_cpu))

import csv
import random
import statistics
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm

from nnet_models import build_model, compute_loss, evaluate


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_runs(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, float], list[dict]] = {}
    for row in rows:
        key = (row["activation"], row["model_type"], float(row["p"]))
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict] = []
    for (activation, model_type, p), items in grouped.items():
        accs = [float(r["test_accuracy"]) for r in items]
        losses = [float(r["test_loss"]) for r in items]
        summary_rows.append(
            {
                "activation": activation,
                "model_type": model_type,
                "corruption_mode": "binary_replacement",
                "p": p,
                "repeats": len(items),
                "mean_test_accuracy": statistics.mean(accs),
                "std_test_accuracy": statistics.pstdev(accs) if len(accs) > 1 else 0.0,
                "stderr_test_accuracy": (statistics.pstdev(accs) / (len(items) ** 0.5))
                if len(items) > 1
                else 0.0,
                "mean_test_loss": statistics.mean(losses),
                "std_test_loss": statistics.pstdev(losses) if len(losses) > 1 else 0.0,
                "stderr_test_loss": (statistics.pstdev(losses) / (len(items) ** 0.5))
                if len(items) > 1
                else 0.0,
            }
        )
    summary_rows.sort(key=lambda r: (r["model_type"], r["activation"], r["p"]))
    return summary_rows


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


class BinaryCorruptedDataset(Dataset):
    def __init__(self, base: Dataset, p: float) -> None:
        self.base = base
        self.p = p

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        x, y = self.base[idx]
        x = apply_binary_replacement(x, self.p)
        return x, y


def _limit_dataset(base: Dataset, max_samples: int | None) -> Dataset:
    if max_samples is None:
        return base
    max_samples = max(1, min(len(base), max_samples))
    return torch.utils.data.Subset(base, list(range(max_samples)))


def get_binary_dataloaders(
    batch_size: int,
    p: float,
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
    train_set = BinaryCorruptedDataset(train_base, p)

    train_loader = DataLoader(
        train_set,
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


def train_and_evaluate_binary(cfg: dict) -> dict:
    torch.set_num_threads(max(1, cfg["cpu_threads"]))
    device = torch.device("cpu")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"[{timestamp}] start: model={cfg['model_type']} act={cfg['activation']} "
        f"p={cfg['p']:.3f} seed={cfg['seed']}"
    )

    train_loader, test_loader = get_binary_dataloaders(
        batch_size=cfg["batch_size"],
        p=cfg["p"],
        num_workers=cfg["num_workers"],
        max_train_samples=cfg["max_train_samples"],
        custom_split=cfg["custom_split"],
        test_fraction=cfg["test_fraction"],
        split_seed=cfg["split_seed"],
        split_source=cfg["split_source"],
    )
    model = build_model(
        cfg["model_type"],
        cfg["activation"],
        mlp_hidden_sizes=cfg["mlp_hidden_sizes"],
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    for _ in tqdm(
        range(cfg["epochs"]),
        desc=f"epochs (w={cfg['mlp_width']}, p={cfg['p']:.3f})",
        unit="epoch",
        leave=False,
        disable=cfg["epochs"] <= 1,
    ):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = compute_loss(logits, y, cfg["loss_type"])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        _ = total_loss / len(train_loader.dataset)

    test_loss, test_acc = evaluate(model, test_loader, device, cfg["loss_type"])
    return {
        "activation": cfg["activation"],
        "model_type": cfg["model_type"],
        "corruption_mode": "binary_replacement",
        "p": cfg["p"],
        "seed": cfg["seed"],
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "mlp_hidden_sizes": cfg["mlp_hidden_sizes"],
    }


def build_cache_key(
    model_types: list[str],
    repeats: int,
    epochs: int,
    test_fraction: float,
    ps: list[float],
    mlp_width: int,
    mlp_depth: int,
    loss_type: str,
) -> str:
    model_tag = "-".join(model_types)
    p_min, p_max = min(ps), max(ps)
    strength_tag = f"p{p_min:.2f}-{p_max:.2f}"
    return (
        f"{model_tag}_r{repeats}_e{epochs}_tf{test_fraction:.3f}_"
        f"{strength_tag}_w{mlp_width}_d{mlp_depth}_loss-{loss_type}"
    )


def write_summary_pdf(path: str, summary_rows: list[dict], metadata: dict) -> None:
    df = pd.DataFrame(summary_rows)
    model_types = sorted(df["model_type"].unique())

    with PdfPages(path) as pdf:
        for model_type in model_types:
            sub = df[df["model_type"] == model_type]
            activations = sorted(sub["activation"].unique())
            n = len(activations)
            fig, axes = plt.subplots(1, n, figsize=(4 * n, 3), sharey=True)
            if n == 1:
                axes = [axes]
            for ax, act in zip(axes, activations):
                d = sub[sub["activation"] == act].sort_values("p")
                ax.errorbar(
                    d["p"],
                    d["mean_test_accuracy"],
                    yerr=d["stderr_test_accuracy"],
                    marker="o",
                )
                ax.set_title(f"{model_type} / {act}")
                ax.set_xlabel("p")
                ax.grid(True, alpha=0.3)
            axes[0].set_ylabel("mean test accuracy")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            fig, axes = plt.subplots(1, n, figsize=(4 * n, 3), sharey=True)
            if n == 1:
                axes = [axes]
            for ax, act in zip(axes, activations):
                d = sub[sub["activation"] == act].sort_values("p")
                ax.errorbar(
                    d["p"],
                    d["mean_test_loss"],
                    yerr=d["stderr_test_loss"],
                    marker="o",
                )
                ax.set_title(f"{model_type} / {act}")
                ax.set_xlabel("p")
                ax.grid(True, alpha=0.3)
            axes[0].set_ylabel("mean test loss")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            fig, axes = plt.subplots(1, n, figsize=(4 * n, 3), sharey=True)
            if n == 1:
                axes = [axes]
            for ax, act in zip(axes, activations):
                d = sub[sub["activation"] == act].sort_values("p")
                ax.plot(d["p"], d["std_test_accuracy"], marker="o")
                ax.set_title(f"{model_type} / {act}")
                ax.set_xlabel("p")
                ax.grid(True, alpha=0.3)
            axes[0].set_ylabel("std test accuracy")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        meta_lines = [f"{key}: {value}" for key, value in metadata.items()]
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.05, 0.95, "Run Metadata", fontsize=14, fontweight="bold", va="top")
        fig.text(0.05, 0.9, "\n".join(meta_lines), fontsize=10, va="top")
        plt.axis("off")
        pdf.savefig(fig)
        plt.close(fig)


def main() -> None:
    start_time = datetime.now()
    # Manual configuration (edit these values directly).
    activations = ["relu", "tanh", "sigmoid", "gelu"]
    model_types = ["mlp"]  # options: "mlp", "cnn"
    ps = np.linspace(0.0, 1.0, 21)  # corruption probabilities
    repeats = 100
    epochs = 20
    batch_size = 128
    learning_rate = 1e-3
    weight_decay = 0.0
    loss_type = "cross_entropy"  # options: "cross_entropy", "quadratic"
    mlp_width = 256
    mlp_depth = 2
    max_workers = max_cpu
    data_workers = 1
    cpu_threads_per_worker = 1
    cpu_cores = list(range(max_cpu)) # Example: [0, 1, 2, 3] to pin processes.
    custom_split = False
    test_fraction = 0.2
    split_seed = 1234
    split_source = "train"
    output_dir = "results"
    suffix = "binary_mnist_data"
    seed = 1234
    max_train_samples = None
    use_cuda = False

    if use_cuda and max_workers > 1:
        raise ValueError("use_cuda=True only supports max_workers=1 for now.")

    apply_cpu_affinity(cpu_cores)
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    configs: list[dict] = []
    for activation in activations:
        for model_type in model_types:
            for p in ps:
                mlp_hidden_sizes = [mlp_width] * mlp_depth if model_type == "mlp" else None
                for _ in range(repeats):
                    run_seed = random.randint(1, 1_000_000_000)
                    configs.append(
                        {
                            "activation": activation,
                            "model_type": model_type,
                            "p": p,
                            "epochs": epochs,
                            "batch_size": batch_size,
                            "learning_rate": learning_rate,
                            "weight_decay": weight_decay,
                            "loss_type": loss_type,
                            "seed": run_seed,
                            "num_workers": data_workers,
                            "cpu_threads": cpu_threads_per_worker,
                            "mlp_hidden_sizes": mlp_hidden_sizes,
                            "mlp_width": mlp_width,
                            "custom_split": custom_split,
                            "test_fraction": test_fraction,
                            "split_seed": split_seed,
                            "split_source": split_source,
                            "max_train_samples": max_train_samples,
                        }
                    )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    suffix_tag = f"_{suffix}" if suffix else ""
    suffix_note = f" (suffix={suffix})" if suffix else ""
    print(
        f"[{timestamp}] Launching {len(configs)} binary runs with max_workers={max_workers}..."
        f"{suffix_note}"
    )

    results: list[dict] = []
    seen_ps: set[float] = set()
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(cpu_cores,),
    ) as executor:
        futures = [executor.submit(train_and_evaluate_binary, cfg) for cfg in configs]
        desc = f"Binary runs{suffix_note}"
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc, unit="run"):
            res = future.result()
            results.append(res)
            if res["p"] not in seen_ps:
                seen_ps.add(res["p"])
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{timestamp}] new p encountered: {res['p']:.3f}")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"[{timestamp}] done: model={res['model_type']} act={res['activation']} "
                f"p={res['p']:.3f} acc={res['test_accuracy']:.4f}"
            )

    cache_key = build_cache_key(
        model_types=model_types,
        repeats=repeats,
        epochs=epochs,
        test_fraction=test_fraction,
        ps=ps,
        mlp_width=mlp_width,
        mlp_depth=mlp_depth,
        loss_type=loss_type,
    )
    per_run_path = os.path.join(output_dir, f"results_per_run_{cache_key}{suffix_tag}.csv")
    summary_path = os.path.join(output_dir, f"results_summary_{cache_key}{suffix_tag}.csv")
    write_csv(per_run_path, results)

    summary_rows = summarize_runs(results)
    write_csv(summary_path, summary_rows)

    pdf_path = os.path.join(output_dir, f"accuracy_vs_{cache_key}{suffix_tag}.pdf")
    metadata = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "activations": activations,
        "model_types": model_types,
        "corruption_mode": "binary_replacement",
        "ps": ps,
        "repeats": repeats,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "loss_type": loss_type,
        "mlp_width": mlp_width,
        "mlp_depth": mlp_depth,
        "max_workers": max_workers,
        "data_workers": data_workers,
        "cpu_threads_per_worker": cpu_threads_per_worker,
        "cpu_cores": cpu_cores,
        "custom_split": custom_split,
        "test_fraction": test_fraction,
        "split_seed": split_seed,
        "split_source": split_source,
        "output_dir": output_dir,
        "seed": seed,
        "max_train_samples": max_train_samples,
        "use_cuda": use_cuda,
        "total_runs": len(configs),
        "suffix": suffix,
    }
    write_summary_pdf(pdf_path, summary_rows, metadata)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Saved:")
    print(f"  {per_run_path}")
    print(f"  {summary_path}")
    print(f"  {pdf_path}")
    elapsed = datetime.now() - start_time
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Elapsed: {elapsed}")


if __name__ == "__main__":
    main()
