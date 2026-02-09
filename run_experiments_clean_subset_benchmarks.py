import os

import csv
import random
import statistics
from collections import deque
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

from nnet_models import build_model, compute_loss, evaluate


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_runtime_log(path: str, row: dict) -> None:
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _limit_dataset(base: Dataset, max_samples: int | None) -> Dataset:
    if max_samples is None:
        return base
    max_samples = max(1, min(len(base), max_samples))
    return torch.utils.data.Subset(base, list(range(max_samples)))


def _split_dataset(
    base: Dataset,
    test_fraction: float,
    seed: int,
) -> tuple[Dataset, Dataset]:
    if test_fraction <= 0.0:
        return base, torch.utils.data.Subset(base, [])
    if test_fraction >= 1.0:
        return torch.utils.data.Subset(base, []), base
    generator = torch.Generator().manual_seed(seed)
    test_size = int(round(len(base) * test_fraction))
    test_size = max(1, min(len(base) - 1, test_size))
    train_size = len(base) - test_size
    return torch.utils.data.random_split(base, [train_size, test_size], generator=generator)


def get_dataloaders_clean_subset(
    batch_size: int,
    clean_frac: float,
    num_workers: int,
    brightness_scale: float = 1.0,
    max_train_samples: int | None = None,
    test_fraction: float = 0.0,
    split_seed: int = 0,
    subset_seed: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.clamp(x * brightness_scale, 0.0, 1.0)),
        ]
    )
    train_base = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_base = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_base = _limit_dataset(train_base, max_train_samples)
    if test_fraction > 0.0:
        train_base, extra_test = _split_dataset(train_base, test_fraction, seed=split_seed)
        test_base = torch.utils.data.ConcatDataset([test_base, extra_test])

    clean_frac = max(0.0, min(1.0, clean_frac))
    clean_count = int(round(clean_frac * len(train_base)))
    clean_count = max(0, min(len(train_base), clean_count))
    if clean_count == 0:
        # Avoid empty DataLoader; force at least one sample.
        clean_count = 1
    generator = torch.Generator().manual_seed(subset_seed)
    perm = torch.randperm(len(train_base), generator=generator).tolist()
    clean_indices = perm[:clean_count]
    clean_total = len(clean_indices)
    train_set = torch.utils.data.Subset(train_base, clean_indices)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    train_eval_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=False,
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
    return train_loader, train_eval_loader, test_loader, clean_total


def summarize_runs(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, float], list[dict]] = {}
    for row in rows:
        key = (row["activation"], row["model_type"], float(row["clean_frac"]))
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict] = []
    for (activation, model_type, clean_frac), items in grouped.items():
        test_accs = [float(r["test_accuracy"]) for r in items]
        test_losses = [float(r["test_loss"]) for r in items]
        train_accs = [float(r["train_accuracy"]) for r in items]
        train_losses = [float(r["train_loss"]) for r in items]
        clean_total = int(items[0].get("clean_total", 0))
        train_set_size = int(items[0].get("train_set_size", 0))
        clean_frac_actual = (clean_total / train_set_size) if train_set_size else 0.0
        summary_rows.append(
            {
                "activation": activation,
                "model_type": model_type,
                "clean_frac": clean_frac,
                "train_set_size": train_set_size,
                "clean_total": clean_total,
                "clean_frac_actual": clean_frac_actual,
                "repeats": len(items),
                "mean_test_accuracy": statistics.mean(test_accs),
                "std_test_accuracy": statistics.pstdev(test_accs) if len(test_accs) > 1 else 0.0,
                "stderr_test_accuracy": (statistics.pstdev(test_accs) / (len(items) ** 0.5))
                if len(items) > 1
                else 0.0,
                "mean_test_loss": statistics.mean(test_losses),
                "std_test_loss": statistics.pstdev(test_losses) if len(test_losses) > 1 else 0.0,
                "stderr_test_loss": (statistics.pstdev(test_losses) / (len(items) ** 0.5))
                if len(items) > 1
                else 0.0,
                "mean_train_accuracy": statistics.mean(train_accs),
                "std_train_accuracy": statistics.pstdev(train_accs) if len(train_accs) > 1 else 0.0,
                "stderr_train_accuracy": (statistics.pstdev(train_accs) / (len(items) ** 0.5))
                if len(items) > 1
                else 0.0,
                "mean_train_loss": statistics.mean(train_losses),
                "std_train_loss": statistics.pstdev(train_losses) if len(train_losses) > 1 else 0.0,
                "stderr_train_loss": (statistics.pstdev(train_losses) / (len(items) ** 0.5))
                if len(items) > 1
                else 0.0,
            }
        )
    summary_rows.sort(key=lambda r: (r["model_type"], r["activation"], r["clean_frac"]))
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
        torch.set_num_threads(1)
    except Exception:
        pass


def train_and_evaluate_clean_subset(cfg: dict) -> dict:
    torch.set_num_threads(max(1, cfg["cpu_threads"]))
    device = torch.device("cpu")
    train_loader, train_eval_loader, test_loader, clean_total = get_dataloaders_clean_subset(
        batch_size=cfg["batch_size"],
        clean_frac=cfg["clean_frac"],
        num_workers=cfg["num_workers"],
        brightness_scale=cfg["brightness_scale"],
        max_train_samples=cfg["max_train_samples"],
        test_fraction=cfg["test_fraction"],
        split_seed=cfg["split_seed"],
        subset_seed=cfg["subset_seed"],
    )

    model = build_model(cfg["model_type"], cfg["activation"], mlp_hidden_sizes=cfg["mlp_hidden_sizes"]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    last_train_loss = 0.0
    for _ in tqdm(
        range(cfg["epochs"]),
        desc=f"epochs (w={cfg['mlp_width']}, clean_frac={cfg['clean_frac']:.3f})",
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
        last_train_loss = total_loss / len(train_loader.dataset)

    train_loss, train_acc = evaluate(model, train_eval_loader, device, cfg["loss_type"])
    test_loss, test_acc = evaluate(model, test_loader, device, cfg["loss_type"])
    train_set_size = len(train_loader.dataset)
    return {
        "activation": cfg["activation"],
        "model_type": cfg["model_type"],
        "clean_frac": cfg["clean_frac"],
        "seed": cfg["seed"],
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "mlp_hidden_sizes": cfg["mlp_hidden_sizes"],
        "train_set_size": train_set_size,
        "clean_total": clean_total,
        "clean_frac_actual": clean_total / train_set_size if train_set_size else 0.0,
    }


def build_cache_key(
    model_types: list[str],
    repeats: int,
    epochs: int,
    clean_fracs: list[float],
    mlp_width: int,
    mlp_depth: int,
    loss_type: str,
    test_fraction: float,
) -> str:
    model_tag = "-".join(model_types)
    f_min, f_max = min(clean_fracs), max(clean_fracs)
    return (
        f"{model_tag}_r{repeats}_e{epochs}_tf{test_fraction:.3f}_clean{f_min:.3f}-{f_max:.3f}_"
        f"w{mlp_width}_d{mlp_depth}_loss-{loss_type}"
    )


def write_summary_pdf(path: str, summary_rows: list[dict]) -> None:
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
                d = sub[sub["activation"] == act].sort_values("clean_frac")
                ax.errorbar(
                    d["clean_frac"],
                    d["mean_test_accuracy"],
                    yerr=d["stderr_test_accuracy"],
                    marker="o",
                    label="test",
                )
                ax.errorbar(
                    d["clean_frac"],
                    d["mean_train_accuracy"],
                    yerr=d["stderr_train_accuracy"],
                    marker="o",
                    label="train",
                )
                ax.set_title(f"{model_type} / {act}")
                ax.set_xlabel("clean fraction")
                ax.grid(True, alpha=0.3)
            axes[0].set_ylabel("mean accuracy")
            axes[0].legend(fontsize=8)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def main() -> None:
    start_time = datetime.now()
    # Manual configuration (edit these values directly).
    activations = ["relu", "tanh", "linear", "quadratic"]
    model_types = ["mlp"]
    clean_fracs = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]  # fraction of clean training data
    repeats = 100
    epochs = 20
    batch_size = 128
    learning_rate = 1e-3
    weight_decay = 0.0
    loss_type = "cross_entropy"  # options: "cross_entropy", "quadratic"
    mlp_width = 64
    mlp_depth = 1
    test_fraction = 1/2
    split_seed = 1234
    cpu_max = 20
    max_workers = cpu_max
    max_in_flight = max(1, cpu_max//2)
    data_workers = 0
    cpu_threads_per_worker = 1
    cpu_cores = list(range(60, 60 + cpu_max))  # e.g., use cores 100-119 if cpu_max=20
    brightness_scale = 1.0
    output_dir = "results"
    suffix = "clean_subset_benchmarks"
    seed = 1234
    max_train_samples = None
    use_cuda = False

    if use_cuda and max_workers > 1:
        raise ValueError("use_cuda=True only supports max_workers=1 for now.")

    os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads_per_worker))
    os.environ.setdefault("MKL_NUM_THREADS", str(cpu_threads_per_worker))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(cpu_threads_per_worker))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(cpu_threads_per_worker))

    apply_cpu_affinity(cpu_cores)
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    configs: list[dict] = []
    for activation in activations:
        for model_type in model_types:
            for frac in clean_fracs:
                    mlp_hidden_sizes = [mlp_width] * mlp_depth
                    for _ in range(repeats):
                        run_seed = random.randint(1, 1_000_000_000)
                        configs.append(
                            {
                                "activation": activation,
                                "model_type": model_type,
                                "clean_frac": frac,
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
                                "brightness_scale": brightness_scale,
                                "max_train_samples": max_train_samples,
                                "test_fraction": test_fraction,
                                "split_seed": split_seed,
                                "subset_seed": run_seed,
                            }
                        )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    suffix_tag = f"_{suffix}" if suffix else ""
    suffix_note = f" (suffix={suffix})" if suffix else ""
    print(
        f"[{timestamp}] Launching {len(configs)} runs with max_workers={max_workers}..."
        f"{suffix_note}"
    )

    results: list[dict] = []
    seen_fracs: set[float] = set()
    run_start_times = {}
    queue = deque(configs)
    total_runs = len(configs)
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(cpu_cores,),
    ) as executor:
        in_flight: dict = {}
        desc = f"Runs{suffix_note}"
        pbar = tqdm(total=total_runs, desc=desc, unit="run")
        while queue or in_flight:
            while queue and len(in_flight) < max_in_flight:
                cfg = queue.popleft()
                fut = executor.submit(train_and_evaluate_clean_subset, cfg)
                in_flight[fut] = cfg
                run_start_times[fut] = datetime.now()
            for future in as_completed(in_flight):
                res = future.result()
                results.append(res)
                pbar.update(1)
                frac = res["clean_frac"]
                if frac not in seen_fracs:
                    seen_fracs.add(frac)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[{timestamp}] new clean fraction encountered: {frac:.3f}")
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                elapsed_run = datetime.now() - run_start_times.get(future, datetime.now())
                print(
                    f"[{timestamp}] done: model={res['model_type']} act={res['activation']} "
                    f"clean_frac={frac:.3f} acc={res['test_accuracy']:.4f} "
                    f"elapsed={elapsed_run}"
                )
                del in_flight[future]
                break
        pbar.close()

    cache_key = build_cache_key(
        model_types=model_types,
        repeats=repeats,
        epochs=epochs,
        clean_fracs=clean_fracs,
        mlp_width=mlp_width,
        mlp_depth=mlp_depth,
        loss_type=loss_type,
        test_fraction=test_fraction,
    )
    per_run_path = os.path.join(output_dir, f"results_per_run_{cache_key}{suffix_tag}.csv")
    summary_path = os.path.join(output_dir, f"results_summary_{cache_key}{suffix_tag}.csv")
    write_csv(per_run_path, results)

    summary_rows = summarize_runs(results)
    write_csv(summary_path, summary_rows)

    pdf_path = os.path.join(output_dir, f"accuracy_vs_{cache_key}{suffix_tag}.pdf")
    write_summary_pdf(pdf_path, summary_rows)

    elapsed = datetime.now() - start_time
    runtime_log_path = os.path.join(output_dir, f"runtime_log_{os.path.splitext(os.path.basename(__file__))[0]}.csv")
    append_runtime_log(
        runtime_log_path,
        {
            "timestamp_start": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp_end": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": int(elapsed.total_seconds()),
            "script": os.path.basename(__file__),
            "cache_key": cache_key,
            "output_dir": output_dir,
            "total_runs": len(configs),
        },
    )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Saved:")
    print(f"  {per_run_path}")
    print(f"  {summary_path}")
    print(f"  {pdf_path}")
    print(f"[{timestamp}] Elapsed: {elapsed}")


if __name__ == "__main__":
    main()
