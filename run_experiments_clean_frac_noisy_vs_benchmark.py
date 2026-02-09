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
from matplotlib.lines import Line2D
from tqdm import tqdm

from nnet_models import build_model, compute_loss, evaluate, apply_corruption


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


class PartiallyCorruptedDataset(Dataset):
    def __init__(
        self,
        base: Dataset,
        clean_indices: set[int],
        corruption_mode: str,
        p: float,
        sigma: float,
    ) -> None:
        self.base = base
        self.clean_indices = clean_indices
        self.corruption_mode = corruption_mode
        self.p = p
        self.sigma = sigma

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        x, y = self.base[idx]
        if idx in self.clean_indices:
            return x, y
        x = apply_corruption(x, self.corruption_mode, self.p, self.sigma)
        return x, y


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


def _build_mnist(
    brightness_scale: float,
    max_train_samples: int | None,
    test_fraction: float,
    split_seed: int,
) -> tuple[Dataset, Dataset]:
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
    return train_base, test_base


def get_dataloaders_noisy_complement(
    batch_size: int,
    clean_frac: float,
    corruption_mode: str,
    p: float,
    sigma: float,
    num_workers: int,
    brightness_scale: float = 1.0,
    max_train_samples: int | None = None,
    test_fraction: float = 0.0,
    split_seed: int = 0,
    subset_seed: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    train_base, test_base = _build_mnist(
        brightness_scale=brightness_scale,
        max_train_samples=max_train_samples,
        test_fraction=test_fraction,
        split_seed=split_seed,
    )

    clean_frac = max(0.0, min(1.0, clean_frac))
    clean_count = int(round(clean_frac * len(train_base)))
    clean_count = max(0, min(len(train_base), clean_count))
    generator = torch.Generator().manual_seed(subset_seed)
    perm = torch.randperm(len(train_base), generator=generator).tolist()
    clean_indices = set(perm[:clean_count])
    clean_total = len(clean_indices)

    train_set = PartiallyCorruptedDataset(train_base, clean_indices, corruption_mode, p, sigma)
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


def get_dataloaders_clean_only(
    batch_size: int,
    clean_frac: float,
    num_workers: int,
    brightness_scale: float = 1.0,
    max_train_samples: int | None = None,
    test_fraction: float = 0.0,
    split_seed: int = 0,
    subset_seed: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    train_base, test_base = _build_mnist(
        brightness_scale=brightness_scale,
        max_train_samples=max_train_samples,
        test_fraction=test_fraction,
        split_seed=split_seed,
    )

    clean_frac = max(0.0, min(1.0, clean_frac))
    clean_count = int(round(clean_frac * len(train_base)))
    clean_count = max(0, min(len(train_base), clean_count))
    if clean_count == 0:
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
    grouped: dict[tuple[str, str, str, float, float], list[dict]] = {}
    for row in rows:
        key = (
            row["mode"],
            row["activation"],
            row["model_type"],
            float(row["clean_frac"]),
            float(row.get("p", 0.0)),
        )
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict] = []
    for (mode, activation, model_type, clean_frac, p_val), items in grouped.items():
        test_accs = [float(r["test_accuracy"]) for r in items]
        test_losses = [float(r["test_loss"]) for r in items]
        train_accs = [float(r["train_accuracy"]) for r in items]
        train_losses = [float(r["train_loss"]) for r in items]
        clean_total = int(items[0].get("clean_total", 0))
        train_set_size = int(items[0].get("train_set_size", 0))
        clean_frac_actual = (clean_total / train_set_size) if train_set_size else 0.0
        summary_rows.append(
            {
                "mode": mode,
                "activation": activation,
                "model_type": model_type,
                "clean_frac": clean_frac,
                "p": p_val,
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
    summary_rows.sort(key=lambda r: (r["mode"], r["model_type"], r["activation"], r["clean_frac"], r["p"]))
    return summary_rows


def compute_p_min(summary_rows: list[dict]) -> list[dict]:
    df = pd.DataFrame(summary_rows)
    noisy = df[df["mode"] == "noisy_complement"].copy()
    if noisy.empty:
        return []
    minima_rows: list[dict] = []
    for (activation, clean_frac), sub in noisy.groupby(["activation", "clean_frac"]):
        idx = sub["mean_test_accuracy"].idxmin()
        row = sub.loc[idx]
        minima_rows.append(
            {
                "activation": activation,
                "clean_frac": clean_frac,
                "p_min": float(row["p"]),
                "min_mean_test_accuracy": float(row["mean_test_accuracy"]),
            }
        )
    minima_rows.sort(key=lambda r: (r["activation"], r["clean_frac"]))
    return minima_rows


def write_comparison_grids(
    png_base_path: str,
    pdf_path: str,
    summary_rows: list[dict],
    title_prefix: str,
) -> None:
    # Lock visual encoding across all subplots.
    color_noisy = "#1f77b4"
    color_clean_subset = "#000000"
    color_full_clean = "#666666"

    df = pd.DataFrame(summary_rows)
    noisy = df[df["mode"] == "noisy_complement"].copy()
    bench = df[df["mode"] == "clean_only"].copy()
    full_clean = df[df["mode"] == "full_clean_once"].copy()
    if noisy.empty:
        return

    activations = sorted(noisy["activation"].unique())
    clean_fracs = sorted([f for f in noisy["clean_frac"].unique() if f > 0])
    if not clean_fracs:
        clean_fracs = sorted(noisy["clean_frac"].unique())
    bench_lookup = {
        (row["activation"], row["clean_frac"]): row["mean_test_accuracy"]
        for _, row in bench.iterrows()
    }
    full_clean_lookup = {
        row["activation"]: row["mean_test_accuracy"]
        for _, row in full_clean.iterrows()
    }

    png_root, png_ext = os.path.splitext(png_base_path)
    if not png_ext:
        png_ext = ".png"

    with PdfPages(pdf_path) as pdf:
        for act in activations:
            n = len(clean_fracs)
            ncols = 2
            nrows = max(1, (n + ncols - 1) // ncols)
            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(6 * ncols, 3.5 * nrows),
                sharex=True,
                sharey=False,
            )
            axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)

            for i, frac in enumerate(clean_fracs):
                r = i // ncols
                c = i % ncols
                ax = axes_arr[r][c]
                d = noisy[(noisy["activation"] == act) & (noisy["clean_frac"] == frac)].sort_values("p")
                if d.empty:
                    ax.set_visible(False)
                    continue

                ax.errorbar(
                    d["p"],
                    d["mean_test_accuracy"],
                    yerr=d["stderr_test_accuracy"],
                    marker="o",
                    color=color_noisy,
                    label="clean training subset + noisy complementary set",
                )
                bench_y = bench_lookup.get((act, frac))
                if bench_y is not None:
                    ax.axhline(
                        bench_y,
                        color=color_clean_subset,
                        linestyle="--",
                        linewidth=1.5,
                        label="clean training subset benchmark",
                    )
                full_clean_y = full_clean_lookup.get(act)
                if full_clean_y is not None:
                    ax.axhline(
                        full_clean_y,
                        color=color_full_clean,
                        linestyle=":",
                        linewidth=1.8,
                        label="entired clean dataset",
                    )
                candidates = [float(d["mean_test_accuracy"].min())]
                if bench_y is not None:
                    candidates.append(float(bench_y))
                if full_clean_y is not None:
                    candidates.append(float(full_clean_y))
                ax.set_ylim(min(candidates) - 0.1, 1.0)
                ax.set_title(f"{act} / frac={frac:.3f}")
                ax.set_xlabel("p")
                ax.set_ylabel("mean test accuracy")
                ax.grid(True, alpha=0.3)

            for j in range(len(clean_fracs), nrows * ncols):
                r = j // ncols
                c = j % ncols
                axes_arr[r][c].set_visible(False)

            legend_handles = [
                Line2D(
                    [0],
                    [0],
                    color=color_noisy,
                    marker="o",
                    linestyle="-",
                    label="clean training subset + noisy complementary set",
                ),
                Line2D(
                    [0],
                    [0],
                    color=color_clean_subset,
                    linestyle="--",
                    linewidth=1.5,
                    label="clean training subset benchmark",
                ),
                Line2D(
                    [0],
                    [0],
                    color=color_full_clean,
                    linestyle=":",
                    linewidth=1.8,
                    label="entired clean dataset",
                ),
            ]
            fig.legend(
                handles=legend_handles,
                loc="lower center",
                ncol=1,
                fontsize=9,
                frameon=True,
                bbox_to_anchor=(0.5, 0.01),
            )
            fig.suptitle(f"{title_prefix} ({act})", y=0.98)
            fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.16, wspace=0.22, hspace=0.28)
            pdf.savefig(fig, dpi=200, bbox_inches="tight")
            fig.savefig(f"{png_root}_{act}{png_ext}", dpi=200, bbox_inches="tight")
            plt.close(fig)


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


def _train_and_eval(cfg: dict) -> dict:
    torch.set_num_threads(max(1, cfg["cpu_threads"]))
    device = torch.device("cpu")
    if cfg["mode"] == "noisy_complement":
        train_loader, train_eval_loader, test_loader, clean_total = get_dataloaders_noisy_complement(
            batch_size=cfg["batch_size"],
            clean_frac=cfg["clean_frac"],
            corruption_mode=cfg["corruption_mode"],
            p=cfg["p"],
            sigma=cfg["sigma"],
            num_workers=cfg["num_workers"],
            brightness_scale=cfg["brightness_scale"],
            max_train_samples=cfg["max_train_samples"],
            test_fraction=cfg["test_fraction"],
            split_seed=cfg["split_seed"],
            subset_seed=cfg["subset_seed"],
        )
    else:
        train_loader, train_eval_loader, test_loader, clean_total = get_dataloaders_clean_only(
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

    for _ in tqdm(
        range(cfg["epochs"]),
        desc=f"epochs (w={cfg['mlp_width']}, clean={cfg['clean_frac']:.3f}, mode={cfg['mode']})",
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

    train_loss, train_acc = evaluate(model, train_eval_loader, device, cfg["loss_type"])
    test_loss, test_acc = evaluate(model, test_loader, device, cfg["loss_type"])
    train_set_size = len(train_loader.dataset)
    return {
        "mode": cfg["mode"],
        "activation": cfg["activation"],
        "model_type": cfg["model_type"],
        "clean_frac": cfg["clean_frac"],
        "p": cfg.get("p", 0.0),
        "sigma": cfg.get("sigma", 0.0),
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
    repeats_noisy: int,
    repeats_clean: int,
    repeats_full_clean: int,
    epochs: int,
    clean_fracs: list[float],
    p_values: list[float],
    mlp_width: int,
    mlp_depth: int,
    loss_type: str,
    test_fraction: float,
) -> str:
    model_tag = "-".join(model_types)
    f_min, f_max = min(clean_fracs), max(clean_fracs)
    p_min, p_max = min(p_values), max(p_values)
    return (
        f"{model_tag}_rn{repeats_noisy}_rc{repeats_clean}_rfc{repeats_full_clean}_"
        f"e{epochs}_tf{test_fraction:.3f}_"
        f"clean{f_min:.3f}-{f_max:.3f}_p{p_min:.2f}-{p_max:.2f}_w{mlp_width}_d{mlp_depth}_"
        f"loss-{loss_type}"
    )


def main() -> None:
    start_time = datetime.now()
    activations = ["relu", "tanh", "linear", "quadratic"]
    model_types = ["mlp"]
    clean_fracs = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]
    corruption_mode = "replacement"  # options: "replacement", "additive"
    p_values = np.linspace(0.0, 1.0, 21).tolist()
    sigma = 0.0
    repeats_noisy = 100
    repeats_clean = 100
    repeats_full_clean = 1
    epochs = 20
    batch_size = 128
    learning_rate = 1e-3
    weight_decay = 0.0
    loss_type = "cross_entropy"
    mlp_width = 64
    mlp_depth = 1
    test_fraction = 1 / 2
    split_seed = 1234
    cpu_max = 50
    max_workers = cpu_max
    max_in_flight = max(1, 9*cpu_max//10)
    data_workers = 0
    cpu_threads_per_worker = 1
    cpu_cores = list(range(cpu_max))
    brightness_scale = 1.0
    output_dir = "results"
    suffix = "clean_frac_noisy_vs_benchmark"
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
            for clean_frac in clean_fracs:
                mlp_hidden_sizes = [mlp_width] * mlp_depth
                # Noisy complementary set runs
                for p in p_values:
                    for _ in range(repeats_noisy):
                        run_seed = random.randint(1, 1_000_000_000)
                        configs.append(
                            {
                                "mode": "noisy_complement",
                                "activation": activation,
                                "model_type": model_type,
                                "clean_frac": clean_frac,
                                "corruption_mode": corruption_mode,
                                "p": p,
                                "sigma": sigma,
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
                # Clean-only benchmark runs
                for _ in range(repeats_clean):
                    run_seed = random.randint(1, 1_000_000_000)
                    configs.append(
                        {
                            "mode": "clean_only",
                            "activation": activation,
                            "model_type": model_type,
                            "clean_frac": clean_frac,
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
            # Full clean dataset benchmark (single run per activation/model)
            for _ in range(repeats_full_clean):
                run_seed = random.randint(1, 1_000_000_000)
                configs.append(
                    {
                        "mode": "full_clean_once",
                        "activation": activation,
                        "model_type": model_type,
                        "clean_frac": 1.0,
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
                fut = executor.submit(_train_and_eval, cfg)
                in_flight[fut] = cfg
                run_start_times[fut] = datetime.now()
            for future in as_completed(in_flight):
                res = future.result()
                results.append(res)
                pbar.update(1)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                elapsed_run = datetime.now() - run_start_times.get(future, datetime.now())
                if res["mode"] == "noisy_complement":
                    print(
                        f"[{timestamp}] done: mode={res['mode']} act={res['activation']} "
                        f"clean_frac={res['clean_frac']:.3f} p={res['p']:.3f} "
                        f"acc={res['test_accuracy']:.4f} elapsed={elapsed_run}"
                    )
                else:
                    print(
                        f"[{timestamp}] done: mode={res['mode']} act={res['activation']} "
                        f"clean_frac={res['clean_frac']:.3f} acc={res['test_accuracy']:.4f} "
                        f"elapsed={elapsed_run}"
                    )
                del in_flight[future]
                break
        pbar.close()

    cache_key = build_cache_key(
        model_types=model_types,
        repeats_noisy=repeats_noisy,
        repeats_clean=repeats_clean,
        repeats_full_clean=repeats_full_clean,
        epochs=epochs,
        clean_fracs=clean_fracs,
        p_values=p_values,
        mlp_width=mlp_width,
        mlp_depth=mlp_depth,
        loss_type=loss_type,
        test_fraction=test_fraction,
    )
    per_run_path = os.path.join(output_dir, f"results_per_run_{cache_key}{suffix_tag}.csv")
    summary_path = os.path.join(output_dir, f"results_summary_{cache_key}{suffix_tag}.csv")
    minima_path = os.path.join(output_dir, f"p_min_{cache_key}{suffix_tag}.csv")
    plot_path_png_base = os.path.join(output_dir, f"comparison_grid_{cache_key}{suffix_tag}.png")
    plot_path_pdf = os.path.join(output_dir, f"comparison_grid_{cache_key}{suffix_tag}.pdf")
    write_csv(per_run_path, results)

    summary_rows = summarize_runs(results)
    write_csv(summary_path, summary_rows)

    minima_rows = compute_p_min(summary_rows)
    write_csv(minima_path, minima_rows)

    write_comparison_grids(
        plot_path_png_base,
        plot_path_pdf,
        summary_rows,
        title_prefix="Clean subset + noisy complementary set vs clean-only benchmark",
    )

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
    print(f"  {minima_path}")
    print(f"  {plot_path_pdf}")
    for act in activations:
        print(f"  {os.path.splitext(plot_path_png_base)[0]}_{act}.png")
    print(f"[{timestamp}] Elapsed: {elapsed}")


if __name__ == "__main__":
    main()
