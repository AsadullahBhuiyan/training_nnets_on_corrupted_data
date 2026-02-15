import os

import csv
import random
import statistics
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jax.numpy as jnp
from neural_tangents import stax
import torch
from torchvision import datasets, transforms
from tqdm import tqdm


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


def _limit_dataset(base, max_samples: int | None):
    if max_samples is None:
        return base
    max_samples = max(1, min(len(base), max_samples))
    return torch.utils.data.Subset(base, list(range(max_samples)))


def _split_dataset(base, test_fraction: float, seed: int):
    if test_fraction <= 0.0:
        return base, torch.utils.data.Subset(base, [])
    if test_fraction >= 1.0:
        return torch.utils.data.Subset(base, []), base
    generator = torch.Generator().manual_seed(seed)
    test_size = int(round(len(base) * test_fraction))
    test_size = max(1, min(len(base) - 1, test_size))
    train_size = len(base) - test_size
    return torch.utils.data.random_split(base, [train_size, test_size], generator=generator)


def load_mnist_subsets(
    n_train: int,
    n_test: int,
    seed: int,
    max_train_samples: int | None = None,
    test_fraction: float = 0.0,
    split_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_ds = _limit_dataset(train_ds, max_train_samples)
    if test_fraction > 0.0:
        train_ds, extra_test = _split_dataset(train_ds, test_fraction=test_fraction, seed=split_seed)
        test_ds = torch.utils.data.ConcatDataset([test_ds, extra_test])

    rng = np.random.default_rng(seed)
    n_train_eff = min(n_train, len(train_ds))
    n_test_eff = min(n_test, len(test_ds))
    train_idx = rng.choice(len(train_ds), size=n_train_eff, replace=False)
    test_idx = rng.choice(len(test_ds), size=n_test_eff, replace=False)

    x_train = np.stack([train_ds[i][0].numpy().reshape(-1) for i in train_idx]).astype(np.float32)
    y_train = np.array([train_ds[i][1] for i in train_idx], dtype=np.int32)
    x_test = np.stack([test_ds[i][0].numpy().reshape(-1) for i in test_idx]).astype(np.float32)
    y_test = np.array([test_ds[i][1] for i in test_idx], dtype=np.int32)
    return x_train, y_train, x_test, y_test


def corrupt_replacement(x: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    mask = rng.random(x.shape, dtype=np.float32) < p
    repl = rng.random(x.shape, dtype=np.float32)
    return np.where(mask, repl, x).astype(np.float32)


def ntk_predict_scores(
    kernel_fn,
    x_train_corr: np.ndarray,
    y_train_onehot: np.ndarray,
    x_test_clean: np.ndarray,
    ridge: float,
) -> np.ndarray:
    k_tt = np.asarray(kernel_fn(jnp.asarray(x_train_corr), None, get="ntk"), dtype=np.float64)
    k_xt = np.asarray(
        kernel_fn(jnp.asarray(x_test_clean), jnp.asarray(x_train_corr), get="ntk"),
        dtype=np.float64,
    )
    n = k_tt.shape[0]
    reg = ridge * (np.trace(k_tt) / max(n, 1))
    a = k_tt + reg * np.eye(n, dtype=np.float64)
    alpha = np.linalg.solve(a, y_train_onehot.astype(np.float64))
    scores = k_xt @ alpha
    return scores


def build_cache_key(
    repeats: int,
    epochs_label: str,
    n_train: int,
    n_test: int,
    p_values: list[float],
    ridge: float,
    w_std: float,
    b_std: float,
) -> str:
    p_min = min(p_values)
    p_max = max(p_values)
    return (
        f"ntk_relu_r{repeats}_{epochs_label}_ntr{n_train}_nte{n_test}_"
        f"p{p_min:.2f}-{p_max:.2f}_ridge{ridge:.1e}_wstd{w_std:.3f}_bstd{b_std:.3f}"
    )


def main() -> None:
    start_time = datetime.now()
    # Manual configuration.
    num_realizations = 100
    p_values = np.linspace(0.95, 1.0, 101).tolist()
    n_train = 1024
    n_test = 1024
    ridge = 1e-6
    w_std = 1.0
    b_std = 1.0
    max_train_samples = None
    test_fraction = 0.0
    split_seed = 1234
    output_dir = "results"
    suffix = "infinite_width_relu_ntk"
    seed = 1234

    # CPU controls (same style as other experiment scripts).
    cpu_max = 30
    cpu_threads = 1
    cpu_cores = list(range(50, 50 + cpu_max))

    os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault(
        "XLA_FLAGS",
        f"--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={cpu_threads}",
    )
    apply_cpu_affinity(cpu_cores)

    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)
    random.seed(seed)

    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
        f"Loading MNIST subsets (n_train={n_train}, n_test={n_test})"
    )
    x_train, y_train, x_test, y_test = load_mnist_subsets(
        n_train=n_train,
        n_test=n_test,
        seed=seed,
        max_train_samples=max_train_samples,
        test_fraction=test_fraction,
        split_seed=split_seed,
    )
    y_train_onehot = np.eye(10, dtype=np.float32)[y_train]

    rows_per_realization: list[dict] = []
    rows_summary: list[dict] = []
    seed_rng = np.random.default_rng(seed)
    _, _, kernel_fn = stax.serial(
        stax.Dense(1024, W_std=w_std, b_std=b_std),
        stax.Relu(),
        stax.Dense(1, W_std=w_std, b_std=b_std),
    )

    pbar_p = tqdm(p_values, desc="p sweep", unit="p")
    for p in pbar_p:
        accs: list[float] = []
        seed_list = seed_rng.integers(1, 1_000_000_000, size=num_realizations).tolist()
        pbar_r = tqdm(
            seed_list,
            total=num_realizations,
            desc=f"realizations @ p={p:.3f}",
            unit="run",
            leave=False,
        )

        for local_seed in pbar_r:
            rng = np.random.default_rng(int(local_seed))
            x_train_corr = corrupt_replacement(x_train, p=float(p), rng=rng)
            scores = ntk_predict_scores(
                kernel_fn=kernel_fn,
                x_train_corr=x_train_corr,
                y_train_onehot=y_train_onehot,
                x_test_clean=x_test,
                ridge=ridge,
            )
            y_pred = np.argmax(scores, axis=1)
            acc = float(np.mean(y_pred == y_test))
            rows_per_realization.append(
                {
                    "p": float(p),
                    "seed": int(local_seed),
                    "test_accuracy": acc,
                    "w_std": w_std,
                    "b_std": b_std,
                }
            )
            accs.append(acc)

        pbar_r.close()
        mean_acc = statistics.mean(accs)
        std_acc = statistics.pstdev(accs) if len(accs) > 1 else 0.0
        stderr_acc = std_acc / (len(accs) ** 0.5) if len(accs) > 1 else 0.0
        rows_summary.append(
            {
                "p": float(p),
                "repeats": len(accs),
                "mean_test_accuracy": mean_acc,
                "std_test_accuracy": std_acc,
                "stderr_test_accuracy": stderr_acc,
                "n_train": n_train,
                "n_test": n_test,
                "ridge": ridge,
                "w_std": w_std,
                "b_std": b_std,
            }
        )
        pbar_p.set_postfix(mean_acc=f"{mean_acc:.4f}", stderr=f"{stderr_acc:.4f}")

    cache_key = build_cache_key(
        repeats=num_realizations,
        epochs_label="na",
        n_train=n_train,
        n_test=n_test,
        p_values=p_values,
        ridge=ridge,
        w_std=w_std,
        b_std=b_std,
    )
    suffix_tag = f"_{suffix}" if suffix else ""
    per_run_path = os.path.join(output_dir, f"results_per_run_{cache_key}{suffix_tag}.csv")
    summary_path = os.path.join(output_dir, f"results_summary_{cache_key}{suffix_tag}.csv")
    plot_path = os.path.join(output_dir, f"accuracy_vs_p_{cache_key}{suffix_tag}.pdf")
    write_csv(per_run_path, rows_per_realization)
    write_csv(summary_path, rows_summary)

    summary_df = pd.DataFrame(rows_summary).sort_values("p")
    plt.figure(figsize=(6, 4))
    plt.errorbar(
        summary_df["p"],
        summary_df["mean_test_accuracy"],
        yerr=summary_df["stderr_test_accuracy"],
        marker=".",
        color="#1f77b4",
    )
    plt.xlabel("corruption probability p")
    plt.ylabel("mean test accuracy (clean test set)")
    plt.title(
        f"Infinite-width ReLU NTK (n_train={n_train}, n_test={n_test}, repeats={num_realizations})"
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()

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
            "total_runs": len(p_values) * num_realizations,
            "n_train": n_train,
            "n_test": n_test,
            "w_std": w_std,
            "b_std": b_std,
            "cpu_threads": cpu_threads,
            "cpu_cores": cpu_cores,
            "max_train_samples": max_train_samples,
            "test_fraction": test_fraction,
            "split_seed": split_seed,
        },
    )

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Saved:")
    print(f"  {per_run_path}")
    print(f"  {summary_path}")
    print(f"  {plot_path}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Elapsed: {elapsed}")


if __name__ == "__main__":
    main()
