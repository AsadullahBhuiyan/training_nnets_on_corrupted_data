import os
cpu_max = 30

import csv
import random
import statistics
import math
import importlib
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import multiprocessing as mp
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

from nnet_models import TrainConfig, train_and_evaluate


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_runs(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str, float, float, int], list[dict]] = {}
    for row in rows:
        n_train_samples = int(row.get("n_train_samples", -1))
        key = (
            row["activation"],
            row["model_type"],
            row["corruption_mode"],
            float(row["p"]),
            float(row["sigma"]),
            n_train_samples,
        )
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict] = []
    for (activation, model_type, corruption_mode, p, sigma, n_train_samples), items in grouped.items():
        accs = [float(r["test_accuracy"]) for r in items]
        losses = [float(r["test_loss"]) for r in items]
        train_losses = [float(r.get("train_loss", 0.0)) for r in items]
        summary_rows.append(
            {
                "activation": activation,
                "model_type": model_type,
                "corruption_mode": corruption_mode,
                "p": p,
                "sigma": sigma,
                "n_train_samples": n_train_samples,
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
                "mean_train_loss": statistics.mean(train_losses),
                "std_train_loss": statistics.pstdev(train_losses) if len(train_losses) > 1 else 0.0,
                "stderr_train_loss": (statistics.pstdev(train_losses) / (len(items) ** 0.5))
                if len(items) > 1
                else 0.0,
            }
        )
    summary_rows.sort(
        key=lambda r: (
            r["model_type"],
            r["activation"],
            r["corruption_mode"],
            r["n_train_samples"],
            r["p"],
            r["sigma"],
        )
    )
    return summary_rows


def append_runtime_log(path: str, row: dict) -> None:
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _update_running_stats(stats: dict, x: float) -> None:
    n = stats["n"] + 1
    delta = x - stats["mean"]
    mean = stats["mean"] + delta / n
    delta2 = x - mean
    m2 = stats["m2"] + delta * delta2
    stats["n"] = n
    stats["mean"] = mean
    stats["m2"] = m2


def _stderr_from_stats(stats: dict) -> float:
    n = stats["n"]
    if n < 2:
        return float("inf")
    var = stats["m2"] / n  # population variance
    return math.sqrt(max(var, 0.0)) / math.sqrt(n)


def _import_fssa_with_compat():
    # pyfssa relies on scipy.optimize.optimize internals that changed across scipy versions.
    if not hasattr(np, "int"):
        np.int = int
    try:
        import scipy.optimize.optimize as _opt_mod

        _opt_mod = importlib.reload(_opt_mod)
        from scipy.optimize import _optimize as _opt

        for name in ("OptimizeResult", "_status_message", "wrap_function"):
            if not hasattr(_opt_mod, name) and hasattr(_opt, name):
                setattr(_opt_mod, name, getattr(_opt, name))
        if not hasattr(_opt_mod, "wrap_function"):
            def _wrap_function(function, args):
                ncalls = [0]

                def function_wrapper(x):
                    ncalls[0] += 1
                    return function(x, *args)

                return ncalls, function_wrapper

            _opt_mod.wrap_function = _wrap_function
    except Exception:
        pass
    import fssa

    return fssa


def find_pyfssa_collapse_zeta0(
    act_rows: list[dict],
    sweep_label: str,
    min_n_train: int = 10_000,
) -> tuple[dict | None, plt.Figure | None]:
    if sweep_label != "p":
        return None, None

    sub = [r for r in act_rows if int(r["n_train_samples"]) >= min_n_train]
    if not sub:
        return {"error": f"no rows after n_train_samples >= {min_n_train}"}, None

    n_sizes = sorted({int(r["n_train_samples"]) for r in sub})
    if len(n_sizes) < 2:
        return {"error": f"need >=2 n_train values, found {n_sizes}"}, None

    common_p = None
    for n_train in n_sizes:
        pset = {float(r["p"]) for r in sub if int(r["n_train_samples"]) == n_train}
        common_p = pset if common_p is None else (common_p & pset)
    ps = sorted(common_p) if common_p else []
    if len(ps) < 2:
        return {"error": "need >=2 common p values across n_train"}, None

    ps_set = set(ps)
    sub = [r for r in sub if float(r["p"]) in ps_set]
    L = np.array(n_sizes, dtype=float)
    rho = np.array(ps, dtype=float)
    a = np.zeros((len(L), len(rho)))
    da = np.zeros_like(a)
    for i, n_train in enumerate(n_sizes):
        d = {float(r["p"]): r for r in sub if int(r["n_train_samples"]) == n_train}
        for j, p in enumerate(ps):
            row = d[p]
            a[i, j] = float(row["mean_test_accuracy"])
            da[i, j] = float(row["stderr_test_accuracy"])

    min_nonzero = da[da > 0].min() if np.any(da > 0) else 1e-6
    da = np.where(np.isfinite(da), da, min_nonzero)
    da = np.clip(da, 1e-6, None)

    try:
        fssa = _import_fssa_with_compat()
    except Exception as exc:
        return {"error": f"fssa import failed: {exc}"}, None

    zeta_fixed = 0.0
    rho_grid = np.linspace(float(rho.min()) + 1e-3, float(rho.max()) - 1e-3, 40)
    nu_grid = np.linspace(0.2, 10.0, 60)
    best = (float("inf"), math.nan, math.nan)
    for rho_c in rho_grid:
        for nu in nu_grid:
            scaled = fssa.scaledata(L, rho, a, da, rho_c, nu, zeta_fixed)
            s_val = fssa.quality(scaled.x, scaled.y, scaled.dy)
            if s_val < best[0]:
                best = (float(s_val), float(rho_c), float(nu))
    s_star, rho_c_star, nu_star = best
    if not np.isfinite(s_star):
        return {"error": "failed to find finite pyfssa quality"}, None

    scaled = fssa.scaledata(L, rho, a, da, rho_c_star, nu_star, zeta_fixed)
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    for i, n_train in enumerate(n_sizes):
        ax.plot(scaled.x[i], scaled.y[i], marker=".", linestyle="", label=f"n={n_train}")
    ax.set_title(f"pyfssa zeta=0 (S={s_star:.3f})")
    ax.set_xlabel(rf"$(p-p^*)n^{{1/\nu}}$, $p^*={rho_c_star:.4f}$, $\nu={nu_star:.3f}$")
    ax.set_ylabel(r"$\overline{A}(p)-\overline{A}(p^*)$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    plt.tight_layout()
    result = {
        "rho_c": rho_c_star,
        "nu": nu_star,
        "zeta": zeta_fixed,
        "S": s_star,
        "min_n_train": int(min_n_train),
        "n_train_values": [int(v) for v in n_sizes],
    }
    return result, fig


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
        import torch

        torch.set_num_interop_threads(1)
        torch.set_num_threads(1)
    except Exception:
        pass


def write_summary_pdf(
    path: str,
    summary_rows: list[dict],
    metadata: dict,
) -> None:
    if not summary_rows:
        return
    model_types = sorted({r["model_type"] for r in summary_rows})

    with PdfPages(path) as pdf:
        pyfssa_all_results: dict[str, dict] = {}
        for model_type in model_types:
            sub = [r for r in summary_rows if r["model_type"] == model_type]
            activations = sorted({r["activation"] for r in sub})
            n_train_values = sorted({int(r["n_train_samples"]) for r in sub})
            corruption_modes = {r["corruption_mode"] for r in sub}
            only_mode = next(iter(corruption_modes)) if len(corruption_modes) == 1 else None
            sweep_label = "sigma" if only_mode == "additive" else "p"
            n = len(activations)

            fig, axes = plt.subplots(1, n, figsize=(4 * n, 3), sharey=True)
            if n == 1:
                axes = [axes]
            for ax, act in zip(axes, activations):
                for n_train in n_train_values:
                    d = sorted(
                        [
                            r
                            for r in sub
                            if r["activation"] == act and int(r["n_train_samples"]) == n_train
                        ],
                        key=lambda r: float(r[sweep_label]),
                    )
                    if not d:
                        continue
                    ax.errorbar(
                        [float(r[sweep_label]) for r in d],
                        [float(r["mean_test_accuracy"]) for r in d],
                        yerr=[float(r["stderr_test_accuracy"]) for r in d],
                        marker=".",
                        label=f"n={n_train}",
                    )
                ax.set_title(f"{model_type} / {act}")
                ax.set_xlabel(sweep_label)
                ax.grid(True, alpha=0.3)
            axes[0].set_ylabel("mean test accuracy")
            axes[0].legend(fontsize=8)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            fig, axes = plt.subplots(1, n, figsize=(4 * n, 3), sharey=True)
            if n == 1:
                axes = [axes]
            for ax, act in zip(axes, activations):
                for n_train in n_train_values:
                    d = sorted(
                        [
                            r
                            for r in sub
                            if r["activation"] == act and int(r["n_train_samples"]) == n_train
                        ],
                        key=lambda r: float(r[sweep_label]),
                    )
                    if not d:
                        continue
                    ax.plot(
                        [float(r[sweep_label]) for r in d],
                        [float(r["std_test_accuracy"]) for r in d],
                        marker=".",
                        label=f"n={n_train}",
                    )
                ax.set_title(f"{model_type} / {act}")
                ax.set_xlabel(sweep_label)
                ax.grid(True, alpha=0.3)
            axes[0].set_ylabel("std test accuracy")
            axes[0].legend(fontsize=8)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            fig, axes = plt.subplots(1, n, figsize=(4 * n, 3), sharey=True)
            if n == 1:
                axes = [axes]
            for ax, act in zip(axes, activations):
                for n_train in n_train_values:
                    d = sorted(
                        [
                            r
                            for r in sub
                            if r["activation"] == act and int(r["n_train_samples"]) == n_train
                        ],
                        key=lambda r: float(r[sweep_label]),
                    )
                    if not d:
                        continue
                    ax.errorbar(
                        [float(r[sweep_label]) for r in d],
                        [float(r["mean_test_loss"]) for r in d],
                        yerr=[float(r["stderr_test_loss"]) for r in d],
                        marker=".",
                        label=f"n={n_train}",
                    )
                ax.set_title(f"{model_type} / {act}")
                ax.set_xlabel(sweep_label)
                ax.grid(True, alpha=0.3)
            axes[0].set_ylabel("mean test loss")
            axes[0].legend(fontsize=8)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            fig, axes = plt.subplots(1, n, figsize=(4 * n, 3), sharey=True)
            if n == 1:
                axes = [axes]
            for ax, act in zip(axes, activations):
                for n_train in n_train_values:
                    d = sorted(
                        [
                            r
                            for r in sub
                            if r["activation"] == act and int(r["n_train_samples"]) == n_train
                        ],
                        key=lambda r: float(r[sweep_label]),
                    )
                    if not d:
                        continue
                    ax.errorbar(
                        [float(r[sweep_label]) for r in d],
                        [float(r["mean_train_loss"]) for r in d],
                        yerr=[float(r["stderr_train_loss"]) for r in d],
                        marker=".",
                        label=f"n={n_train}",
                    )
                ax.set_title(f"{model_type} / {act}")
                ax.set_xlabel(sweep_label)
                ax.grid(True, alpha=0.3)
            axes[0].set_ylabel("mean train loss")
            axes[0].legend(fontsize=8)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            pyfssa_results = {}
            for act in activations:
                act_rows = [r for r in sub if r["activation"] == act]
                pyfssa_result, pyfssa_fig = find_pyfssa_collapse_zeta0(
                    act_rows,
                    sweep_label=sweep_label,
                    min_n_train=int(metadata.get("pyfssa_min_n_train", 10_000)),
                )
                if pyfssa_result is not None:
                    pyfssa_results[act] = pyfssa_result
                if pyfssa_fig is not None:
                    pdf.savefig(pyfssa_fig)
                    plt.close(pyfssa_fig)
            pyfssa_all_results[model_type] = pyfssa_results

            # Mean test accuracy vs n_train for selected p values (nearest available).
            requested_ps = [0.0, 0.5, 0.8, 0.9, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.00]
            available_ps = sorted({float(r["p"]) for r in sub})

            def _nearest_p(target: float) -> float | None:
                ordered = sorted(available_ps, key=lambda q: abs(q - target))
                for q in ordered:
                    if abs(q - 0.95) > 1e-12:
                        return q
                return None

            mapped_ps: list[float] = []
            for p in requested_ps:
                nearest = _nearest_p(p)
                if nearest is None:
                    continue
                if nearest not in mapped_ps:
                    mapped_ps.append(nearest)

            if mapped_ps:
                fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
                for p in mapped_ps:
                    d = sorted(
                        [r for r in sub if float(r["p"]) == p],
                        key=lambda r: int(r["n_train_samples"]),
                    )
                    if not d:
                        continue
                    ntr = [int(r["n_train_samples"]) for r in d]
                    acc = [float(r["mean_test_accuracy"]) for r in d]
                    ax.plot(ntr, acc, marker=".", label=f"p={p:.2f}")
                act_label = activations[0] if len(activations) == 1 else "all"
                ax.set_title(f"{model_type} / {act_label} (nearest p slices)")
                ax.set_xlabel("n_train")
                ax.set_ylabel("mean test accuracy")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8, ncol=2)
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
        metadata["pyfssa_zeta0_results"] = pyfssa_all_results

        meta_lines = [f"{key}: {value}" for key, value in metadata.items()]
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.05, 0.95, "Run Metadata", fontsize=14, fontweight="bold", va="top")
        fig.text(0.05, 0.9, "\n".join(meta_lines), fontsize=10, va="top")
        plt.axis("off")
        pdf.savefig(fig)
        plt.close(fig)


def build_cache_key(
    max_repeats: int,
    epochs: int,
    exact_sample_counts: bool,
    n_train_values: list[int],
    dataset_name: str,
    dataset_dim: int | None,
    infimnist_cache_dir: str | None,
    loss_type: str,
    ps: list[float],
    width: int,
    mlp_depth: int,
) -> str:
    p_min, p_max = min(ps), max(ps)
    strength_tag = f"p{p_min:.2f}-{p_max:.2f}"
    n_min, n_max = min(n_train_values), max(n_train_values)
    if exact_sample_counts:
        split_tag = f"ntr{n_min}-{n_max}_nte{n_min}-{n_max}"
    else:
        split_tag = "varsplit"
    if dataset_name == "infimnist":
        cache_tag = ""
        if infimnist_cache_dir:
            cache_tag = f"_cache-{os.path.basename(os.path.normpath(infimnist_cache_dir))}"
        dataset_tag = (
            f"{dataset_name}{int(dataset_dim)}{cache_tag}"
            if dataset_dim is not None
            else f"{dataset_name}{cache_tag}"
        )
    else:
        dataset_tag = dataset_name
    return (
        f"mlpns_{dataset_tag}_rmax{max_repeats}_e{epochs}_{split_tag}_"
        f"{strength_tag}_w{width}_d{mlp_depth}_loss-{loss_type}"
    )


def main() -> None:
    start_time = datetime.now()
    # Manual configuration (edit these values directly).
    activations = ["relu"]
    model_types = ["mlp"]  # n-train sweep is MLP-only
    corruption_mode = "replacement"
    ps = np.linspace(0.9, 1.0, 51)
    width = 512
    n_train_values = [10000, 50000, 100000, 500000, 1000000, 5000000, 10000000]
    mlp_depth = 1
    max_repeats = 100
    min_repeats = 10
    stderr_target = 1e-3
    epochs = 20
    batch_size = 128
    learning_rate = 1e-3
    weight_decay = 0.0
    loss_type = "quadratic"  # options: "cross_entropy", "quadratic"
    max_workers = cpu_max
    max_in_flight = 9 * cpu_max // 10
    serial_only = False
    data_workers = 0
    cpu_threads_per_worker = 1
    cpu_cores = list(range(31, 31 + cpu_max))  # Example: [0, 1, 2, 3] to pin processes.
    brightness_scale = 1.0
    dataset_name = "infimnist"  # options: "mnist", "infimnist"
    dataset_dim = 2e7
    infimnist_cache_dir = "data/infimnist_cache_2e7"
    pyfssa_min_n_train = 10_000
    split_seed = 1234
    exact_sample_counts = True
    output_dir = "results"
    suffix = "ntrain_sweep_relu_only_infimnist"
    seed = 1234
    max_train_samples = None
    use_cuda = False

    n_train_values = [int(v) for v in n_train_values]

    if use_cuda and max_workers > 1:
        raise ValueError("use_cuda=True only supports max_workers=1 for now.")
    if not exact_sample_counts:
        raise ValueError("This script requires exact_sample_counts=True.")
    if not n_train_values:
        raise ValueError("Provide at least one n_train value.")
    if corruption_mode != "replacement":
        raise ValueError("This script currently supports corruption_mode='replacement' only.")
    if serial_only:
        max_workers = 1
        max_in_flight = 1
    if dataset_name == "infimnist":
        if dataset_dim is None or dataset_dim <= 1:
            raise ValueError("Set dataset_dim > 1 for infimnist.")
        if infimnist_cache_dir and not os.path.isdir(infimnist_cache_dir):
            raise FileNotFoundError(
                f"infimnist_cache_dir does not exist: {infimnist_cache_dir}. "
                "Run generate_infimnist_cache.py first."
            )
        for n_train in n_train_values:
            if n_train <= 0:
                raise ValueError("All n_train values must be positive.")
            if 2 * n_train > dataset_dim:
                raise ValueError(
                    f"Need 2*n_train <= dataset_dim for disjoint n_train=n_test splits: "
                    f"n_train={n_train}, dataset_dim={dataset_dim}."
                )

    os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads_per_worker))
    os.environ.setdefault("MKL_NUM_THREADS", str(cpu_threads_per_worker))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(cpu_threads_per_worker))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(cpu_threads_per_worker))

    apply_cpu_affinity(cpu_cores)
    max_in_flight = max(1, min(max_in_flight, max_workers))
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    pool_start_method = "spawn" if dataset_name == "infimnist" and not serial_only else None
    if dataset_name == "infimnist" and not serial_only:
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"Using multiprocessing start method='{pool_start_method}' for infimnist."
        )

    sweep_values = ps
    sweep_label = "p"

    group_keys: list[tuple[str, str, int, float, float]] = []
    group_max_repeats: dict[tuple[str, str, int, float, float], int] = {}
    group_min_repeats: dict[tuple[str, str, int, float, float], int] = {}
    for activation in activations:
        for model_type in model_types:
            for n_train in n_train_values:
                for value in sweep_values:
                    p = float(value)
                    sigma = 0.0
                    key = (activation, model_type, n_train, p, sigma)
                    group_keys.append(key)
                    if abs(p) < 1e-12:
                        group_max_repeats[key] = 1
                        group_min_repeats[key] = 1
                    else:
                        group_max_repeats[key] = max_repeats
                        group_min_repeats[key] = min_repeats

    def _make_config(key: tuple[str, str, int, float, float]) -> TrainConfig:
        activation, model_type, n_train, p, sigma = key
        run_seed = random.randint(1, 1_000_000_000)
        return TrainConfig(
            activation=activation,
            model_type=model_type,
            corruption_mode=corruption_mode,
            p=p,
            sigma=sigma,
            mlp_hidden_sizes=[width] * mlp_depth,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            loss_type=loss_type,
            seed=run_seed,
            num_workers=data_workers,
            cpu_threads=cpu_threads_per_worker,
            brightness_scale=brightness_scale,
            split_seed=run_seed,
            dataset_name=dataset_name,
            dataset_dim=dataset_dim,
            infimnist_cache_dir=infimnist_cache_dir,
            exact_sample_counts=exact_sample_counts,
            exact_train_samples=n_train,
            exact_test_samples=n_train,
            max_train_samples=max_train_samples,
            use_cuda=use_cuda,
        )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    suffix_tag = f"_{suffix}" if suffix else ""
    suffix_note = f" (suffix={suffix})" if suffix else ""
    print(
        f"[{timestamp}] Launching {len(group_keys)} groups with max_workers={max_workers} "
        f"(stderr_target={stderr_target}, min_repeats={min_repeats}, max_repeats={max_repeats})..."
        f"{suffix_note} serial_only={serial_only}"
    )

    results: list[dict] = []
    seen_values: set[float] = set()

    def _run_chunk(
        chunk_configs: list[tuple[str, str, int, float, float]],
        chunk_workers: int,
        chunk_max_in_flight: int,
        chunk_label: str,
    ) -> None:
        if not chunk_configs:
            return
        run_start_times: dict = {}
        queue = deque()
        scheduled_per_n_train: dict[int, int] = {n: 0 for n in n_train_values}
        completed_per_n_train: dict[int, int] = {n: 0 for n in n_train_values}
        scheduled_per_group: dict = {k: 0 for k in chunk_configs}
        stats_by_group: dict = {k: {"n": 0, "mean": 0.0, "m2": 0.0} for k in chunk_configs}
        ordered_n_train = sorted(n_train_values)
        total_runs = 0

        def _done_by_n_train() -> str:
            return " ".join(
                f"{n}:{completed_per_n_train[n]}/{scheduled_per_n_train[n]}"
                for n in ordered_n_train
            )

        desc = f"Runs{suffix_note} [{chunk_label}]"
        pbar = tqdm(total=0, desc=desc, unit="run")

        def _enqueue(key: tuple[str, str, int, float, float]) -> None:
            nonlocal total_runs
            cfg = _make_config(key)
            queue.append((key, cfg))
            n_train = int(cfg.exact_train_samples or -1)
            scheduled_per_n_train[n_train] += 1
            scheduled_per_group[key] += 1
            total_runs += 1
            pbar.total = total_runs
            pbar.refresh()

        for key in chunk_configs:
            _enqueue(key)

        if serial_only:
            _init_worker(cpu_cores)
            current_n = -1

            def _update_serial_postfix() -> None:
                active_str = str(current_n) if current_n >= 0 else "-"
                pbar.set_postfix_str(f"n_train={active_str} done={_done_by_n_train()}")

            _update_serial_postfix()
            while queue:
                key, cfg = queue.popleft()
                current_n = int(cfg.exact_train_samples or -1)
                run_start = datetime.now()
                _update_serial_postfix()
                res = train_and_evaluate(cfg)
                n_train = int(cfg.exact_train_samples or -1)
                n_test = int(cfg.exact_test_samples or -1)
                res["n_train_samples"] = n_train
                res["n_test_samples"] = n_test
                results.append(res)
                completed_per_n_train[n_train] += 1
                _update_running_stats(stats_by_group[key], float(res["test_accuracy"]))
                pbar.update(1)
                strength_value = res["p"] if corruption_mode == "replacement" else res["sigma"]
                if strength_value not in seen_values:
                    seen_values.add(strength_value)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[{timestamp}] new {sweep_label} encountered: {strength_value:.3f}")
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                width_label = res["mlp_hidden_sizes"][0] if res["mlp_hidden_sizes"] else -1
                elapsed = datetime.now() - run_start
                print(
                    f"[{timestamp}] done: model={res['model_type']} act={res['activation']} "
                    f"{sweep_label}={strength_value:.3f} n_train={n_train} width={width_label} "
                    f"acc={res['test_accuracy']:.4f} "
                    f"elapsed={elapsed}"
                )
                stderr = _stderr_from_stats(stats_by_group[key])
                n_done = stats_by_group[key]["n"]
                if n_done < group_min_repeats[key]:
                    _enqueue(key)
                elif n_done < group_max_repeats[key] and stderr > stderr_target:
                    _enqueue(key)
                _update_serial_postfix()
            pbar.close()
            return

        executor_kwargs = dict(
            max_workers=chunk_workers,
            initializer=_init_worker,
            initargs=(cpu_cores,),
        )
        if pool_start_method:
            executor_kwargs["mp_context"] = mp.get_context(pool_start_method)
        with ProcessPoolExecutor(**executor_kwargs) as executor:
            in_flight: dict = {}

            def _update_pbar_postfix() -> None:
                active_n = sorted(
                    {int(cfg.exact_train_samples or -1) for _, cfg in in_flight.values()}
                )
                active_str = ",".join(str(n) for n in active_n) if active_n else "-"
                pbar.set_postfix_str(f"n_train={active_str} done={_done_by_n_train()}")

            _update_pbar_postfix()
            while queue or in_flight:
                while queue and len(in_flight) < chunk_max_in_flight:
                    key, cfg = queue.popleft()
                    fut = executor.submit(train_and_evaluate, cfg)
                    in_flight[fut] = (key, cfg)
                    run_start_times[fut] = datetime.now()
                    _update_pbar_postfix()
                for future in as_completed(in_flight):
                    res = future.result()
                    key, cfg = in_flight[future]
                    n_train = int(cfg.exact_train_samples or -1)
                    n_test = int(cfg.exact_test_samples or -1)
                    res["n_train_samples"] = n_train
                    res["n_test_samples"] = n_test
                    results.append(res)
                    completed_per_n_train[n_train] += 1
                    _update_running_stats(stats_by_group[key], float(res["test_accuracy"]))
                    pbar.update(1)
                    strength_value = res["p"] if corruption_mode == "replacement" else res["sigma"]
                    if strength_value not in seen_values:
                        seen_values.add(strength_value)
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(f"[{timestamp}] new {sweep_label} encountered: {strength_value:.3f}")
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    width_label = res["mlp_hidden_sizes"][0] if res["mlp_hidden_sizes"] else -1
                    elapsed = datetime.now() - run_start_times.get(future, datetime.now())
                    print(
                        f"[{timestamp}] done: model={res['model_type']} act={res['activation']} "
                        f"{sweep_label}={strength_value:.3f} n_train={n_train} width={width_label} "
                        f"acc={res['test_accuracy']:.4f} "
                        f"elapsed={elapsed}"
                    )
                    stderr = _stderr_from_stats(stats_by_group[key])
                    n_done = stats_by_group[key]["n"]
                    if n_done < group_min_repeats[key]:
                        _enqueue(key)
                    elif n_done < group_max_repeats[key] and stderr > stderr_target:
                        _enqueue(key)
                    del in_flight[future]
                    _update_pbar_postfix()
                    break
        pbar.close()

    _run_chunk(
        group_keys,
        chunk_workers=max_workers,
        chunk_max_in_flight=max_in_flight,
        chunk_label="all n_train",
    )

    cache_key = build_cache_key(
        max_repeats=max_repeats,
        epochs=epochs,
        exact_sample_counts=exact_sample_counts,
        n_train_values=n_train_values,
        dataset_name=dataset_name,
        dataset_dim=dataset_dim,
        infimnist_cache_dir=infimnist_cache_dir,
        loss_type=loss_type,
        ps=ps,
        width=width,
        mlp_depth=mlp_depth,
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
        "corruption_mode": corruption_mode,
        "ps": ps,
        "width": width,
        "n_train_values": n_train_values,
        "mlp_depth": mlp_depth,
        "max_repeats": max_repeats,
        "min_repeats": min_repeats,
        "stderr_target": stderr_target,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "loss_type": loss_type,
        "dataset_name": dataset_name,
        "dataset_dim": dataset_dim,
        "infimnist_cache_dir": infimnist_cache_dir,
        "pyfssa_min_n_train": pyfssa_min_n_train,
        "max_workers": max_workers,
        "max_in_flight": max_in_flight,
        "serial_only": serial_only,
        "pool_start_method": pool_start_method,
        "data_workers": data_workers,
        "cpu_threads_per_worker": cpu_threads_per_worker,
        "cpu_cores": cpu_cores,
        "brightness_scale": brightness_scale,
        "split_seed": split_seed,
        "exact_sample_counts": exact_sample_counts,
        "output_dir": output_dir,
        "seed": seed,
        "max_train_samples": max_train_samples,
        "use_cuda": use_cuda,
        "total_runs": len(results),
    }
    write_summary_pdf(pdf_path, summary_rows, metadata)

    pyfssa_rows: list[dict] = []
    for model_type, act_map in metadata.get("pyfssa_zeta0_results", {}).items():
        if not isinstance(act_map, dict):
            continue
        for activation, result in act_map.items():
            if not isinstance(result, dict):
                continue
            if "error" in result:
                pyfssa_rows.append(
                    {
                        "model_type": model_type,
                        "activation": activation,
                        "status": "error",
                        "error": result["error"],
                        "rho_c": "",
                        "nu": "",
                        "zeta": "",
                        "S": "",
                        "min_n_train": "",
                        "n_train_values": "",
                    }
                )
                continue
            pyfssa_rows.append(
                {
                    "model_type": model_type,
                    "activation": activation,
                    "status": "ok",
                    "error": "",
                    "rho_c": result.get("rho_c"),
                    "nu": result.get("nu"),
                    "zeta": result.get("zeta"),
                    "S": result.get("S"),
                    "min_n_train": result.get("min_n_train"),
                    "n_train_values": result.get("n_train_values"),
                }
            )
    pyfssa_path = os.path.join(output_dir, f"results_pyfssa_{cache_key}{suffix_tag}.csv")
    write_csv(pyfssa_path, pyfssa_rows)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Saved:")
    print(f"  {per_run_path}")
    print(f"  {summary_path}")
    print(f"  {pyfssa_path}")
    print(f"  {pdf_path}")
    elapsed = datetime.now() - start_time
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Elapsed: {elapsed}")
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    runtime_log_path = os.path.join(output_dir, f"runtime_log_{script_name}.csv")
    append_runtime_log(
        runtime_log_path,
        {
            "timestamp_start": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp_end": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": int(elapsed.total_seconds()),
            "script": "run_experiments_noisy_training_data_ntrain_sweep.py",
            "cache_key": cache_key,
            "output_dir": output_dir,
            "total_runs": len(results),
        },
    )


if __name__ == "__main__":
    main()
