import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import statistics
import csv
import random
import math
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from nnet_models import (
    TrainConfig,
    build_model,
    get_dataloaders,
    evaluate,
    apply_corruption,
    compute_empirical_ntk,
    compute_loss,
)


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


def ntk_effective_rank(eigvals: np.ndarray) -> float:
    eigvals = np.maximum(eigvals, 0.0)
    total = eigvals.sum()
    if total <= 0:
        return 0.0
    probs = eigvals / total
    entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
    return float(np.exp(entropy))


def ntk_spectral_entropy(eigvals: np.ndarray) -> float:
    eigvals = np.maximum(eigvals, 0.0)
    total = eigvals.sum()
    if total <= 0:
        return 0.0
    probs = eigvals / total
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def append_runtime_log(path: str, row: dict) -> None:
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def evaluate_with_corruption(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    corruption_mode: str,
    p: float,
    sigma: float,
    trials: int,
) -> tuple[float, float, float]:
    accuracies: list[float] = []
    model.eval()
    with torch.no_grad():
        for _ in range(trials):
            correct = 0
            total = 0
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)
                x = apply_corruption(x, corruption_mode, p, sigma)
                logits = model(x)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
            accuracies.append(correct / total)
    mean_acc = statistics.mean(accuracies)
    std_acc = statistics.pstdev(accuracies) if len(accuracies) > 1 else 0.0
    stderr_acc = std_acc / (len(accuracies) ** 0.5) if len(accuracies) > 1 else 0.0
    return mean_acc, std_acc, stderr_acc


def _worker_corruption_trials(
    model_state: dict,
    model_type: str,
    activation: str,
    batch_size: int,
    num_workers: int,
    cpu_threads: int,
    max_train_samples: int | None,
    corruption_mode: str,
    p: float,
    sigma: float,
    trials: int,
) -> list[float]:
    torch.set_num_threads(max(1, cpu_threads))
    device = torch.device("cpu")
    model = build_model(model_type, activation).to(device)
    model.load_state_dict(model_state)
    _, test_loader = get_dataloaders(
        batch_size=batch_size,
        p=0.0,
        num_workers=num_workers,
        max_train_samples=max_train_samples,
        corruption_mode=corruption_mode,
        sigma=0.0,
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
                x = apply_corruption(x, corruption_mode, p, sigma)
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
    batch_size: int,
    num_workers: int,
    cpu_threads: int,
    max_train_samples: int | None,
    corruption_mode: str,
    p: float,
    sigma: float,
    trials: int,
    max_workers: int,
    cpu_cores: list[int] | None,
) -> tuple[float, float, float]:
    if max_workers <= 1 or trials <= 1:
        _, test_loader = get_dataloaders(
            batch_size=batch_size,
            p=0.0,
            num_workers=num_workers,
            max_train_samples=max_train_samples,
            corruption_mode=corruption_mode,
            sigma=0.0,
        )
        return evaluate_with_corruption(
            model,
            test_loader,
            torch.device("cpu"),
            corruption_mode,
            p,
            sigma,
            trials,
        )

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
                batch_size,
                num_workers,
                cpu_threads,
                max_train_samples,
                corruption_mode,
                p,
                sigma,
                trials_i,
            )
            for trials_i in splits
        ]
        for future in as_completed(futures):
            all_accs.extend(future.result())

    mean_acc = statistics.mean(all_accs)
    std_acc = statistics.pstdev(all_accs) if len(all_accs) > 1 else 0.0
    stderr_acc = std_acc / (len(all_accs) ** 0.5) if len(all_accs) > 1 else 0.0
    return mean_acc, std_acc, stderr_acc


def main() -> None:
    start_time = datetime.now()
    # Manual configuration (edit these values directly).
    activations = ["relu", "tanh", "sigmoid", "gelu"]
    model_type = "cnn"  # options: "mlp", "cnn"
    corruption_mode = "replacement"  # options: "replacement", "additive"
    ps = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    sigmas = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    corruption_trials = 100
    epochs = 20
    batch_size = 128
    learning_rate = 1e-3
    weight_decay = 0.0
    loss_type = "cross_entropy"  # options: "cross_entropy", "quadratic"
    mlp_width = 256
    mlp_depth = 2
    data_workers = 2
    max_workers = 10
    cpu_threads_per_worker = 1
    cpu_cores = None  # Example: [0, 1, 2, 3] to pin processes.
    max_train_samples = None
    ntk_compute = True
    ntk_subset_size = 64
    ntk_output = "true"  # options: "true", "all"
    ntk_trials_per_p = 10
    ntk_hist_ps = [0.0, 0.5, 1.0]
    ntk_seed = 1234
    ntk_show_progress = True
    use_cuda = False
    custom_split = False
    test_fraction = 0.2
    split_seed = 1234
    split_source = "train"

    output_dir = "results_noisy_test"
    suffix = 'cnn'
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

    suffix_tag = f"_{suffix}" if suffix else ""
    suffix_note = f" (suffix={suffix})" if suffix else ""

    summary_rows = []
    apply_cpu_affinity(cpu_cores)
    torch.set_num_threads(max(1, cpu_threads_per_worker))
    torch.set_num_interop_threads(1)
    sweep_values = ps if corruption_mode == "replacement" else sigmas
    sweep_label = "p" if corruption_mode == "replacement" else "sigma"

    def build_cache_key() -> str:
        if corruption_mode == "replacement":
            p_min, p_max = min(ps), max(ps)
            strength_tag = f"p{p_min:.2f}-{p_max:.2f}"
        else:
            s_min, s_max = min(sigmas), max(sigmas)
            strength_tag = f"s{s_min:.2f}-{s_max:.2f}"
        width_tag = f"w{mlp_width}" if model_type == "mlp" else model_type
        return (
            f"{model_type}_r{corruption_trials}_e{epochs}_tf{test_fraction:.3f}_"
            f"{strength_tag}_{width_tag}_d{mlp_depth}_loss-{loss_type}"
        )

    for activation in activations:
        mlp_hidden_sizes = [mlp_width] * mlp_depth if model_type == "mlp" else None
        cfg = TrainConfig(
            activation=activation,
            model_type=model_type,
            corruption_mode=corruption_mode,
            p=0.0,
            sigma=0.0,
            mlp_hidden_sizes=mlp_hidden_sizes,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            loss_type=loss_type,
            seed=1234,
            num_workers=data_workers,
            cpu_threads=1,
            max_train_samples=max_train_samples,
            use_cuda=use_cuda,
            custom_split=custom_split,
            test_fraction=test_fraction,
            split_seed=split_seed,
            split_source=split_source,
        )

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"[{timestamp}] training model once for noisy-test evaluation..."
            f"{suffix_note} (activation={activation})"
        )

        train_loader, test_loader = get_dataloaders(
            batch_size=cfg.batch_size,
            p=cfg.p,
            num_workers=cfg.num_workers,
            max_train_samples=cfg.max_train_samples,
            corruption_mode=cfg.corruption_mode,
            sigma=cfg.sigma,
            brightness_scale=cfg.brightness_scale,
            custom_split=cfg.custom_split,
            test_fraction=cfg.test_fraction,
            split_seed=cfg.split_seed,
            split_source=cfg.split_source,
        )
        model = build_model(cfg.model_type, cfg.activation).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        for _ in tqdm(range(cfg.epochs), desc=f"epochs ({activation})", unit="epoch"):
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

        ntk_subset = None
        if ntk_compute:
            rng = random.Random(ntk_seed)
            indices = list(range(len(test_loader.dataset)))
            rng.shuffle(indices)
            subset = torch.utils.data.Subset(test_loader.dataset, indices[:ntk_subset_size])
            ntk_subset = torch.utils.data.DataLoader(
                subset,
                batch_size=min(32, ntk_subset_size),
                shuffle=False,
            )

        for value in tqdm(sweep_values, desc=f"{sweep_label} sweep{suffix_note} ({activation})", unit="s"):
            p = value if corruption_mode == "replacement" else 0.0
            sigma = value if corruption_mode == "additive" else 0.0
            mean_acc, std_acc, stderr_acc = evaluate_with_corruption_parallel(
                model=model,
                model_type=model_type,
                activation=activation,
                batch_size=batch_size,
                num_workers=data_workers,
                cpu_threads=cpu_threads_per_worker,
                max_train_samples=max_train_samples,
                corruption_mode=corruption_mode,
                p=p,
                sigma=sigma,
                trials=corruption_trials,
                max_workers=max_workers,
                cpu_cores=cpu_cores,
            )
            ntk_ranks = []
            ntk_entropies = []
            ntk_eigvals_runs = []
            if ntk_compute and ntk_subset is not None:
                for t in range(ntk_trials_per_p):
                    torch.manual_seed(ntk_seed + t + int(p * 1000))
                    _, eigvals = compute_empirical_ntk(
                        model=model,
                        loader=ntk_subset,
                        device=device,
                        max_samples=ntk_subset_size,
                        output_mode=ntk_output,
                        corruption_mode=corruption_mode,
                        p=p,
                        sigma=sigma,
                        use_corruption=True,
                        show_progress=ntk_show_progress,
                        progress_desc=f"ntk {activation} p={p:.2f}",
                    )
                    if eigvals.size > 0:
                        ntk_ranks.append(ntk_effective_rank(eigvals))
                        ntk_entropies.append(ntk_spectral_entropy(eigvals))
                        if value in ntk_hist_ps:
                            ntk_eigvals_runs.append(eigvals.tolist())
            mean_ntk_rank = statistics.mean(ntk_ranks) if ntk_ranks else math.nan
            std_ntk_rank = statistics.pstdev(ntk_ranks) if len(ntk_ranks) > 1 else 0.0
            stderr_ntk_rank = std_ntk_rank / (len(ntk_ranks) ** 0.5) if len(ntk_ranks) > 1 else 0.0
            mean_ntk_entropy = statistics.mean(ntk_entropies) if ntk_entropies else math.nan
            std_ntk_entropy = statistics.pstdev(ntk_entropies) if len(ntk_entropies) > 1 else 0.0
            stderr_ntk_entropy = std_ntk_entropy / (len(ntk_entropies) ** 0.5) if len(ntk_entropies) > 1 else 0.0
            summary_rows.append(
                {
                    "activation": activation,
                    "model_type": model_type,
                    "corruption_mode": corruption_mode,
                    "p": p,
                    "sigma": sigma,
                    "corruption_trials": corruption_trials,
                    "mean_test_accuracy": mean_acc,
                    "std_test_accuracy": std_acc,
                    "stderr_test_accuracy": stderr_acc,
                    "mean_ntk_rank": mean_ntk_rank,
                    "std_ntk_rank": std_ntk_rank,
                    "stderr_ntk_rank": stderr_ntk_rank,
                    "mean_ntk_entropy": mean_ntk_entropy,
                    "std_ntk_entropy": std_ntk_entropy,
                    "stderr_ntk_entropy": stderr_ntk_entropy,
                    "ntk_eigvals_runs": ntk_eigvals_runs if ntk_eigvals_runs else None,
                }
            )
            print(
                f"[{activation}] {sweep_label}={value:.3f} mean_acc={mean_acc:.4f} "
                f"stderr={stderr_acc:.4f}"
            )

    cache_key = build_cache_key()
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
            ax.set_title(f"{model_type} / {act} (noisy test)")
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
            ax.set_title(f"{model_type} / {act} std (noisy test)")
            ax.set_xlabel(sweep_label)
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel("std test accuracy")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        if any(not math.isnan(row.get("mean_ntk_rank", math.nan)) for row in summary_rows):
            fig, axes = plt.subplots(1, n, figsize=(4 * n, 3), sharey=True)
            if n == 1:
                axes = [axes]
            for ax, act in zip(axes, activations):
                sub = [row for row in summary_rows if row["activation"] == act]
                sub = sorted(sub, key=lambda r: r[sweep_label])
                ps_sorted = [row[sweep_label] for row in sub]
                means = [row["mean_ntk_rank"] for row in sub]
                stderrs = [row["stderr_ntk_rank"] for row in sub]
                ax.errorbar(ps_sorted, means, yerr=stderrs, marker="o")
                ax.set_title(f"{model_type} / {act} NTK rank")
                ax.set_xlabel(sweep_label)
                ax.grid(True, alpha=0.3)
            axes[0].set_ylabel("mean NTK effective rank")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        if any(not math.isnan(row.get("mean_ntk_entropy", math.nan)) for row in summary_rows):
            fig, axes = plt.subplots(1, n, figsize=(4 * n, 3), sharey=True)
            if n == 1:
                axes = [axes]
            for ax, act in zip(axes, activations):
                sub = [row for row in summary_rows if row["activation"] == act]
                sub = sorted(sub, key=lambda r: r[sweep_label])
                ps_sorted = [row[sweep_label] for row in sub]
                means = [row["mean_ntk_entropy"] for row in sub]
                stderrs = [row["stderr_ntk_entropy"] for row in sub]
                ax.errorbar(ps_sorted, means, yerr=stderrs, marker="o")
                ax.set_title(f"{model_type} / {act} NTK entropy")
                ax.set_xlabel(sweep_label)
                ax.grid(True, alpha=0.3)
            axes[0].set_ylabel("mean NTK spectral entropy")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        if ntk_compute:
            for act in activations:
                fig, axes = plt.subplots(1, len(ntk_hist_ps), figsize=(4 * len(ntk_hist_ps), 3), sharey=True)
                if len(ntk_hist_ps) == 1:
                    axes = [axes]
                for ax, p_val in zip(axes, ntk_hist_ps):
                    rows = [
                        row for row in summary_rows
                        if row["activation"] == act and abs(row["p"] - p_val) < 1e-9
                    ]
                    eigvals_runs = []
                    for row in rows:
                        ev = row.get("ntk_eigvals_runs")
                        if ev:
                            eigvals_runs.extend(ev)
                    if eigvals_runs:
                        pooled = np.concatenate([np.array(ev, dtype=float) for ev in eigvals_runs])
                        pooled = np.maximum(pooled, 0.0)
                        total = pooled.sum()
                        pooled = pooled / total if total > 0 else pooled
                        counts, bins = np.histogram(pooled, bins=30, density=True)
                        centers = 0.5 * (bins[:-1] + bins[1:])
                        ax.plot(centers, counts)
                    ax.set_title(f"{act} p={p_val:.2f}")
                    ax.set_xlabel("eigenvalue")
                    ax.grid(True, alpha=0.3)
                axes[0].set_ylabel("density")
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

        metadata = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "activations": activations,
            "model_type": model_type,
            "corruption_mode": corruption_mode,
            "ps": ps,
            "sigmas": sigmas,
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
            "ntk_compute": ntk_compute,
            "ntk_subset_size": ntk_subset_size,
            "ntk_output": ntk_output,
            "ntk_trials_per_p": ntk_trials_per_p,
            "ntk_hist_ps": ntk_hist_ps,
            "ntk_seed": ntk_seed,
            "ntk_show_progress": ntk_show_progress,
            "custom_split": custom_split,
            "test_fraction": test_fraction,
            "split_seed": split_seed,
            "split_source": split_source,
            "max_train_samples": max_train_samples,
            "use_cuda": use_cuda,
            "suffix": suffix,
        }
        meta_lines = [f"{key}: {value}" for key, value in metadata.items()]
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(
            0.05,
            0.95,
            "Run Metadata",
            fontsize=14,
            fontweight="bold",
            va="top",
        )
        fig.text(0.05, 0.9, "\n".join(meta_lines), fontsize=10, va="top")
        plt.axis("off")
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Saved PDF: {pdf_path}")
    if ntk_compute:
        eigvals = []
        ps_list = []
        acts_list = []
        for row in summary_rows:
            ev_runs = row.get("ntk_eigvals_runs")
            if ev_runs:
                for ev in ev_runs:
                    eigvals.append(ev)
                    ps_list.append(row["p"])
                    acts_list.append(row["activation"])
        if eigvals:
            ntk_npz_path = os.path.join(output_dir, f"ntk_eigvals_{build_cache_key()}{suffix_tag}.npz")
            np.savez(
                ntk_npz_path,
                eigvals=np.array(eigvals, dtype=object),
                p=np.array(ps_list, dtype=float),
                activation=np.array(acts_list, dtype=object),
            )
            print(f"Saved NTK eigvals: {ntk_npz_path}")

    elapsed = datetime.now() - start_time
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    runtime_log_path = os.path.join(output_dir, f"runtime_log_{script_name}.csv")
    append_runtime_log(
        runtime_log_path,
        {
            "timestamp_start": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp_end": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": int(elapsed.total_seconds()),
            "script": "run_experiments_noisy_testing_data.py",
            "cache_key": build_cache_key(),
            "output_dir": output_dir,
            "total_runs": len(summary_rows),
        },
    )


if __name__ == "__main__":
    main()
