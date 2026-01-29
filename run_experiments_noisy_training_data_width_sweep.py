import os
cpu_max = 30

import csv
import random
import statistics
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
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
        width = int(row["mlp_hidden_sizes"][0]) if row["mlp_hidden_sizes"] else -1
        key = (
            row["activation"],
            row["model_type"],
            row["corruption_mode"],
            float(row["p"]),
            float(row["sigma"]),
            width,
        )
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict] = []
    for (activation, model_type, corruption_mode, p, sigma, width), items in grouped.items():
        accs = [float(r["test_accuracy"]) for r in items]
        losses = [float(r["test_loss"]) for r in items]
        train_losses = [float(r.get("train_loss", 0.0)) for r in items]
        ntk_ranks = [float(r.get("ntk_rank", 0.0)) for r in items if r.get("ntk_rank") is not None]
        ntk_entropies = [float(r.get("ntk_entropy", 0.0)) for r in items if r.get("ntk_entropy") is not None]
        summary_rows.append(
            {
                "activation": activation,
                "model_type": model_type,
                "corruption_mode": corruption_mode,
                "p": p,
                "sigma": sigma,
                "mlp_width": width,
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
                "mean_ntk_rank": statistics.mean(ntk_ranks) if ntk_ranks else math.nan,
                "std_ntk_rank": statistics.pstdev(ntk_ranks) if len(ntk_ranks) > 1 else 0.0,
                "stderr_ntk_rank": (statistics.pstdev(ntk_ranks) / (len(ntk_ranks) ** 0.5))
                if len(ntk_ranks) > 1
                else 0.0,
                "mean_ntk_entropy": statistics.mean(ntk_entropies) if ntk_entropies else math.nan,
                "std_ntk_entropy": statistics.pstdev(ntk_entropies) if len(ntk_entropies) > 1 else 0.0,
                "stderr_ntk_entropy": (statistics.pstdev(ntk_entropies) / (len(ntk_entropies) ** 0.5))
                if len(ntk_entropies) > 1
                else 0.0,
            }
        )
    summary_rows.sort(
        key=lambda r: (
            r["model_type"],
            r["activation"],
            r["corruption_mode"],
            r["mlp_width"],
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


def compute_fss_objective(
    data_by_width: dict[int, tuple[np.ndarray, np.ndarray]],
    pc: float,
    nu: float,
) -> float:
    curves = {}
    xs_all = []
    for width, (p_vals, s_vals) in data_by_width.items():
        if pc < p_vals.min() or pc > p_vals.max():
            continue
        s_pc = np.interp(pc, p_vals, s_vals)
        x_vals = (p_vals - pc) * (width ** (1.0 / nu))
        y_vals = s_vals - s_pc
        curves[width] = (x_vals, y_vals)
        xs_all.append(x_vals)
    if not curves:
        return float("inf")
    xs_all = np.unique(np.concatenate(xs_all))
    total = 0.0
    for x in xs_all:
        ys = []
        for x_vals, y_vals in curves.values():
            if x < x_vals.min() or x > x_vals.max():
                continue
            ys.append(np.interp(x, x_vals, y_vals))
        if len(ys) < 2:
            continue
        ybar = float(np.mean(ys))
        total += float(np.sum((np.array(ys) - ybar) ** 2))
    return total


def find_best_fss(act_df: pd.DataFrame, sweep_label: str) -> tuple[float, float, plt.Figure | None]:
    widths = sorted(act_df["mlp_width"].unique())
    data_by_width = {}
    for width in widths:
        sub = act_df[act_df["mlp_width"] == width].sort_values(sweep_label)
        p_vals = sub[sweep_label].to_numpy(dtype=float)
        s_vals = sub["mean_test_accuracy"].to_numpy(dtype=float)
        if len(p_vals) < 3:
            continue
        data_by_width[width] = (p_vals, s_vals)
    if len(data_by_width) < 2:
        return math.nan, math.nan, None

    p_min = min(v[0].min() for v in data_by_width.values())
    p_max = max(v[0].max() for v in data_by_width.values())
    pc_grid = np.linspace(p_min, p_max, 41)
    nu_grid = np.linspace(0.2, 5.0, 41)

    best = (float("inf"), math.nan, math.nan)
    for pc in pc_grid:
        for nu in nu_grid:
            r = compute_fss_objective(data_by_width, pc, nu)
            if r < best[0]:
                best = (r, pc, nu)

    _, pc0, nu0 = best
    if math.isnan(pc0) or math.isnan(nu0):
        return math.nan, math.nan, None

    pc_grid = np.linspace(max(p_min, pc0 - 0.05), min(p_max, pc0 + 0.05), 41)
    nu_grid = np.linspace(max(0.2, nu0 - 1.0), min(5.0, nu0 + 1.0), 41)
    best = (float("inf"), pc0, nu0)
    for pc in pc_grid:
        for nu in nu_grid:
            r = compute_fss_objective(data_by_width, pc, nu)
            if r < best[0]:
                best = (r, pc, nu)
    _, pc_opt, nu_opt = best

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    for width, (p_vals, s_vals) in data_by_width.items():
        s_pc = np.interp(pc_opt, p_vals, s_vals)
        x_vals = (p_vals - pc_opt) * (width ** (1.0 / nu_opt))
        y_vals = s_vals - s_pc
        ax.plot(x_vals, y_vals, marker="o", label=f"w={width}")
    ax.set_title(f"FSS collapse ({sweep_label}*), pc={pc_opt:.3f}, nu={nu_opt:.2f}")
    ax.set_xlabel(f"({sweep_label} - pc) * w^(1/nu)")
    ax.set_ylabel("S(p,w) - S(pc,w)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    return pc_opt, nu_opt, fig


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
    raw_rows: list[dict],
    metadata: dict,
    ntk_hist_ps: list[float],
) -> None:
    df = pd.DataFrame(summary_rows)
    raw_df = pd.DataFrame(raw_rows)
    model_types = sorted(df["model_type"].unique())
    sweep_label = "sigma" if (not df.empty and "additive" in df["corruption_mode"].unique()) else "p"
    if not df.empty:
        grid_vals = sorted(df[sweep_label].unique())
        tol = metadata.get("ntk_hist_tol", 1e-6)
        snapped = []
        for target in ntk_hist_ps:
            nearest = min(grid_vals, key=lambda v: abs(v - target))
            if abs(nearest - target) <= tol:
                snapped.append(float(nearest))
        ntk_hist_ps = sorted(set(snapped))

    with PdfPages(path) as pdf:
        for model_type in model_types:
            sub = df[df["model_type"] == model_type]
            activations = sorted(sub["activation"].unique())
            widths = sorted(sub["mlp_width"].unique())
            corruption_modes = sub["corruption_mode"].unique()
            sweep_label = "sigma" if len(corruption_modes) == 1 and corruption_modes[0] == "additive" else "p"
            n = len(activations)

            fig, axes = plt.subplots(1, n, figsize=(4 * n, 3), sharey=True)
            if n == 1:
                axes = [axes]
            for ax, act in zip(axes, activations):
                for width in widths:
                    d = sub[(sub["activation"] == act) & (sub["mlp_width"] == width)].sort_values(
                        sweep_label
                    )
                    ax.errorbar(
                        d[sweep_label],
                        d["mean_test_accuracy"],
                        yerr=d["stderr_test_accuracy"],
                        marker="o",
                        label=f"w={width}",
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
                for width in widths:
                    d = sub[(sub["activation"] == act) & (sub["mlp_width"] == width)].sort_values(
                        sweep_label
                    )
                    ax.plot(d[sweep_label], d["std_test_accuracy"], marker="o", label=f"w={width}")
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
                for width in widths:
                    d = sub[(sub["activation"] == act) & (sub["mlp_width"] == width)].sort_values(
                        sweep_label
                    )
                    ax.errorbar(
                        d[sweep_label],
                        d["mean_test_loss"],
                        yerr=d["stderr_test_loss"],
                        marker="o",
                        label=f"w={width}",
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
                for width in widths:
                    d = sub[(sub["activation"] == act) & (sub["mlp_width"] == width)].sort_values(
                        sweep_label
                    )
                    ax.errorbar(
                        d[sweep_label],
                        d["mean_train_loss"],
                        yerr=d["stderr_train_loss"],
                        marker="o",
                        label=f"w={width}",
                    )
                ax.set_title(f"{model_type} / {act}")
                ax.set_xlabel(sweep_label)
                ax.grid(True, alpha=0.3)
            axes[0].set_ylabel("mean train loss")
            axes[0].legend(fontsize=8)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            if sub["mean_ntk_rank"].notna().any():
                fig, axes = plt.subplots(1, n, figsize=(4 * n, 3), sharey=True)
                if n == 1:
                    axes = [axes]
                for ax, act in zip(axes, activations):
                    for width in widths:
                        d = sub[(sub["activation"] == act) & (sub["mlp_width"] == width)].sort_values(
                            sweep_label
                        )
                        ax.errorbar(
                            d[sweep_label],
                            d["mean_ntk_rank"],
                            yerr=d["stderr_ntk_rank"],
                            marker="o",
                            label=f"w={width}",
                        )
                    ax.set_title(f"{model_type} / {act}")
                    ax.set_xlabel(sweep_label)
                    ax.grid(True, alpha=0.3)
                axes[0].set_ylabel("mean NTK effective rank")
                axes[0].legend(fontsize=8)
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

            if sub["mean_ntk_entropy"].notna().any():
                fig, axes = plt.subplots(1, n, figsize=(4 * n, 3), sharey=True)
                if n == 1:
                    axes = [axes]
                for ax, act in zip(axes, activations):
                    for width in widths:
                        d = sub[(sub["activation"] == act) & (sub["mlp_width"] == width)].sort_values(
                            sweep_label
                        )
                        ax.errorbar(
                            d[sweep_label],
                            d["mean_ntk_entropy"],
                            yerr=d["stderr_ntk_entropy"],
                            marker="o",
                            label=f"w={width}",
                        )
                    ax.set_title(f"{model_type} / {act}")
                    ax.set_xlabel(sweep_label)
                    ax.grid(True, alpha=0.3)
                axes[0].set_ylabel("mean NTK spectral entropy")
                axes[0].legend(fontsize=8)
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

            if not raw_df.empty and "ntk_eigvals" in raw_df.columns:
                for act in activations:
                    fig, axes = plt.subplots(1, len(ntk_hist_ps), figsize=(4 * len(ntk_hist_ps), 3), sharey=True)
                    if len(ntk_hist_ps) == 1:
                        axes = [axes]
                    for ax, p_val in zip(axes, ntk_hist_ps):
                        for width in widths:
                            mask = (
                                (raw_df["activation"] == act)
                                & (raw_df["mlp_hidden_sizes"].apply(lambda x: x[0] if isinstance(x, list) else -1) == width)
                                & (raw_df["p"] == p_val)
                            )
                            eigvals_list = raw_df.loc[mask, "ntk_eigvals"].dropna().tolist()
                            if not eigvals_list:
                                continue
                            pooled = np.concatenate([np.array(ev, dtype=float) for ev in eigvals_list])
                            pooled = np.maximum(pooled, 0.0)
                            total = pooled.sum()
                            pooled = pooled / total if total > 0 else pooled
                            counts, bins = np.histogram(pooled, bins=30, density=True)
                            centers = 0.5 * (bins[:-1] + bins[1:])
                            ax.plot(centers, counts, label=f"w={width}")
                        ax.set_title(f"{act} p={p_val:.2f}")
                        ax.set_xlabel("eigenvalue")
                        ax.grid(True, alpha=0.3)
                    axes[0].set_ylabel("density")
                    axes[0].legend(fontsize=8)
                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)

            # Finite-size scaling collapse for each activation.
            fss_results = {}
            for act in activations:
                act_df = sub[sub["activation"] == act]
                pc, nu, fss_fig = find_best_fss(act_df, sweep_label)
                if fss_fig is not None:
                    fss_results[act] = {"pc": pc, "nu": nu}
                    pdf.savefig(fss_fig)
                    plt.close(fss_fig)
            metadata["fss_results"] = fss_results

        meta_lines = [f"{key}: {value}" for key, value in metadata.items()]
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.05, 0.95, "Run Metadata", fontsize=14, fontweight="bold", va="top")
        fig.text(0.05, 0.9, "\n".join(meta_lines), fontsize=10, va="top")
        plt.axis("off")
        pdf.savefig(fig)
        plt.close(fig)


def build_cache_key(
    repeats: int,
    epochs: int,
    test_fraction: float,
    loss_type: str,
    corruption_mode: str,
    ps: list[float],
    sigmas: list[float],
    widths: list[int],
    mlp_depth: int,
) -> str:
    if corruption_mode == "replacement":
        p_min, p_max = min(ps), max(ps)
        strength_tag = f"p{p_min:.2f}-{p_max:.2f}"
    else:
        s_min, s_max = min(sigmas), max(sigmas)
        strength_tag = f"s{s_min:.2f}-{s_max:.2f}"
    w_min, w_max = min(widths), max(widths)
    return (
        f"mlpws_r{repeats}_e{epochs}_tf{test_fraction:.3f}_"
        f"{strength_tag}_w{w_min}-{w_max}_d{mlp_depth}_loss-{loss_type}"
    )


def main() -> None:
    start_time = datetime.now()
    # Manual configuration (edit these values directly).
    activations = ["relu", "tanh", "sigmoid"]
    model_types = ["mlp"]  # width sweep is MLP-only
    corruption_mode = "replacement"  # options: "replacement", "additive"
    ps = np.linspace(0.75, 1.0, 31)
    sigmas = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    widths = [64, 128, 256]
    mlp_depth = 1
    repeats = 100
    epochs = 20
    batch_size = 64
    learning_rate = 1e-3
    weight_decay = 0.0
    loss_type = "cross_entropy"  # options: "cross_entropy", "quadratic"
    max_workers = cpu_max
    data_workers = 0
    cpu_threads_per_worker = 1
    cpu_cores = list(range(cpu_max))  # Example: [0, 1, 2, 3] to pin processes.
    brightness_scale = 1.0
    ntk_compute = True
    ntk_subset_size = 64
    ntk_output = "true"  # options: "true", "all"
    ntk_use_corrupted_inputs = True
    ntk_seed = 1234
    ntk_hist_ps = [0, 0.5, 0.7, 0.8, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0]
    ntk_hist_tol = 1e-3
    ntk_show_progress = True
    custom_split = False
    test_fraction = 1/2
    split_seed = 1234
    split_source = "train"
    output_dir = "results"
    suffix = None
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

    sweep_values = ps if corruption_mode == "replacement" else sigmas
    sweep_label = "p" if corruption_mode == "replacement" else "sigma"

    configs: list[TrainConfig] = []
    for activation in activations:
        for model_type in model_types:
            for width in widths:
                for value in sweep_values:
                    p = value if corruption_mode == "replacement" else 0.0
                    sigma = value if corruption_mode == "additive" else 0.0
                    for _ in range(repeats):
                        run_seed = random.randint(1, 1_000_000_000)
                        configs.append(
                            TrainConfig(
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
                                ntk_compute=ntk_compute,
                                ntk_subset_size=ntk_subset_size,
                                ntk_output=ntk_output,
                                ntk_use_corrupted_inputs=ntk_use_corrupted_inputs,
                                ntk_seed=ntk_seed,
                                ntk_show_progress=ntk_show_progress,
                                seed=run_seed,
                                num_workers=data_workers,
                                cpu_threads=cpu_threads_per_worker,
                                brightness_scale=brightness_scale,
                                custom_split=custom_split,
                                test_fraction=test_fraction,
                                split_seed=split_seed,
                                split_source=split_source,
                                max_train_samples=max_train_samples,
                                use_cuda=use_cuda,
                            )
                        )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    suffix_tag = f"_{suffix}" if suffix else ""
    suffix_note = f" (suffix={suffix})" if suffix else ""
    print(
        f"[{timestamp}] Launching {len(configs)} runs with max_workers={max_workers}..."
        f"{suffix_note}"
    )

    results: list[dict] = []
    seen_values: set[float] = set()
    run_start_times = {}
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(cpu_cores,),
    ) as executor:
        futures = []
        for cfg in configs:
            fut = executor.submit(train_and_evaluate, cfg)
            futures.append(fut)
            run_start_times[fut] = datetime.now()
        desc = f"Runs{suffix_note} (widths={widths})"
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc, unit="run"):
            res = future.result()
            results.append(res)
            strength_value = res["p"] if corruption_mode == "replacement" else res["sigma"]
            if strength_value not in seen_values:
                seen_values.add(strength_value)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{timestamp}] new {sweep_label} encountered: {strength_value:.3f}")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            width = res["mlp_hidden_sizes"][0] if res["mlp_hidden_sizes"] else -1
            elapsed = datetime.now() - run_start_times.get(future, datetime.now())
            print(
                f"[{timestamp}] done: model={res['model_type']} act={res['activation']} "
                f"{sweep_label}={strength_value:.3f} width={width} acc={res['test_accuracy']:.4f} "
                f"elapsed={elapsed}"
            )

    cache_key = build_cache_key(
        repeats=repeats,
        epochs=epochs,
        test_fraction=test_fraction,
        loss_type=loss_type,
        corruption_mode=corruption_mode,
        ps=ps,
        sigmas=sigmas,
        widths=widths,
        mlp_depth=mlp_depth,
    )
    per_run_path = os.path.join(output_dir, f"results_per_run_{cache_key}{suffix_tag}.csv")
    summary_path = os.path.join(output_dir, f"results_summary_{cache_key}{suffix_tag}.csv")
    write_csv(per_run_path, results)

    summary_rows = summarize_runs(results)
    write_csv(summary_path, summary_rows)

    pdf_path = os.path.join(output_dir, f"accuracy_vs_{cache_key}{suffix_tag}.pdf")
    ntk_npz_path = os.path.join(output_dir, f"ntk_eigvals_{cache_key}{suffix_tag}.npz")
    metadata = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "activations": activations,
        "model_types": model_types,
        "corruption_mode": corruption_mode,
        "ps": ps,
        "sigmas": sigmas,
        "widths": widths,
        "mlp_depth": mlp_depth,
        "repeats": repeats,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "loss_type": loss_type,
        "ntk_compute": ntk_compute,
        "ntk_subset_size": ntk_subset_size,
        "ntk_output": ntk_output,
        "ntk_use_corrupted_inputs": ntk_use_corrupted_inputs,
        "ntk_seed": ntk_seed,
        "ntk_hist_ps": ntk_hist_ps,
        "ntk_hist_tol": ntk_hist_tol,
        "ntk_show_progress": ntk_show_progress,
        "max_workers": max_workers,
        "data_workers": data_workers,
        "cpu_threads_per_worker": cpu_threads_per_worker,
        "cpu_cores": cpu_cores,
        "brightness_scale": brightness_scale,
        "custom_split": custom_split,
        "test_fraction": test_fraction,
        "split_seed": split_seed,
        "split_source": split_source,
        "output_dir": output_dir,
        "seed": seed,
        "max_train_samples": max_train_samples,
        "use_cuda": use_cuda,
        "total_runs": len(configs),
    }
    write_summary_pdf(pdf_path, summary_rows, results, metadata, ntk_hist_ps)

    if ntk_compute:
        eig_rows = [r for r in results if r.get("ntk_eigvals") is not None]
        if eig_rows:
            np.savez(
                ntk_npz_path,
                eigvals=np.array([r["ntk_eigvals"] for r in eig_rows], dtype=object),
                p=np.array([r["p"] for r in eig_rows], dtype=float),
                activation=np.array([r["activation"] for r in eig_rows], dtype=object),
                width=np.array(
                    [r["mlp_hidden_sizes"][0] if r.get("mlp_hidden_sizes") else -1 for r in eig_rows],
                    dtype=int,
                ),
            )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Saved:")
    print(f"  {per_run_path}")
    print(f"  {summary_path}")
    print(f"  {pdf_path}")
    if ntk_compute and os.path.exists(ntk_npz_path):
        print(f"  {ntk_npz_path}")
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
            "script": "run_experiments_noisy_training_data_width_sweep.py",
            "cache_key": cache_key,
            "output_dir": output_dir,
            "total_runs": len(configs),
        },
    )


if __name__ == "__main__":
    main()
