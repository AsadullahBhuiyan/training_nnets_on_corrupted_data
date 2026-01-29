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
from tqdm import tqdm

from nnet_models import TrainConfig, train_and_evaluate, get_model_hparams


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_runs(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str, float, float], list[dict]] = {}
    for row in rows:
        key = (
            row["activation"],
            row["model_type"],
            row["corruption_mode"],
            float(row["p"]),
            float(row["sigma"]),
        )
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict] = []
    for (activation, model_type, corruption_mode, p, sigma), items in grouped.items():
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
        key=lambda r: (r["model_type"], r["activation"], r["corruption_mode"], r["p"], r["sigma"])
    )
    return summary_rows


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


def _init_worker(cores: list[int] | None) -> None:
    apply_cpu_affinity(cores)
    try:
        import torch

        torch.set_num_interop_threads(1)
    except Exception:
        pass


def write_summary_pdf(path: str, summary_rows: list[dict], metadata: dict) -> None:
    df = pd.DataFrame(summary_rows)
    model_types = sorted(df["model_type"].unique())

    with PdfPages(path) as pdf:
        for model_type in model_types:
            sub = df[df["model_type"] == model_type]
            activations = sorted(sub["activation"].unique())
            n = len(activations)
            corruption_modes = sub["corruption_mode"].unique()
            sweep_label = "sigma" if len(corruption_modes) == 1 and corruption_modes[0] == "additive" else "p"
            fig, axes = plt.subplots(1, n, figsize=(4 * n, 3), sharey=True)
            if n == 1:
                axes = [axes]
            for ax, act in zip(axes, activations):
                d = sub[sub["activation"] == act].sort_values(sweep_label)
                ax.errorbar(
                    d[sweep_label],
                    d["mean_test_accuracy"],
                    yerr=d["stderr_test_accuracy"],
                    marker="o",
                )
                ax.set_title(f"{model_type} / {act}")
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
                d = sub[sub["activation"] == act].sort_values(sweep_label)
                ax.plot(d[sweep_label], d["std_test_accuracy"], marker="o")
                ax.set_title(f"{model_type} / {act}")
                ax.set_xlabel(sweep_label)
                ax.grid(True, alpha=0.3)
            axes[0].set_ylabel("std test accuracy")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            fig, axes = plt.subplots(1, n, figsize=(4 * n, 3), sharey=True)
            if n == 1:
                axes = [axes]
            for ax, act in zip(axes, activations):
                d = sub[sub["activation"] == act].sort_values(sweep_label)
                ax.errorbar(
                    d[sweep_label],
                    d["mean_test_loss"],
                    yerr=d["stderr_test_loss"],
                    marker="o",
                )
                ax.set_title(f"{model_type} / {act}")
                ax.set_xlabel(sweep_label)
                ax.grid(True, alpha=0.3)
            axes[0].set_ylabel("mean test loss")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            fig, axes = plt.subplots(1, n, figsize=(4 * n, 3), sharey=True)
            if n == 1:
                axes = [axes]
            for ax, act in zip(axes, activations):
                d = sub[sub["activation"] == act].sort_values(sweep_label)
                ax.errorbar(
                    d[sweep_label],
                    d["mean_train_loss"],
                    yerr=d["stderr_train_loss"],
                    marker="o",
                )
                ax.set_title(f"{model_type} / {act}")
                ax.set_xlabel(sweep_label)
                ax.grid(True, alpha=0.3)
            axes[0].set_ylabel("mean train loss")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

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


def build_cache_key(
    model_types: list[str],
    repeats: int,
    epochs: int,
    test_fraction: float,
    loss_type: str,
    corruption_mode: str,
    ps: list[float],
    sigmas: list[float],
    mlp_width: int,
    mlp_depth: int,
    width_sweep: bool = False,
) -> str:
    model_tag = "-".join(model_types)
    if corruption_mode == "replacement":
        p_min, p_max = min(ps), max(ps)
        strength_tag = f"p{p_min:.2f}-{p_max:.2f}"
    else:
        s_min, s_max = min(sigmas), max(sigmas)
        strength_tag = f"s{s_min:.2f}-{s_max:.2f}"
    width_tag = f"w{mlp_width}" if not width_sweep else f"w{mlp_width}"
    return (
        f"{model_tag}_r{repeats}_e{epochs}_tf{test_fraction:.3f}_"
        f"{strength_tag}_{width_tag}_d{mlp_depth}_loss-{loss_type}"
    )


def main() -> None:
    start_time = datetime.now()
    # Manual configuration (edit these values directly).
    activations = ["relu", "tanh", "sigmoid", "gelu"]
    model_types = ["mlp"]  # options: "mlp", "cnn"
    corruption_mode = "replacement"  # options: "replacement", "additive"
    #ps = np.linspace(0.8, 1.0, 21)
    ps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    sigmas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    repeats = 100
    epochs = 20
    batch_size = 128
    learning_rate = 1e-3
    weight_decay = 0.0
    loss_type = "cross_entropy"  # options: "cross_entropy", "quadratic"
    mlp_width = 128
    mlp_depth = 1
    max_workers = max_cpu
    data_workers = 1
    cpu_threads_per_worker = 1
    cpu_cores = list(range(max_cpu))  # Example: [0, 1, 2, 3] to pin processes.
    brightness_scale = 1.0
    custom_split = False
    test_fraction = 0.2
    split_seed = 1234
    split_source = "train"
    output_dir = "results"
    suffix = None
    seed = 1234
    max_train_samples = None
    use_cuda = False

    if use_cuda and max_workers > 1:
        raise ValueError("use_cuda=True only supports max_workers=1 for now.")

    apply_cpu_affinity(cpu_cores)
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    configs: list[TrainConfig] = []
    run_id = 0
    sweep_values = ps if corruption_mode == "replacement" else sigmas
    sweep_label = "p" if corruption_mode == "replacement" else "sigma"

    for activation in activations:
        for model_type in model_types:
            for value in sweep_values:
                p = value if corruption_mode == "replacement" else 0.0
                sigma = value if corruption_mode == "additive" else 0.0
                mlp_hidden_sizes = [mlp_width] * mlp_depth if model_type == "mlp" else None
                for _ in range(repeats):
                    run_seed = random.randint(1, 1_000_000_000)
                    configs.append(
                        TrainConfig(
                            activation=activation,
                            model_type=model_type,
                            corruption_mode=corruption_mode,
                            p=p,
                            sigma=sigma,
                            mlp_hidden_sizes=mlp_hidden_sizes,
                            epochs=epochs,
                            batch_size=batch_size,
                            learning_rate=learning_rate,
                            weight_decay=weight_decay,
                            loss_type=loss_type,
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
                    run_id += 1

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    suffix_tag = f"_{suffix}" if suffix else ""
    suffix_note = f" (suffix={suffix})" if suffix else ""
    print(
        f"[{timestamp}] Launching {len(configs)} runs with max_workers={max_workers}..."
        f"{suffix_note}"
    )
    results: list[dict] = []
    seen_values: set[float] = set()

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(cpu_cores,),
    ) as executor:
        futures = [executor.submit(train_and_evaluate, cfg) for cfg in configs]
        desc = f"Runs{suffix_note}"
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc, unit="run"):
            res = future.result()
            results.append(res)
            strength_value = res["p"] if corruption_mode == "replacement" else res["sigma"]
            if strength_value not in seen_values:
                seen_values.add(strength_value)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{timestamp}] new {sweep_label} encountered: {strength_value:.3f}")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"[{timestamp}] done: model={res['model_type']} act={res['activation']} "
                f"{sweep_label}={strength_value:.3f} acc={res['test_accuracy']:.4f}"
            )

    cache_key = build_cache_key(
        model_types=model_types,
        repeats=repeats,
        epochs=epochs,
        test_fraction=test_fraction,
        loss_type=loss_type,
        corruption_mode=corruption_mode,
        ps=ps,
        sigmas=sigmas,
        mlp_width=mlp_width,
        mlp_depth=mlp_depth,
    )
    per_run_path = os.path.join(output_dir, f"results_per_run_{cache_key}{suffix_tag}.csv")
    summary_path = os.path.join(output_dir, f"results_summary_{cache_key}{suffix_tag}.csv")
    write_csv(per_run_path, results)

    summary_rows = summarize_runs(results)
    write_csv(summary_path, summary_rows)

    pdf_path = os.path.join(output_dir, f"accuracy_vs_{cache_key}{suffix_tag}.pdf")
    model_hparams = {mt: get_model_hparams(mt) for mt in model_types}
    metadata = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "activations": activations,
        "model_types": model_types,
        "corruption_mode": corruption_mode,
        "model_hparams": model_hparams,
        "ps": ps,
        "sigmas": sigmas,
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
    write_summary_pdf(pdf_path, summary_rows, metadata)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Saved:")
    print(f"  {per_run_path}")
    print(f"  {summary_path}")
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
            "script": "run_experiments_noisy_training_data.py",
            "cache_key": cache_key,
            "output_dir": output_dir,
            "total_runs": len(configs),
        },
    )


if __name__ == "__main__":
    main()
