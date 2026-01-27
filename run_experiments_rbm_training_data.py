import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

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

from nnet_models import TrainConfig, train_and_evaluate


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_runs(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, float, float, int], list[dict]] = {}
    for row in rows:
        key = (
            row["model_type"],
            row["corruption_mode"],
            float(row["p"]),
            float(row["sigma"]),
            int(row.get("rbm_components", 0)),
        )
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict] = []
    for (model_type, corruption_mode, p, sigma, components), items in grouped.items():
        accs = [float(r["test_accuracy"]) for r in items]
        losses = [float(r["test_loss"]) for r in items]
        summary_rows.append(
            {
                "model_type": model_type,
                "corruption_mode": corruption_mode,
                "p": p,
                "sigma": sigma,
                "rbm_components": components,
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
    summary_rows.sort(key=lambda r: (r["corruption_mode"], r["p"], r["sigma"]))
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
        import torch

        torch.set_num_interop_threads(1)
    except Exception:
        pass


def build_cache_key(
    repeats: int,
    test_fraction: float,
    corruption_mode: str,
    ps: list[float],
    sigmas: list[float],
    rbm_components: int,
    rbm_n_iter: int,
) -> str:
    if corruption_mode == "replacement":
        p_min, p_max = min(ps), max(ps)
        strength_tag = f"p{p_min:.2f}-{p_max:.2f}"
    else:
        s_min, s_max = min(sigmas), max(sigmas)
        strength_tag = f"s{s_min:.2f}-{s_max:.2f}"
    return (
        f"rbm_r{repeats}_tf{test_fraction:.3f}_{strength_tag}_"
        f"c{rbm_components}_iter{rbm_n_iter}"
    )


def write_summary_pdf(path: str, summary_rows: list[dict], metadata: dict) -> None:
    df = pd.DataFrame(summary_rows)
    corruption_modes = df["corruption_mode"].unique()
    sweep_label = "sigma" if len(corruption_modes) == 1 and corruption_modes[0] == "additive" else "p"

    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        df = df.sort_values(sweep_label)
        ax.errorbar(
            df[sweep_label],
            df["mean_test_accuracy"],
            yerr=df["stderr_test_accuracy"],
            marker="o",
        )
        ax.set_title("rbm")
        ax.set_xlabel(sweep_label)
        ax.set_ylabel("mean test accuracy")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.errorbar(
            df[sweep_label],
            df["mean_test_loss"],
            yerr=df["stderr_test_loss"],
            marker="o",
        )
        ax.set_title("rbm")
        ax.set_xlabel(sweep_label)
        ax.set_ylabel("mean test loss")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.plot(df[sweep_label], df["std_test_accuracy"], marker="o")
        ax.set_title("rbm")
        ax.set_xlabel(sweep_label)
        ax.set_ylabel("std test accuracy")
        ax.grid(True, alpha=0.3)
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
    corruption_mode = "replacement"  # options: "replacement", "additive"
    ps = np.linspace(0.8, 1.0, 21)
    sigmas = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    repeats = 100
    rbm_components = 256
    rbm_learning_rate = 0.01
    rbm_batch_size = 128
    rbm_n_iter = 20
    rbm_classifier_max_iter = 1000
    max_workers = 10
    data_workers = 0
    cpu_threads_per_worker = 1
    cpu_cores = None  # Example: [0, 1, 2, 3] to pin processes.
    brightness_scale = 1.0
    custom_split = False
    test_fraction = 0.2
    split_seed = 1234
    split_source = "train"
    output_dir = "results"
    suffix = "rbm"
    seed = 1234
    max_train_samples = None
    use_cuda = False

    if use_cuda and max_workers > 1:
        raise ValueError("use_cuda=True only supports max_workers=1 for now.")

    apply_cpu_affinity(cpu_cores)
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    sweep_values = ps if corruption_mode == "replacement" else sigmas
    sweep_label = "p" if corruption_mode == "replacement" else "sigma"
    configs: list[TrainConfig] = []
    for value in sweep_values:
        p = value if corruption_mode == "replacement" else 0.0
        sigma = value if corruption_mode == "additive" else 0.0
        for _ in range(repeats):
            run_seed = random.randint(1, 1_000_000_000)
            configs.append(
                TrainConfig(
                    activation="rbm",
                    model_type="rbm",
                    corruption_mode=corruption_mode,
                    p=p,
                    sigma=sigma,
                    epochs=1,
                    batch_size=rbm_batch_size,
                    learning_rate=rbm_learning_rate,
                    weight_decay=0.0,
                    loss_type="cross_entropy",
                    rbm_components=rbm_components,
                    rbm_learning_rate=rbm_learning_rate,
                    rbm_batch_size=rbm_batch_size,
                    rbm_n_iter=rbm_n_iter,
                    rbm_classifier_max_iter=rbm_classifier_max_iter,
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
        f"[{timestamp}] Launching {len(configs)} RBM runs with max_workers={max_workers}..."
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
        desc = f"RBM runs{suffix_note}"
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
                f"[{timestamp}] done: model=rbm {sweep_label}={strength_value:.3f} "
                f"acc={res['test_accuracy']:.4f}"
            )

    cache_key = build_cache_key(
        repeats=repeats,
        test_fraction=test_fraction,
        corruption_mode=corruption_mode,
        ps=ps,
        sigmas=sigmas,
        rbm_components=rbm_components,
        rbm_n_iter=rbm_n_iter,
    )
    per_run_path = os.path.join(output_dir, f"results_per_run_{cache_key}{suffix_tag}.csv")
    summary_path = os.path.join(output_dir, f"results_summary_{cache_key}{suffix_tag}.csv")
    write_csv(per_run_path, results)

    summary_rows = summarize_runs(results)
    write_csv(summary_path, summary_rows)

    pdf_path = os.path.join(output_dir, f"accuracy_vs_{cache_key}{suffix_tag}.pdf")
    metadata = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": "rbm",
        "corruption_mode": corruption_mode,
        "ps": ps,
        "sigmas": sigmas,
        "repeats": repeats,
        "rbm_components": rbm_components,
        "rbm_learning_rate": rbm_learning_rate,
        "rbm_batch_size": rbm_batch_size,
        "rbm_n_iter": rbm_n_iter,
        "rbm_classifier_max_iter": rbm_classifier_max_iter,
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


if __name__ == "__main__":
    main()
