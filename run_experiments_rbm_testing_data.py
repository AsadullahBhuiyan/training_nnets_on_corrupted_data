import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import statistics
import csv
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss

from nnet_models import apply_corruption_numpy, load_mnist_numpy


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


def append_runtime_log(path: str, row: dict) -> None:
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def build_cache_key(
    corruption_trials: int,
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
    return f"rbmtest_r{corruption_trials}_tf{test_fraction:.3f}_{strength_tag}_c{rbm_components}_iter{rbm_n_iter}"


def main() -> None:
    start_time = datetime.now()
    # Manual configuration (edit these values directly).
    corruption_mode = "replacement"  # options: "replacement", "additive"
    ps = np.linspace(0.8, 1.0, 21)
    sigmas = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    corruption_trials = 100
    rbm_components = 256
    rbm_learning_rate = 0.01
    rbm_batch_size = 128
    rbm_n_iter = 20
    rbm_classifier_max_iter = 1000
    brightness_scale = 1.0
    custom_split = False
    test_fraction = 0.2
    split_seed = 1234
    split_source = "train"
    output_dir = "results_noisy_test"
    suffix = "rbm"
    seed = 1234
    max_train_samples = None

    apply_cpu_affinity(None)
    os.makedirs(output_dir, exist_ok=True)

    sweep_values = ps if corruption_mode == "replacement" else sigmas
    sweep_label = "p" if corruption_mode == "replacement" else "sigma"
    suffix_tag = f"_{suffix}" if suffix else ""
    suffix_note = f" (suffix={suffix})" if suffix else ""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Training RBM once for noisy-test evaluation...{suffix_note}")

    x_train, y_train, x_test, y_test = load_mnist_numpy(
        brightness_scale=brightness_scale,
        max_train_samples=max_train_samples,
        custom_split=custom_split,
        test_fraction=test_fraction,
        split_seed=split_seed,
        split_source=split_source,
    )
    x_train = apply_corruption_numpy(x_train, corruption_mode, 0.0, 0.0)

    rbm = BernoulliRBM(
        n_components=rbm_components,
        learning_rate=rbm_learning_rate,
        batch_size=rbm_batch_size,
        n_iter=rbm_n_iter,
        random_state=seed,
        verbose=False,
    )
    clf = LogisticRegression(
        max_iter=rbm_classifier_max_iter,
        n_jobs=1,
        multi_class="auto",
    )
    model = Pipeline([("rbm", rbm), ("logreg", clf)])
    model.fit(x_train, y_train)
    probs = model.predict_proba(x_test)
    clean_acc = float((probs.argmax(axis=1) == y_test).mean())
    clean_loss = float(log_loss(y_test, probs))
    print(f"[rbm] clean test accuracy: {clean_acc:.4f} (loss={clean_loss:.4f})")

    summary_rows: list[dict] = []
    for value in tqdm(sweep_values, desc=f"{sweep_label} sweep{suffix_note}", unit="s"):
        p = value if corruption_mode == "replacement" else 0.0
        sigma = value if corruption_mode == "additive" else 0.0
        accs = []
        for _ in range(corruption_trials):
            x_corrupt = apply_corruption_numpy(x_test, corruption_mode, p, sigma)
            preds = model.predict(x_corrupt)
            accs.append(float((preds == y_test).mean()))
        mean_acc = statistics.mean(accs)
        std_acc = statistics.pstdev(accs) if len(accs) > 1 else 0.0
        stderr_acc = std_acc / (len(accs) ** 0.5) if len(accs) > 1 else 0.0
        summary_rows.append(
            {
                "model_type": "rbm",
                "corruption_mode": corruption_mode,
                "p": p,
                "sigma": sigma,
                "corruption_trials": corruption_trials,
                "mean_test_accuracy": mean_acc,
                "std_test_accuracy": std_acc,
                "stderr_test_accuracy": stderr_acc,
            }
        )
        print(
            f"[rbm] {sweep_label}={value:.3f} mean_acc={mean_acc:.4f} stderr={stderr_acc:.4f}"
        )

    cache_key = build_cache_key(
        corruption_trials=corruption_trials,
        test_fraction=test_fraction,
        corruption_mode=corruption_mode,
        ps=ps,
        sigmas=sigmas,
        rbm_components=rbm_components,
        rbm_n_iter=rbm_n_iter,
    )
    pdf_path = os.path.join(output_dir, f"noisy_test_summary_{cache_key}{suffix_tag}.pdf")
    with PdfPages(pdf_path) as pdf:
        sub = sorted(summary_rows, key=lambda r: r[sweep_label])
        ps_sorted = [row[sweep_label] for row in sub]
        means = [row["mean_test_accuracy"] for row in sub]
        stderrs = [row["stderr_test_accuracy"] for row in sub]
        stds = [row["std_test_accuracy"] for row in sub]

        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.errorbar(ps_sorted, means, yerr=stderrs, marker="o")
        ax.set_title("rbm (noisy test)")
        ax.set_xlabel(sweep_label)
        ax.set_ylabel("mean test accuracy")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.plot(ps_sorted, stds, marker="o")
        ax.set_title("rbm std (noisy test)")
        ax.set_xlabel(sweep_label)
        ax.set_ylabel("std test accuracy")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        metadata = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_type": "rbm",
            "corruption_mode": corruption_mode,
            "ps": ps,
            "sigmas": sigmas,
            "corruption_trials": corruption_trials,
            "rbm_components": rbm_components,
            "rbm_learning_rate": rbm_learning_rate,
            "rbm_batch_size": rbm_batch_size,
            "rbm_n_iter": rbm_n_iter,
            "rbm_classifier_max_iter": rbm_classifier_max_iter,
            "brightness_scale": brightness_scale,
            "custom_split": custom_split,
            "test_fraction": test_fraction,
            "split_seed": split_seed,
            "split_source": split_source,
            "output_dir": output_dir,
            "seed": seed,
            "max_train_samples": max_train_samples,
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

    elapsed = datetime.now() - start_time
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    runtime_log_path = os.path.join(output_dir, f"runtime_log_{script_name}.csv")
    append_runtime_log(
        runtime_log_path,
        {
            "timestamp_start": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp_end": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": int(elapsed.total_seconds()),
            "script": "run_experiments_rbm_testing_data.py",
            "cache_key": cache_key,
            "output_dir": output_dir,
            "total_runs": len(summary_rows),
        },
    )


if __name__ == "__main__":
    main()
