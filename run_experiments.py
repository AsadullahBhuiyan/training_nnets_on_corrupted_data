import csv
import os
import random
import statistics
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    grouped: dict[tuple[str, str, float], list[dict]] = {}
    for row in rows:
        key = (row["activation"], row["model_type"], float(row["p"]))
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict] = []
    for (activation, model_type, p), items in grouped.items():
        accs = [float(r["test_accuracy"]) for r in items]
        summary_rows.append(
            {
                "activation": activation,
                "model_type": model_type,
                "p": p,
                "repeats": len(items),
                "mean_test_accuracy": statistics.mean(accs),
                "std_test_accuracy": statistics.pstdev(accs) if len(accs) > 1 else 0.0,
                "stderr_test_accuracy": (statistics.pstdev(accs) / (len(items) ** 0.5))
                if len(items) > 1
                else 0.0,
            }
        )
    summary_rows.sort(key=lambda r: (r["model_type"], r["activation"], r["p"]))
    return summary_rows


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

        table_df = df.sort_values(["model_type", "activation", "p"]).copy()
        table_df["p"] = table_df["p"].map(lambda v: f"{v:.3f}")
        table_df["mean_test_accuracy"] = table_df["mean_test_accuracy"].map(lambda v: f"{v:.4f}")
        table_df["std_test_accuracy"] = table_df["std_test_accuracy"].map(lambda v: f"{v:.4f}")
        table_df["stderr_test_accuracy"] = table_df["stderr_test_accuracy"].map(lambda v: f"{v:.4f}")

        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.set_title("Results Summary", fontweight="bold")
        table = ax.table(
            cellText=table_df.values,
            colLabels=table_df.columns,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.2)
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


def main() -> None:
    # Manual configuration (edit these values directly).
    activations = ["relu", "tanh", "sigmoid"]
    model_types = ["mlp"]  # options: "mlp", "cnn"
    ps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    repeats = 100
    epochs = 20
    batch_size = 128
    learning_rate = 1e-3
    weight_decay = 0.0
    max_workers = 50
    data_workers = 2
    cpu_threads_per_worker = 1
    output_dir = "results"
    seed = 1234
    max_train_samples = None
    use_cuda = False

    if use_cuda and max_workers > 1:
        raise ValueError("use_cuda=True only supports max_workers=1 for now.")

    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    configs: list[TrainConfig] = []
    run_id = 0
    for activation in activations:
        for model_type in model_types:
            for p in ps:
                for _ in range(repeats):
                    run_seed = random.randint(1, 1_000_000_000)
                    configs.append(
                        TrainConfig(
                            activation=activation,
                            model_type=model_type,
                            p=p,
                            epochs=epochs,
                            batch_size=batch_size,
                            learning_rate=learning_rate,
                            weight_decay=weight_decay,
                            seed=run_seed,
                            num_workers=data_workers,
                            cpu_threads=cpu_threads_per_worker,
                            max_train_samples=max_train_samples,
                            use_cuda=use_cuda,
                        )
                    )
                    run_id += 1

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Launching {len(configs)} runs with max_workers={max_workers}...")
    results: list[dict] = []
    seen_ps: set[float] = set()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(train_and_evaluate, cfg) for cfg in configs]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Runs", unit="run"):
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

    per_run_path = os.path.join(output_dir, "results_per_run.csv")
    summary_path = os.path.join(output_dir, "results_summary.csv")
    write_csv(per_run_path, results)

    summary_rows = summarize_runs(results)
    write_csv(summary_path, summary_rows)

    pdf_path = os.path.join(output_dir, "accuracy_vs_p.pdf")
    metadata = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "activations": activations,
        "model_types": model_types,
        "ps": ps,
        "repeats": repeats,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "max_workers": max_workers,
        "data_workers": data_workers,
        "cpu_threads_per_worker": cpu_threads_per_worker,
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


if __name__ == "__main__":
    main()
