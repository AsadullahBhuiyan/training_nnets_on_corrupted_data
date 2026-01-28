import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import statistics
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch
from tqdm import tqdm

from nnet_models import (
    TrainConfig,
    build_model,
    get_dataloaders,
    evaluate,
    apply_corruption,
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


def _worker_corruption_trials(
    model_state: dict,
    model_type: str,
    activation: str,
    mlp_hidden_sizes: list[int] | None,
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
    model = build_model(model_type, activation, mlp_hidden_sizes=mlp_hidden_sizes).to(device)
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
    mlp_hidden_sizes: list[int] | None,
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
        accuracies = []
        model.eval()
        with torch.no_grad():
            for _ in range(trials):
                correct = 0
                total = 0
                for x, y in test_loader:
                    x = x.to("cpu")
                    y = y.to("cpu")
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
                mlp_hidden_sizes,
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


def build_cache_key(
    corruption_trials: int,
    epochs: int,
    test_fraction: float,
    corruption_mode: str,
    ps: list[float],
    sigmas: list[float],
    widths: list[int],
    depths: list[int],
) -> str:
    if corruption_mode == "replacement":
        p_min, p_max = min(ps), max(ps)
        strength_tag = f"p{p_min:.2f}-{p_max:.2f}"
    else:
        s_min, s_max = min(sigmas), max(sigmas)
        strength_tag = f"s{s_min:.2f}-{s_max:.2f}"
    w_min, w_max = min(widths), max(widths)
    d_min, d_max = min(depths), max(depths)
    return (
        f"mlptest_r{corruption_trials}_e{epochs}_tf{test_fraction:.3f}_"
        f"{strength_tag}_w{w_min}-{w_max}_d{d_min}-{d_max}"
    )


def main() -> None:
    # Manual configuration (edit these values directly).
    activations = ["relu", "tanh", "sigmoid", "gelu"]
    model_type = "mlp"
    corruption_mode = "replacement"  # options: "replacement", "additive"
    ps = [0.8, 0.85, 0.9, 0.95, 1.0]
    sigmas = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    corruption_trials = 100
    epochs = 20
    batch_size = 128
    learning_rate = 1e-3
    weight_decay = 0.0
    loss_type = "cross_entropy"  # options: "cross_entropy", "quadratic"
    widths = [256]
    depths = [1, 2, 3]
    data_workers = 0
    max_workers = 8
    cpu_threads_per_worker = 1
    cpu_cores = None  # Example: [0, 1, 2, 3] to pin processes.
    max_train_samples = None
    use_cuda = False
    custom_split = False
    test_fraction = 0.2
    split_seed = 1234
    split_source = "train"
    output_dir = "results_noisy_test"
    suffix = None

    if use_cuda and max_workers > 1:
        raise ValueError("use_cuda=True only supports max_workers=1 for now.")

    apply_cpu_affinity(cpu_cores)
    torch.set_num_threads(max(1, cpu_threads_per_worker))
    torch.set_num_interop_threads(1)

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

    sweep_values = ps if corruption_mode == "replacement" else sigmas
    sweep_label = "p" if corruption_mode == "replacement" else "sigma"
    suffix_tag = f"_{suffix}" if suffix else ""
    suffix_note = f" (suffix={suffix})" if suffix else ""

    summary_rows = []
    for activation in activations:
        for width in widths:
            for depth in depths:
                mlp_hidden_sizes = [width] * depth
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
                    cpu_threads=cpu_threads_per_worker,
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
                    f"{suffix_note} (activation={activation}, w={width}, d={depth})"
                )

                train_loader, test_loader = get_dataloaders(
                    batch_size=cfg.batch_size,
                    p=cfg.p,
                    num_workers=cfg.num_workers,
                    max_train_samples=cfg.max_train_samples,
                    corruption_mode=cfg.corruption_mode,
                    sigma=cfg.sigma,
                    custom_split=cfg.custom_split,
                    test_fraction=cfg.test_fraction,
                    split_seed=cfg.split_seed,
                    split_source=cfg.split_source,
                )
                model = build_model(cfg.model_type, cfg.activation, mlp_hidden_sizes=mlp_hidden_sizes).to(device)
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=cfg.learning_rate,
                    weight_decay=cfg.weight_decay,
                )

                for _ in tqdm(
                    range(cfg.epochs),
                    desc=f"epochs ({activation}, w={width}, d={depth})",
                    unit="epoch",
                ):
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
                print(
                    f"[{activation} w={width} d={depth}] clean test accuracy: "
                    f"{clean_acc:.4f} (loss={clean_loss:.4f})"
                )

                for value in tqdm(
                    sweep_values,
                    desc=f"{sweep_label} sweep{suffix_note} ({activation}, w={width}, d={depth})",
                    unit="s",
                ):
                    p = value if corruption_mode == "replacement" else 0.0
                    sigma = value if corruption_mode == "additive" else 0.0
                    mean_acc, std_acc, stderr_acc = evaluate_with_corruption_parallel(
                        model=model,
                        model_type=model_type,
                        activation=activation,
                        mlp_hidden_sizes=mlp_hidden_sizes,
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
                    summary_rows.append(
                        {
                            "activation": activation,
                            "model_type": model_type,
                            "width": width,
                            "depth": depth,
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
                        f"[{activation} w={width} d={depth}] {sweep_label}={value:.3f} "
                        f"mean_acc={mean_acc:.4f} stderr={stderr_acc:.4f}"
                    )

    cache_key = build_cache_key(
        corruption_trials=corruption_trials,
        epochs=epochs,
        test_fraction=test_fraction,
        corruption_mode=corruption_mode,
        ps=ps,
        sigmas=sigmas,
        widths=widths,
        depths=depths,
    )
    pdf_path = os.path.join(output_dir, f"noisy_test_summary_{cache_key}{suffix_tag}.pdf")
    df = pd.DataFrame(summary_rows)

    with PdfPages(pdf_path) as pdf:
        for act in activations:
            sub_act = df[df["activation"] == act]
            fig, ax = plt.subplots(1, 1, figsize=(4, 3))
            for depth in depths:
                sub = sub_act[sub_act["depth"] == depth].sort_values(sweep_label)
                ax.errorbar(
                    sub[sweep_label],
                    sub["mean_test_accuracy"],
                    yerr=sub["stderr_test_accuracy"],
                    marker="o",
                    label=f"d={depth}",
                )
            ax.set_title(f"{model_type} / {act} (noisy test)")
            ax.set_xlabel(sweep_label)
            ax.set_ylabel("mean test accuracy")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        for depth in depths:
            sub = df[df["depth"] == depth].sort_values(sweep_label)
            ax.plot(sub[sweep_label], sub["std_test_accuracy"], marker="o", label=f"d={depth}")
        ax.set_title("std test accuracy")
        ax.set_xlabel(sweep_label)
        ax.set_ylabel("std test accuracy")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
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
            "widths": widths,
            "depths": depths,
            "data_workers": data_workers,
            "max_workers": max_workers,
            "cpu_threads_per_worker": cpu_threads_per_worker,
            "cpu_cores": cpu_cores,
            "custom_split": custom_split,
            "test_fraction": test_fraction,
            "split_seed": split_seed,
            "split_source": split_source,
            "output_dir": output_dir,
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


if __name__ == "__main__":
    main()
