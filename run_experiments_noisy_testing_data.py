import os
import statistics
from datetime import datetime

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from nnet_models import TrainConfig, build_model, get_dataloaders, evaluate, corrupt_pixels


def evaluate_with_corruption(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    p: float,
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
                x = corrupt_pixels(x, p)
                logits = model(x)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
            accuracies.append(correct / total)
    mean_acc = statistics.mean(accuracies)
    std_acc = statistics.pstdev(accuracies) if len(accuracies) > 1 else 0.0
    stderr_acc = std_acc / (len(accuracies) ** 0.5) if len(accuracies) > 1 else 0.0
    return mean_acc, std_acc, stderr_acc


def main() -> None:
    # Manual configuration (edit these values directly).
    activation = "relu"
    model_type = "mlp"  # options: "mlp", "cnn"
    ps = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    corruption_trials = 100
    epochs = 5
    batch_size = 128
    learning_rate = 1e-3
    weight_decay = 0.0
    data_workers = 2
    max_train_samples = None
    use_cuda = False

    output_dir = "results_noisy_test"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

    cfg = TrainConfig(
        activation=activation,
        model_type=model_type,
        p=0.0,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=1234,
        num_workers=data_workers,
        cpu_threads=1,
        max_train_samples=max_train_samples,
        use_cuda=use_cuda,
    )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] training model once for noisy-test evaluation...")

    train_loader, test_loader = get_dataloaders(
        batch_size=cfg.batch_size,
        p=cfg.p,
        num_workers=cfg.num_workers,
        max_train_samples=cfg.max_train_samples,
    )
    model = build_model(cfg.model_type, cfg.activation).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    for _ in tqdm(range(cfg.epochs), desc="epochs", unit="epoch"):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()

    clean_loss, clean_acc = evaluate(model, test_loader, device)
    print(f"clean test accuracy: {clean_acc:.4f} (loss={clean_loss:.4f})")

    summary_rows = []
    for p in tqdm(ps, desc="p sweep", unit="p"):
        mean_acc, std_acc, stderr_acc = evaluate_with_corruption(
            model,
            test_loader,
            device,
            p=p,
            trials=corruption_trials,
        )
        summary_rows.append(
            {
                "activation": activation,
                "model_type": model_type,
                "p": p,
                "corruption_trials": corruption_trials,
                "mean_test_accuracy": mean_acc,
                "std_test_accuracy": std_acc,
                "stderr_test_accuracy": stderr_acc,
            }
        )
        print(f"p={p:.3f} mean_acc={mean_acc:.4f} stderr={stderr_acc:.4f}")

    # Plot
    ps_sorted = [row["p"] for row in summary_rows]
    means = [row["mean_test_accuracy"] for row in summary_rows]
    stderrs = [row["stderr_test_accuracy"] for row in summary_rows]

    plt.figure(figsize=(5, 3))
    plt.errorbar(ps_sorted, means, yerr=stderrs, marker="o")
    plt.title(f"{model_type} / {activation} (noisy test)")
    plt.xlabel("p")
    plt.ylabel("mean test accuracy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "accuracy_vs_p_noisy_test.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
