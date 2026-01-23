# Project Notes

## Resource Estimates (CPU Only)

These are approximate costs for the "reliable" setup: full MNIST (60k/10k), MLP (1024-512-256), CNN (32-64-128), batch size 256-512, 10 p-values, 40 repeats.

### Per-run (single training job)

- MLP, 15 epochs, batch 512: ~1–3 minutes CPU time; ~0.3–0.8 GB RAM.
- CNN, 20 epochs, batch 256: ~2–6 minutes CPU time; ~0.5–1.2 GB RAM.

### Aggregate campaign example

Assuming 3 activations × 2 model types × 10 p-values × 40 repeats = 2,400 runs.

- Total CPU time (serial):
  - MLP: ~2,400–7,200 minutes.
  - CNN: ~4,800–14,400 minutes.
- With ~80 workers:
  - MLP: ~1–3 hours wall time.
  - CNN: ~2–6 hours wall time.
- Peak RAM with ~80 workers: ~40–100 GB.

### Scaling rules of thumb

- Runtime scales ~linearly with `epochs` and `repeats`.
- Peak RAM scales ~linearly with `max_workers`.
- Larger batch sizes increase per-worker RAM but can improve throughput.

## File Descriptions

### `nnet_models.py`

Core modeling and training utilities.

- `TrainConfig`: Dataclass collecting all hyperparameters for a single run.
- `set_seed`, `get_device`: Seed setup and CPU/GPU selection.
- `get_activation`: Maps string names to PyTorch activation modules.
- `MLP`: Fully connected MNIST classifier (default hidden sizes `(256, 128)`).
- `SimpleCNN`: Small conv net with two conv blocks + MLP head.
- `build_model`: Chooses `mlp` or `cnn` based on a string.
- `corrupt_pixels`: Replaces each pixel with a random uniform value in `[0,1]` with probability `p`.
- `CorruptedDataset`: Wraps a dataset and injects corruption on each training sample.
- `get_dataloaders`: Loads MNIST, wraps training set in `CorruptedDataset`, returns loaders.
- `train_one_epoch`: Single epoch training loop with cross-entropy loss.
- `evaluate`: Computes test loss and accuracy on the clean test set.
- `train_and_evaluate`: Runs a full training + evaluation; returns a results dict.

### `run_experiments.py`

Parallel experiment runner with manual configuration.

- Manual config block (edit inside `main()`):
  - `activations`, `model_types`, `ps`, `repeats`, `epochs`, `batch_size`.
  - `learning_rate`, `weight_decay`, `max_workers`, `data_workers`.
  - `cpu_threads_per_worker`, `output_dir`, `seed`, `max_train_samples`, `use_cuda`.
- Builds all `TrainConfig` combinations and dispatches them via `ProcessPoolExecutor`.
- Writes per-run and aggregated CSVs:
  - `results/results_per_run.csv`
  - `results/results_summary.csv`
- Prints progress with p and accuracy for each finished job.

### `analysis.ipynb`

Notebook for visualization.

- Loads `results/results_summary.csv`.
- For each model type, produces one subplot per activation.
- Plots mean test accuracy vs p with error bars (standard error).

### `packages.md`

Minimal environment dependency list.

- Conda example (CPU-only) and pip alternative.
- Includes: `python`, `pytorch`, `torchvision`, `numpy`, `pandas`, `matplotlib`, `jupyter`.

## Tunable Hyperparameters

These are set in the manual configuration block inside `run_experiments.py`.

- `activations`: List of activation names (e.g., `["relu", "tanh"]`).
- `model_types`: List of model families (`"mlp"` or `"cnn"`).
- `ps`: Corruption probabilities to sweep over (list of floats in `[0, 1]`).
- `repeats`: Number of independent runs per configuration.
- `epochs`: Training epochs per run.
- `batch_size`: Training batch size.
- `learning_rate`: Optimizer learning rate.
- `weight_decay`: L2 regularization.
- `max_workers`: Number of parallel training processes.
- `data_workers`: DataLoader worker processes per training job.
- `cpu_threads_per_worker`: CPU threads allocated to each training job.
- `output_dir`: Directory for CSV outputs.
- `seed`: Global seed used to generate per-run seeds.
- `max_train_samples`: Optional training-set cap for quick tests.
- `use_cuda`: Whether to use GPU (single worker only).

## `run_experiments.py` Step-by-Step Walkthrough

This section explains the `run_experiments.py` script linearly, highlighting key function calls and variable assignments.

1. Imports core modules and training entry points.
   - `from concurrent.futures import ProcessPoolExecutor, as_completed`
   - `from nnet_models import TrainConfig, train_and_evaluate`

2. Defines `write_csv(path: str, rows: list[dict])`.
   - Opens a file with `open(path, "w", newline="")`.
   - Uses `csv.DictWriter(...).writeheader()` and `writerows(rows)` to persist results.

3. Defines `summarize_runs(rows: list[dict])`.
   - Creates grouping dictionary: `grouped: dict[tuple[str, str, float], list[dict]] = {}`.
   - For each row, computes key: `key = (row["activation"], row["model_type"], float(row["p"]))`.
   - Builds summary rows with:
     - `statistics.mean(accs)` and `statistics.pstdev(accs)` for `mean_test_accuracy` / `std_test_accuracy`.
     - Standard error as `stderr_test_accuracy = std_test_accuracy / sqrt(repeats)`.
   - Sorts with `summary_rows.sort(key=lambda r: (r["model_type"], r["activation"], r["p"]))`.

4. Enters `main()` and sets the manual configuration.
   - Example assignments:
     - `activations = ["relu", "tanh", "sigmoid"]`
     - `model_types = ["mlp"]`
     - `ps = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]`
     - `repeats = 5`, `epochs = 5`, `batch_size = 128`
     - `learning_rate = 1e-3`, `weight_decay = 0.0`
     - `max_workers = os.cpu_count() or 2`
     - `data_workers = 2`, `cpu_threads_per_worker = 1`
     - `output_dir = "results"`, `seed = 1234`
     - `max_train_samples = None`, `use_cuda = False`

5. Enforces a GPU constraint when parallelized.
   - If `use_cuda` is `True` and `max_workers > 1`, raises:
     - `ValueError("use_cuda=True only supports max_workers=1 for now.")`

6. Initializes randomness and output directory.
   - `random.seed(seed)`
   - `os.makedirs(output_dir, exist_ok=True)`

7. Builds a list of `TrainConfig` objects.
   - Initializes `configs: list[TrainConfig] = []` and `run_id = 0`.
   - Nested loops over `activations`, `model_types`, `ps`, and `repeats`.
   - Picks a per-run seed: `run_seed = random.randint(1, 1_000_000_000)`.
   - Appends:
     - `TrainConfig(activation=..., model_type=..., p=..., epochs=..., batch_size=..., learning_rate=..., weight_decay=..., seed=..., num_workers=data_workers, cpu_threads=cpu_threads_per_worker, max_train_samples=max_train_samples, use_cuda=use_cuda)`

8. Launches parallel training jobs.
   - Prints: `print(f"Launching {len(configs)} runs with max_workers={max_workers}...")`
   - Creates executor: `ProcessPoolExecutor(max_workers=max_workers)`
   - Submits jobs: `executor.submit(train_and_evaluate, cfg)` for each config.
   - Iterates `for future in as_completed(futures)`:
     - `res = future.result()` collects result dict.
     - Appends to `results`.
     - Prints per-run status using `res["model_type"]`, `res["activation"]`, `res["p"]`, `res["test_accuracy"]`.

9. Writes outputs to CSV.
   - Defines paths:
     - `per_run_path = os.path.join(output_dir, "results_per_run.csv")`
     - `summary_path = os.path.join(output_dir, "results_summary.csv")`
   - Calls:
     - `write_csv(per_run_path, results)`
     - `summary_rows = summarize_runs(results)`
     - `write_csv(summary_path, summary_rows)`
   - Prints saved file locations.

10. Entry point for execution.
    - `if __name__ == "__main__": main()`

## `nnet_models.py` Step-by-Step Walkthrough

This section explains the `nnet_models.py` script linearly, highlighting key function calls and variable assignments.

1. Imports core modules, PyTorch, and torchvision.
   - `import torch`
   - `from torch import nn`
   - `from torchvision import datasets, transforms`

2. Defines `TrainConfig` dataclass.
   - Fields include `activation`, `model_type`, `p`, `epochs`, `batch_size`, `learning_rate`, `weight_decay`, `seed`, `num_workers`, `cpu_threads`, `max_train_samples`, `use_cuda`.

3. Defines `set_seed(seed: int)`.
   - `random.seed(seed)`
   - `torch.manual_seed(seed)`
   - `torch.cuda.manual_seed_all(seed)`

4. Defines `get_device(use_cuda: bool)`.
   - Returns `torch.device("cuda")` if `use_cuda` and CUDA is available; otherwise `torch.device("cpu")`.

5. Defines `get_activation(name: str)`.
   - Maps strings to modules:
     - `"relu" -> nn.ReLU()`
     - `"tanh" -> nn.Tanh()`
     - `"sigmoid" -> nn.Sigmoid()`
     - `"leaky_relu" -> nn.LeakyReLU(0.1)`
     - `"gelu" -> nn.GELU()`
   - Raises `ValueError` if unsupported.

6. Defines `MLP` model class.
   - Constructor:
     - `layers = [nn.Flatten()]`
     - For each hidden size: appends `nn.Linear(in_dim, h)` and `get_activation(activation)`.
     - Final layer: `nn.Linear(in_dim, 10)`.
   - `forward` returns `self.net(x)`.

7. Defines `SimpleCNN` model class.
   - `self.features`:
     - `nn.Conv2d(1, 32, kernel_size=3, padding=1)`
     - `get_activation(activation)`
     - `nn.MaxPool2d(2)`
     - `nn.Conv2d(32, 64, kernel_size=3, padding=1)`
     - `get_activation(activation)`
     - `nn.MaxPool2d(2)`
   - `self.classifier`:
     - `nn.Flatten()`
     - `nn.Linear(64 * 7 * 7, 128)`
     - `get_activation(activation)`
     - `nn.Linear(128, 10)`
   - `forward` applies `features` then `classifier`.

8. Defines `build_model(model_type: str, activation: str)`.
   - Returns `MLP(activation)` if `model_type == "mlp"`.
   - Returns `SimpleCNN(activation)` if `model_type == "cnn"`.
   - Raises `ValueError` if unsupported.

9. Defines `corrupt_pixels(x: torch.Tensor, p: float)`.
   - If `p <= 0`, returns `x` unchanged.
   - Otherwise: `mask = torch.rand_like(x) < p`.
   - Replaces masked pixels with `torch.rand_like(x)` via `torch.where`.

10. Defines `CorruptedDataset` wrapper.
    - Stores `base` and `p`.
    - `__getitem__`:
      - Reads `(x, y) = self.base[idx]`.
      - Applies `x = corrupt_pixels(x, self.p)`.
      - Returns `(x, y)`.

11. Defines `_limit_dataset(base: Dataset, max_samples: int | None)`.
    - If `max_samples` is `None`, returns `base`.
    - Otherwise returns `torch.utils.data.Subset(base, list(range(max_samples)))`.

12. Defines `get_dataloaders(batch_size, p, num_workers, max_train_samples)`.
    - `transform = transforms.ToTensor()`
    - Loads MNIST:
      - `datasets.MNIST(..., train=True, download=True, transform=transform)`
      - `datasets.MNIST(..., train=False, download=True, transform=transform)`
    - Applies `_limit_dataset` to training set if requested.
    - Wraps training set with `CorruptedDataset(train_base, p)`.
    - Builds `DataLoader` for train/test with `batch_size`, `shuffle`, and `num_workers`.

13. Defines `train_one_epoch(model, loader, optimizer, device)`.
    - `criterion = nn.CrossEntropyLoss()`
    - For each batch:
      - Moves to device: `x = x.to(device)`, `y = y.to(device)`
      - Clears grads: `optimizer.zero_grad(set_to_none=True)`
      - Forward: `logits = model(x)`
      - Loss: `loss = criterion(logits, y)`
      - Backprop: `loss.backward()`
      - Step: `optimizer.step()`
    - Returns average loss over dataset.

14. Defines `evaluate(model, loader, device)`.
    - `model.eval()` and `torch.no_grad()`.
    - Computes loss and accuracy.
    - Returns `(avg_loss, accuracy)`.

15. Defines `train_and_evaluate(cfg: TrainConfig)`.
    - Sets CPU threads: `torch.set_num_threads(max(1, cfg.cpu_threads))`.
    - Seeds: `set_seed(cfg.seed)`.
    - Chooses device: `device = get_device(cfg.use_cuda)`.
    - Builds loaders: `get_dataloaders(...)`.
    - Builds model: `build_model(cfg.model_type, cfg.activation).to(device)`.
    - Optimizer: `torch.optim.Adam(...)`.
    - Trains for `cfg.epochs` epochs using `train_one_epoch`.
    - Evaluates on clean test set with `evaluate`.
    - Returns a results dict including `test_accuracy`.
