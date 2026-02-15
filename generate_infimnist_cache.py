import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm


def import_infimnist_module():
    try:
        import _infimnist as infimnist  # type: ignore
        return infimnist
    except ImportError:
        local_path = os.path.join(os.path.dirname(__file__), "infimnist_py")
        if os.path.isdir(local_path) and local_path not in sys.path:
            sys.path.insert(0, local_path)
        import _infimnist as infimnist  # type: ignore
        return infimnist


def validate_infimnist_data_dir(infimnist_module) -> str:
    module_dir = os.path.dirname(infimnist_module.__file__)
    data_dir = os.path.join(module_dir, "data")
    required = [
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
        "fields_float_1522x28x28.bin",
        "tangVec_float_60000x28x28.bin",
    ]
    missing = [name for name in required if not os.path.exists(os.path.join(data_dir, name))]
    if missing:
        raise FileNotFoundError(
            f"InfiMNIST data files missing under '{data_dir}'. Missing: {missing}"
        )
    return data_dir


def main() -> None:
    # Manual configuration.
    total_images = 20_000_000
    chunk_size = 100_000
    alpha = 1.0
    translate = True
    output_dir = Path("data/infimnist_cache_2e7")
    overwrite = False
    start_index = 0

    if total_images <= 0:
        raise ValueError("total_images must be > 0.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0.")

    infimnist = import_infimnist_module()
    data_dir = validate_infimnist_data_dir(infimnist)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] using infimnist data dir: {data_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    images_path = output_dir / "images_uint8.npy"
    labels_path = output_dir / "labels_uint8.npy"
    meta_path = output_dir / "metadata.json"

    existing = [p for p in [images_path, labels_path, meta_path] if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Cache files already exist. Set overwrite=True to replace them. "
            f"Existing: {[str(p) for p in existing]}"
        )

    images_mm = np.lib.format.open_memmap(
        images_path,
        mode="w+",
        dtype=np.uint8,
        shape=(total_images, 28, 28),
    )
    labels_mm = np.lib.format.open_memmap(
        labels_path,
        mode="w+",
        dtype=np.uint8,
        shape=(total_images,),
    )

    generator = infimnist.InfimnistGenerator(alpha=alpha, translate=translate)
    pbar = tqdm(total=total_images, desc="Generating InfiMNIST cache", unit="img")
    for start in range(0, total_images, chunk_size):
        end = min(start + chunk_size, total_images)
        idx = np.arange(start_index + start, start_index + end, dtype=np.int64)
        digits, labels = generator.gen(idx)
        images_mm[start:end] = digits.reshape(end - start, 28, 28).astype(np.uint8, copy=False)
        labels_mm[start:end] = labels.astype(np.uint8, copy=False)
        pbar.update(end - start)
    pbar.close()

    images_mm.flush()
    labels_mm.flush()

    metadata = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_images": total_images,
        "shape": [28, 28],
        "images_dtype": "uint8",
        "labels_dtype": "uint8",
        "alpha": alpha,
        "translate": translate,
        "start_index": start_index,
        "source_data_dir": data_dir,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] saved cache:")
    print(f"  {images_path}")
    print(f"  {labels_path}")
    print(f"  {meta_path}")


if __name__ == "__main__":
    main()
