import json
import torch
from pathlib import Path


DATA_DIR = Path("data_processed")
SPLITS_FILE = Path("splits.json")
OUTPUT_FILE = Path("normalization.json")


def main():
    with open(SPLITS_FILE, "r", encoding="utf-8") as file:
        splits = json.load(file)

    train_files = splits["train_files"]

    total_sum = 0.0
    total_squared_sum = 0.0
    total_count = 0

    for filename in train_files:
        tensor = torch.load(DATA_DIR / filename).double()

        total_sum += tensor.sum().item()
        total_squared_sum += (tensor ** 2).sum().item()
        total_count += tensor.numel()

    mean = total_sum / total_count
    variance = total_squared_sum / total_count - mean ** 2
    std = variance ** 0.5

    normalization = {
        "mean": mean,
        "std": std
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(normalization, file, indent=2)

    print("Train mean:", mean)
    print("Train std:", std)
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()