import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_metric_learning.samplers import MPerClassSampler
from pathlib import Path
import json

class GazeDataset(Dataset):
    def __init__(
        self,
        file_list,
        label_map,
        mean,
        std,
        data_dir="data_processed"
    ):
        self.files = [Path(data_dir) / f for f in file_list]
        self.label_map = label_map
        self.mean = mean
        self.std = std


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]

        tensor = torch.load(file_path)
        tensor = (tensor - self.mean) / self.std

        parts = file_path.stem.split("_")

        raw_user = parts[1].lstrip("S")
        user_id = f"{int(raw_user) % 1000:03d}"

        mapped_label = self.label_map.get(user_id, -1)

        label = torch.tensor(mapped_label, dtype=torch.long)

        return tensor, label, user_id
    
def load_split_info():
    with open("splits.json", "r", encoding="utf-8") as file:
        splits = json.load(file)

    train_users = splits["train_users"]

    label_map = {
        user_id: label
        for label, user_id in enumerate(train_users)
    }

    return splits, label_map

def load_normalization():
    with open("normalization.json", "r", encoding="utf-8") as file:
        normalization = json.load(file)

    return normalization["mean"], normalization["std"]

def create_dataloaders(batch_size=32, num_workers=0):
    splits, label_map = load_split_info()
    mean, std = load_normalization()

    train_dataset = GazeDataset(
        splits["train_files"],
        label_map,
        mean,
        std
    )

    val_dataset = GazeDataset(
        splits["validation_files"],
        label_map,
        mean,
        std
    )

    test_dataset = GazeDataset(
        splits["test_files"],
        label_map,
        mean,
        std
    )


    train_labels = []

    for file_path in train_dataset.files:
        parts = file_path.stem.split("_")

        raw_user = parts[1].lstrip("S")
        user_id = f"{int(raw_user) % 1000:03d}"

        train_labels.append(label_map[user_id])


    train_sampler = MPerClassSampler(
        labels=train_labels,
        m=4,
        batch_size=batch_size,
        length_before_new_iter=len(train_dataset)
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0
    )

    return train_loader, val_loader, test_loader, label_map


if __name__ == "__main__":
    train_loader, val_loader, test_loader, label_map = create_dataloaders()
    signals, labels, user_ids = next(iter(train_loader))

    unique_labels, counts = torch.unique(
        labels,
        return_counts=True
    )

    print("Labels:", labels)
    print("Unique labels:", unique_labels)
    print("Counts:", counts)