import argparse
import json
from collections import defaultdict
from pathlib import Path


DATA_DIR = Path("data_processed")
OUTPUT_FILE = Path("splits.json")

N_FOLDS = 4
TEST_ROUND = 6

_parser = argparse.ArgumentParser(description="Create subject-disjoint train/validation/test split.")
_parser.add_argument("--fold", type=int, default=0, choices=range(N_FOLDS),
                     help="Which of the 4 development folds is used for validation.")
CURRENT_FOLD = _parser.parse_args().fold


def parse_filename(file_path):
    parts = file_path.stem.split("_")

    if len(parts) != 10:
        raise ValueError(f"Unknown format name: {file_path.name}")

    raw_user = parts[1].lstrip("S")
    user = f"{int(raw_user) % 1000:03d}"

    return {
        "user": user,
        "round": int(parts[3]),
        "session": parts[5],
        "task": parts[7],
        "window": int(parts[9]),
    }

def main():
    files = sorted(DATA_DIR.glob("*.pt"))

    if not files:
        raise FileNotFoundError(
            f"There is no .pt files in {DATA_DIR}"
        )

    metadata = {}

    recordings_by_user = defaultdict(set)

    for file_path in files:
        info = parse_filename(file_path)
        metadata[file_path] = info

        recording_id = (
            info["round"],
            info["session"],
            info["task"],
        )

        recordings_by_user[info["user"]].add(recording_id)

    test_users = {
        info["user"]
        for info in metadata.values()
        if info["round"] == TEST_ROUND
    }

    all_users = set(recordings_by_user)
    development_users = all_users - test_users

    recording_counts = {
        user: len(recordings_by_user[user])
        for user in development_users
    }


    sorted_users = sorted(
        development_users,
        key=lambda user: (-recording_counts[user], user),
    )

    folds = [[] for _ in range(N_FOLDS)]
    fold_recording_counts = [0] * N_FOLDS

    for user in sorted_users:
        fold_id = min(
            range(N_FOLDS),
            key=lambda i: (
                len(folds[i]),
                fold_recording_counts[i],
                i,
            ),
        )

        folds[fold_id].append(user)
        fold_recording_counts[fold_id] += recording_counts[user]

    validation_users = set(folds[CURRENT_FOLD])

    train_users = development_users - validation_users

    train_files = []
    validation_files = []
    test_files = []

    excluded_blg = 0

    for file_path in files:
        info = metadata[file_path]
        user = info["user"]
        task = info["task"].upper()

        if user in test_users:
            test_files.append(file_path.name)

        elif task == "BLG":
            excluded_blg += 1

        elif user in validation_users:
            validation_files.append(file_path.name)

        elif user in train_users:
            train_files.append(file_path.name)

    assert train_users.isdisjoint(validation_users)
    assert train_users.isdisjoint(test_users)
    assert validation_users.isdisjoint(test_users)

    result = {
        "current_fold": CURRENT_FOLD,
        "folds": {
            str(i): sorted(folds[i])
            for i in range(N_FOLDS)
        },
        "train_users": sorted(train_users),
        "validation_users": sorted(validation_users),
        "test_users": sorted(test_users),
        "train_files": sorted(train_files),
        "validation_files": sorted(validation_files),
        "test_files": sorted(test_files),
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print(f"Total users: {len(all_users)}")
    print(f"Development users: {len(development_users)}")
    print(f"Test users: {len(test_users)}")
    print()

    for i in range(N_FOLDS):
        print(
            f"Fold {i}: "
            f"{len(folds[i])} users, "
            f"{fold_recording_counts[i]} recordings"
        )

    print()
    print(f"Train windows: {len(train_files)}")
    print(f"Validation windows: {len(validation_files)}")
    print(f"Test windows: {len(test_files)}")
    print(f"Excluded BLG windows: {excluded_blg}")
    print()
    print(f"Saved to: {OUTPUT_FILE}")

    if len(test_users) != 59:
        print(
            f"Warning: expected 59 test users, found {len(test_users)}."
        )


if __name__ == "__main__":
    main()