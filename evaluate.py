import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import roc_curve

from dataset import load_split_info, load_normalization
from train import EyeKnowYouTooLightning
from quality import get_valid_users

import argparse

DATA_DIR = Path("data_processed")

_parser = argparse.ArgumentParser(description="Evaluate a trained fold on the held-out Round 6 subjects.")
_parser.add_argument("--fold", type=int, default=0)
_parser.add_argument("--checkpoint", default=None,
                     help="Defaults to the final-epoch checkpoint of the fold (no test-set model selection).")
ARGS = _parser.parse_args()
CHECKPOINT = ARGS.checkpoint or f"checkpoints/fold{ARGS.fold}/last.ckpt"


def parse_filename(filename):
    parts = Path(filename).stem.split("_")

    raw_user = parts[1].lstrip("S")
    user_id = f"{int(raw_user) % 1000:03d}"

    return {
        "user": user_id,
        "round": int(parts[3]),
        "session": parts[5].upper(),
        "task": parts[7].upper(),
        "window": int(parts[9]),
    }


def load_batch(file_list, mean, std):
    tensors = []

    for filename in file_list:
        tensor = torch.load(DATA_DIR / filename)
        tensor = (tensor - mean) / std
        tensors.append(tensor)

    return torch.stack(tensors)


def get_embeddings(model, signals, device):
    signals = signals.to(device)

    with torch.inference_mode():
        embeddings, _ = model.model(signals)

    embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings.cpu()

def get_fnmr_at_fmr(fmr, fnmr, thresholds, target_fmr):
    valid_indices = np.where(fmr <= target_fmr)[0]
    index = valid_indices[-1]

    return (
        fmr[index],
        fnmr[index],
        thresholds[index],
    )

def main():

    # -------------------------
    # Load split + normalization
    # -------------------------

    splits, label_map = load_split_info()
    mean, std = load_normalization()

    if splits["current_fold"] != ARGS.fold:
        raise SystemExit(
            f"splits.json was made for fold {splits['current_fold']}, not {ARGS.fold}. "
            f"Run: python split.py --fold {ARGS.fold} && python normalize.py"
        )


    # -------------------------
    # Select EKYT test protocol
    #
    # Enrollment:
    # R1 / S1 / TEX / first 5 s
    #
    # Probe:
    # R1 / S2 / TEX / first 5 s
    # -------------------------

    enrollment_files = {}
    probe_files = {}

    test_users = set(splits["test_users"])

    valid_enrollment_users = get_valid_users("S1", test_users)
    valid_probe_users = get_valid_users("S2", test_users)

    for filename in splits["test_files"]:
        info = parse_filename(filename)
        if (
            info["round"] == 1
            and info["task"] == "TEX"
            and info["window"] == 0
        ):
            if (
                info["session"] == "S1"
                and info["user"] in valid_enrollment_users
            ):
                enrollment_files[info["user"]] = filename

            elif (
                info["session"] == "S2"
                and info["user"] in valid_probe_users
            ):
                probe_files[info["user"]] = filename


    enrollment_users = sorted(enrollment_files)
    probe_users = sorted(probe_files)

    enrollment_names = [
        enrollment_files[user]
        for user in enrollment_users
    ]

    probe_names = [
        probe_files[user]
        for user in probe_users
    ]


    # -------------------------
    # Load model
    # -------------------------

    model = EyeKnowYouTooLightning.load_from_checkpoint(
        CHECKPOINT,
        num_classes=len(label_map)
    )

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    model = model.to(device)
    model.eval()


    # -------------------------
    # Load signals
    # -------------------------

    enrollment_signals = load_batch(
        enrollment_names,
        mean,
        std
    )

    probe_signals = load_batch(
        probe_names,
        mean,
        std
    )


    # -------------------------
    # Create 128D embeddings
    # -------------------------

    enrollment_embeddings = get_embeddings(
        model,
        enrollment_signals,
        device
    )

    probe_embeddings = get_embeddings(
        model,
        probe_signals,
        device
    )


    # -------------------------
    # Cosine similarity
    #
    # Rows = enrollment users
    # Columns = probe users
    # -------------------------

    similarity_matrix = (
        enrollment_embeddings
        @ probe_embeddings.T
    )



    enrollment_ids = np.array(enrollment_users)

    probe_ids = np.array(probe_users)

    same_user = (
        enrollment_ids[:, None]
        == probe_ids[None, :]
    )

    all_scores = similarity_matrix.numpy()

    genuine_scores = all_scores[same_user]
    impostor_scores = all_scores[~same_user]


    # -------------------------
    # EER
    # -------------------------

    labels = np.concatenate([
        np.ones(len(genuine_scores)),
        np.zeros(len(impostor_scores)),
    ])

    scores = np.concatenate([
        genuine_scores,
        impostor_scores,
    ])

    fmr, tmr, thresholds = roc_curve(
        labels,
        scores
    )

    fnmr = 1 - tmr


    eer_index = np.argmin(
        np.abs(fmr - fnmr)
    )

    eer = (
        fmr[eer_index]
        + fnmr[eer_index]
    ) / 2

    eer_threshold = thresholds[eer_index]

    fmr_1, fnmr_1, threshold_1 = get_fnmr_at_fmr(
    fmr,
    fnmr,
    thresholds,
    0.01
    )

    fmr_01, fnmr_01, threshold_01 = get_fnmr_at_fmr(
        fmr,
        fnmr,
        thresholds,
        0.001
    )


    # -------------------------
    # Rank-1 identification
    # -------------------------

    predicted_indices = similarity_matrix.argmax(dim=0)

    predicted_users = [
        enrollment_users[index]
        for index in predicted_indices.tolist()
    ]

    rank1_results = []

    for predicted_user, probe_user in zip(
        predicted_users,
        probe_users
    ):
        if probe_user in enrollment_files:
            rank1_results.append(
                predicted_user == probe_user
            )

    rank1 = np.mean(rank1_results)


    # -------------------------
    # Results
    # -------------------------

    print()
    print("=== EKYT simplified evaluation ===")
    print()

    print(
    "Enrollment users:",
    len(enrollment_users)
)

    print(
        "Probe users:",
        len(probe_users)
    )

    print(
        "Genuine scores:",
        len(genuine_scores)
    )

    print(
        "Impostor scores:",
        len(impostor_scores)
    )

    print()

    print(
        f"Mean genuine cosine: "
        f"{genuine_scores.mean():.4f}"
    )

    print(
        f"Mean impostor cosine: "
        f"{impostor_scores.mean():.4f}"
    )

    print()

    print(
        f"EER: {eer * 100:.2f}%"
    )

    print(
        f"EER threshold: "
        f"{eer_threshold:.4f}"
    )

    print(
        f"Rank-1: {rank1 * 100:.2f}%"
    )

    print(
    f"FNMR @ 1% FMR: {fnmr_1 * 100:.2f}% "
    f"(actual FMR: {fmr_1 * 100:.3f}%, "
    f"threshold: {threshold_1:.4f})"
)

    print(
        f"FNMR @ 0.1% FMR: {fnmr_01 * 100:.2f}% "
        f"(actual FMR: {fmr_01 * 100:.3f}%, "
        f"threshold: {threshold_01:.4f})"
    )


    RESULTS_DIR = Path("results")
    RESULTS_DIR.mkdir(exist_ok=True)

    np.savez(
        RESULTS_DIR / f"evaluation_fold{ARGS.fold}.npz",
        genuine_scores=genuine_scores,
        impostor_scores=impostor_scores,
        fmr=fmr,
        tmr=tmr,
        thresholds=thresholds,
        eer=eer,
        eer_threshold=eer_threshold,
        rank1=rank1,
    )
if __name__ == "__main__":
    main()