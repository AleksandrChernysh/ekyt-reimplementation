import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


_parser = argparse.ArgumentParser(description="Plot training curves and score distributions for one fold.")
_parser.add_argument("--fold", type=int, default=0)
_args = _parser.parse_args()

_log_versions = sorted(Path(f"lightning_logs/fold{_args.fold}").glob("version_*/metrics.csv"))
if not _log_versions:
    raise FileNotFoundError(f"No metrics.csv found under lightning_logs/fold{_args.fold}/")
METRICS_FILE = _log_versions[-1]
EVALUATION_FILE = f"results/evaluation_fold{_args.fold}.npz"

PLOTS_DIR = Path("plots") / f"fold{_args.fold}"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

metrics = pd.read_csv(METRICS_FILE)

val = metrics.dropna(subset=["val_map_at_r"])

plt.figure(figsize=(8, 5))

plt.plot(
    val["epoch"],
    val["val_map_at_r"]
)

plt.xlabel("Epoch")
plt.ylabel("Validation MAP@R")
plt.title("Validation MAP@R During Training")
_best = val.loc[val["val_map_at_r"].idxmax()]
plt.scatter(
    _best["epoch"],
    _best["val_map_at_r"],
    label=f"Best validation (epoch {int(_best['epoch'])})"
)

plt.legend()


plt.tight_layout()
plt.savefig(
    PLOTS_DIR / "val_map_at_r.png",
    dpi=300
)
plt.close()

train = metrics.dropna(
    subset=["train_pos_cos", "train_neg_cos"]
)

plt.figure(figsize=(8, 5))

plt.plot(
    train["epoch"],
    train["train_pos_cos"],
    label="Positive cosine"
)

plt.plot(
    train["epoch"],
    train["train_neg_cos"],
    label="Negative cosine"
)

plt.xlabel("Epoch")
plt.ylabel("Mean Cosine Similarity")
plt.title("Training Embedding Separation")

plt.legend()
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "train_cosine_separation.png",
    dpi=300
)

plt.close()


evaluation = np.load(EVALUATION_FILE)

genuine = evaluation["genuine_scores"]
impostor = evaluation["impostor_scores"]
eer_threshold = float(
    evaluation["eer_threshold"]
)

plt.figure(figsize=(8, 5))

plt.hist(
    impostor,
    bins=40,
    density=True,
    alpha=0.6,
    label="Impostor"
)

plt.hist(
    genuine,
    bins=20,
    density=True,
    alpha=0.6,
    label="Genuine"
)

plt.axvline(
    eer_threshold,
    linestyle="--",
    label=f"EER threshold = {eer_threshold:.3f}"
)

plt.xlabel("Cosine Similarity")
plt.ylabel("Density")
plt.title("Genuine and Impostor Score Distributions")

plt.legend()
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "score_distributions.png",
    dpi=300
)

plt.close()

fmr = evaluation["fmr"]
tmr = evaluation["tmr"]

plt.figure(figsize=(6, 6))

plt.plot(
    fmr,
    tmr
)

plt.xlabel("False Match Rate")
plt.ylabel("True Match Rate")
plt.title("ROC Curve")

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "roc_curve.png",
    dpi=300
)

plt.close()