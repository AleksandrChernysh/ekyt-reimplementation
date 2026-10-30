"""Aggregate per-fold evaluation results into mean ± std.

Run after `evaluate.py --fold k` has been run for every fold.
"""
from pathlib import Path

import numpy as np


def fnmr_at_fmr(fmr, tmr, target):
    index = np.where(fmr <= target)[0][-1]
    return 1 - tmr[index]


def main():
    files = sorted(Path("results").glob("evaluation_fold*.npz"))
    if not files:
        raise SystemExit("No results/evaluation_fold*.npz files found. Run evaluate.py first.")

    rows = []
    for path in files:
        data = np.load(path)
        rows.append({
            "fold": path.stem.replace("evaluation_fold", ""),
            "eer": float(data["eer"]) * 100,
            "rank1": float(data["rank1"]) * 100,
            "fnmr_1": fnmr_at_fmr(data["fmr"], data["tmr"], 0.01) * 100,
            "genuine": len(data["genuine_scores"]),
            "impostor": len(data["impostor_scores"]),
        })

    print("| Fold | EER (%) | Rank-1 (%) | FNMR @ 1% FMR (%) | Genuine / impostor pairs |")
    print("|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['fold']} | {r['eer']:.2f} | {r['rank1']:.2f} | {r['fnmr_1']:.2f} "
              f"| {r['genuine']} / {r['impostor']} |")

    if len(rows) > 1:
        def ms(key):
            values = np.array([r[key] for r in rows])
            return f"{values.mean():.2f} ± {values.std(ddof=1):.2f}"
        print(f"| **Mean ± SD** | {ms('eer')} | {ms('rank1')} | {ms('fnmr_1')} | |")


if __name__ == "__main__":
    main()
