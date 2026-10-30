# Eye Know You Too — PyTorch Lightning re-implementation

A from-scratch re-implementation of **Eye Know You Too (EKYT)**, a deep metric-learning model that identifies people from their eye movements, trained and evaluated on the public **GazeBase** dataset.

Original method: D. J. Lohr and O. V. Komogortsev, *"Eye Know You Too: Toward Viable End-to-End Eye Movement Biometrics for User Authentication,"* IEEE Transactions on Information Forensics and Security, 2022.
This repository is an independent learning re-implementation, not the authors' code.

## What it does

1. **Preprocessing** (`preprocess.py`): reads 1000 Hz GazeBase recordings, interpolates missing samples, computes horizontal/vertical gaze velocity with a Savitzky–Golay filter (window 7, order 2), clips to ±1000 °/s and cuts non-overlapping 5 s windows (5000 samples).
2. **Subject-disjoint split** (`split.py`): subjects present in Round 6 form the held-out test set; the remaining subjects are divided into 4 folds, one of which is used for validation.
3. **Model** (`model.py`): a 1-D DenseNet-style CNN with dilated convolutions (dilations 1–64, growth rate 32) and a 128-D embedding head, plus an auxiliary classification head.
4. **Training** (`train.py`): Multi-Similarity loss + 0.1 × cross-entropy, batches of 16 identities × 4 windows (`MPerClassSampler`), Adam with a one-cycle learning-rate schedule, 100 epochs. Validation metric: MAP@R.
5. **Evaluation** (`evaluate.py`): enrollment = first 5 s of the Round 1 Session 1 reading task (TEX), probe = first 5 s of the Round 1 Session 2 reading task, cosine similarity between L2-normalised embeddings. Reports EER, Rank-1 identification and FNMR at 1% and 0.1% FMR. The final-epoch checkpoint is used, so the test set is never used for model selection.

## Results

<!-- TODO: replace with the output of `python summarize.py` after running all four folds -->

| Fold | EER (%) | Rank-1 (%) | FNMR @ 1% FMR (%) | FNMR @ 0.1% FMR (%) |
|---|---|---|---|---|
| 0 | 5.02 | 91.38 | 10.34 | 34.48 |

58 held-out test subjects, one 5 s enrollment and one 5 s probe per subject (58 genuine and ~3,300 impostor comparisons).
With only 58 genuine comparisons, one misclassified subject moves the EER by about 1.7 percentage points, so single-fold numbers are noisy; the four-fold mean ± SD is the number to cite.

<!-- TODO: add plots/fold0/score_distributions.png and plots/fold0/val_map_at_r.png -->

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

1. Download GazeBase (link below) and unpack it into `data_raw/` so that the `Round_N` folders sit directly inside it (`data_raw/Round_1/S1/...`, the layout `quality.py` expects).
2. Preprocess once:
   ```bash
   python preprocess.py
   ```
3. Train and evaluate all four folds and print the summary table:
   ```bash
   bash run_all_folds.sh
   ```
   Or a single fold:
   ```bash
   python split.py --fold 0 && python normalize.py
   python train.py --fold 0
   python evaluate.py --fold 0
   python plots.py --fold 0
   ```

Training uses CUDA or Apple MPS automatically when available.

## Data

GazeBase: H. Griffith, D. Lohr, E. Abdulin and O. Komogortsev, *"GazeBase, a large-scale, multi-stimulus, longitudinal eye movement dataset,"* Scientific Data, 2021. Available on Figshare. The data are not included in this repository.

## Limitations

- Results so far are from one fold; four-fold results are pending.
- Enrollment and probe are a single 5 s window each; the original paper also reports longer durations, which perform much better.
- The architecture follows the paper's description but has not been checked layer-by-layer against the authors' code, so the numbers are not directly comparable to the published ones.
