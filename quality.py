from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


RAW_DIR = Path("data_raw")
WINDOW_SIZE = 5000


def get_user_id(file_path):
    raw_user = file_path.stem.split("_")[1]
    return f"{int(raw_user) % 1000:03d}"


def get_nan_fraction(file_path):
    df = pd.read_csv(file_path)

    positions = df[["x", "y"]].to_numpy(dtype=np.float32)

    velocity = savgol_filter(
        positions,
        window_length=7,
        polyorder=2,
        deriv=1,
        delta=0.001,
        axis=0,
        mode="nearest",
    )

    first_window = velocity[:WINDOW_SIZE]

    return np.isnan(first_window).mean()


def get_valid_users(session, test_users):
    folder = RAW_DIR / "Round_1" / session

    valid_users = set()

    for file_path in folder.rglob("*_TEX.csv"):
        user_id = get_user_id(file_path)

        if user_id not in test_users:
            continue

        if get_nan_fraction(file_path) <= 0.5:
            valid_users.add(user_id)

    return valid_users