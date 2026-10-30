import os
import pandas as pd
import torch
from pathlib import Path
import numpy as np
from scipy.signal import savgol_filter

def process_and_save_gazebase_file(csv_path, save_dir, user_id, session_id, round_id, task_id):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return


    if 'x' in df.columns and 'y' in df.columns:
        cols = ['x', 'y']
    else:
        print(f"Couldn't find coordinate columns {csv_path}")
        return

    positions = (
    df[cols]
    .interpolate(limit_direction="both")
    .to_numpy(dtype=np.float32)
    )

    vel_array = savgol_filter(
        positions,
        window_length=7,
        polyorder=2,
        deriv=1,
        delta=0.001,
        axis=0,
        mode="interp",
    )

    vel_array = np.nan_to_num(
        vel_array,
        nan=0.0,
        posinf=1000.0,
        neginf=-1000.0,
    )

    vel_array = np.clip(vel_array, -1000.0, 1000.0)
    

    window_size = 5000
    num_windows = len(vel_array) // window_size
    
    for i in range(num_windows):
        start_idx = i * window_size
        end_idx = start_idx + window_size
        
        window = vel_array[start_idx:end_idx]
        window_transposed = window.T 
        tensor_window = torch.tensor(window_transposed, dtype=torch.float32)
        
        save_name = f"user_{user_id}_{round_id}_sess_{session_id}_task_{task_id}_win_{i}.pt"
        torch.save(tensor_window, os.path.join(save_dir, save_name))

def main():
    raw_dir = Path("data_raw")
    processed_dir = Path("data_processed")
    
    processed_dir.mkdir(exist_ok=True)
    
    csv_files = list(raw_dir.rglob("*.csv"))
    print(f"Found CSV files: {len(csv_files)}")
    
    for csv_path in csv_files:

        round_id = "UnknownRound"
        for part in csv_path.parts:
            if "Round" in part:
                round_id = part
                break

        filename = csv_path.name
        parts = filename.split('_') 
        
        if len(parts) >= 4:
            user_id = parts[1]
            session_id = parts[2]
            task_id = parts[3].replace('.csv', '')
            
            process_and_save_gazebase_file(csv_path, processed_dir, user_id, session_id, round_id, task_id)
        else:        
            print(f"There is a file with uncommon name {filename}")
            
    print("Preprocessing is done")

if __name__ == "__main__":
    main()