import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Constants for MELD
LABEL_DICT = {
    'neutral':  0,
    'sadness':  1,
    'joy':      2,
    'anger':    3,
    'disgust':  4,
    'fear':     5,
    'surprise': 6,
}

def load_meld_raw(args):
    """Loads the raw CSV and filters for existing audio files."""
    label_path = Path(args.raw_data_dir) / "MELD.Raw" / "train_sent_emo.csv"
    data_path = Path(args.raw_data_dir) / "MELD.Raw" / "train_splits" / "waves"
    
    if not label_path.exists():
        raise FileNotFoundError(f"Could not find MELD labels at {label_path}")

    df = pd.read_csv(label_path)
    
    # Map labels and drop invalid ones
    df['Label'] = df['Emotion'].map(LABEL_DICT)
    df = df[df['Label'].notna()].copy()
    df['Label'] = df['Label'].astype(int)

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing MELD utterances"):
        fname = f"dia{row.Dialogue_ID}_utt{row.Utterance_ID}"
        wav_path = data_path / f"{fname}.wav"
        
        # In MELD, we ensure the file exists as the dataset is often fragmented
        if wav_path.exists():
            rows.append({
                "Dialogue_ID": row.Dialogue_ID,
                "Utterance_ID": row.Utterance_ID,
                "filename": fname,
                "path": str(wav_path),
                "label": int(row.Label),
                "text": row.Utterance
            })
            
    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser(description="Repartition MELD to mirror IEMOCAP logic.")
    parser.add_argument("--raw_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--scene_size", type=int, default=15)
    parser.add_argument("--num_clients", type=int, default=10)
    args = parser.parse_args()

    df = load_meld_raw(args)

    # 1. Group by Dialogue_ID to ensure boundary-respecting construction
    all_scenes = []
    dialogues = sorted(df['Dialogue_ID'].unique())

    for dia_id in dialogues:
        dia_df = df[df['Dialogue_ID'] == dia_id].sort_values('Utterance_ID')
        
        # 2. Chunk within the dialogue (no scene crosses a conversation boundary)
        dia_rows = dia_df.to_dict('records')
        for i in range(0, len(dia_rows), args.scene_size):
            chunk = dia_rows[i : i + args.scene_size]
            
            # Format: [Filename, Path, Label, Utterance, None]
            scene = [[r['filename'], r['path'], r['label'], r['text'], None] for r in chunk]
            all_scenes.append(scene)

    # 3. Sequential 80/10/10 Split
    n_scenes = len(all_scenes)
    n_dev = int(n_scenes * 0.1)
    n_test = int(n_scenes * 0.1)
    n_train = n_scenes - n_dev - n_test

    train_scenes = all_scenes[:n_train]
    dev_scenes = all_scenes[n_train : n_train + n_dev]
    test_scenes = all_scenes[n_train + n_dev :]

    # 4. Sequential (Round-Robin) Partitioning for FL clients
    partition = {str(i): [] for i in range(args.num_clients)}
    for i, scene in enumerate(train_scenes):
        client_id = str(i % args.num_clients)
        partition[client_id].append(scene)

    partition['dev'] = dev_scenes
    partition['test'] = test_scenes

    # Save
    out_dir = Path(args.output_dir) / "partition" / "meld"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "partition.json", "w") as f:
        json.dump(partition, f, indent=2)

    print(f"Total Scenes: {n_scenes}")
    print(f"Train: {len(train_scenes)}, Dev: {len(dev_scenes)}, Test: {len(test_scenes)}")
    print(f"Partitioned into {args.num_clients} clients (Sequential Round-Robin).")

if __name__ == "__main__":
    main()