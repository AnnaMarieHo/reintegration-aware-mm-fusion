import os
import json
import argparse
import numpy as np
import pandas as pd
import pickle, pdb

from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

from .partition_manager import PartitionManager


# Label dictionary
LABEL_DICT = {
    'neutral':  0,
    'sadness':  1,
    'joy':      2,
    'anger':    3,
    'disgust':  4,
    'fear':     5,
    'surprise': 6,
}

# Utterance entries that look like generic/non-character speakers
SPEAKER_FILTER = {"All", "Man", "Policeman", "Tag", "Woman"}


def load_split(args, split: str, pm: PartitionManager) -> list:
    """

    python -m my_extensions.reintegration.features.data_partition   --raw_data_dir /mnt/c/Users/aymie/Documents/UK_projects/masters-proj/my_extensions/reintegration   --output_partition_path /mnt/c/Users/aymie/Documents/UK_projects/masters-proj/my_extensions/reintegration/output


    Load one MELD split (train / dev / test) and return a list of scenes.

    Each scene is a list of utterances ordered by Utterance_ID:
        scene = [
            [filename, path, label, utterance_text],   # utt 0
            [filename, path, label, utterance_text],   # utt 1
            ...
        ]

    Scenes with fewer than min_scene_len valid utterances are dropped.
    Utterances whose emotion label is not in LABEL_DICT are dropped; scenes
    that become too short after filtering are also dropped.

    Args:
        args:  parsed argument namespace (provides raw_data_dir, min_scene_len)
        split: "train" | "dev" | "test"
        pm:    PartitionManager instance (provides label_dict)

    Returns:
        List of scenes (each scene is a list of utterance rows).
    """
    if split == 'train':
        label_path = f'{args.raw_data_dir}/MELD.Raw/train_sent_emo.csv'
        data_path  = f'{args.raw_data_dir}/MELD.Raw/train_splits'
    elif split == 'dev':
        label_path = f'{args.raw_data_dir}/MELD.Raw/dev_sent_emo.csv'
        data_path  = f'{args.raw_data_dir}/MELD.Raw/dev_splits_complete'
    elif split == 'test':
        label_path = f'{args.raw_data_dir}/MELD.Raw/test_sent_emo.csv'
        data_path  = f'{args.raw_data_dir}/MELD.Raw/output_repeated_splits_test'
    else:
        raise ValueError(f'Unknown split: {split}')

    df = pd.read_csv(label_path)

    # 1. Drop missing / corrupt audio files
    missing_indices = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc=f'Checking {split} files'):
        wav = Path(f"{data_path}/waves/dia{row.Dialogue_ID}_utt{row.Utterance_ID}.wav")
        if not wav.is_file():
            missing_indices.append(i)
    if missing_indices:
        print(f'[{split}] Dropping {len(missing_indices)} missing/corrupt files.')
    df = df.drop(missing_indices)

    # 2. Map emotion labels; drop unknown emotions
    df['Label'] = df['Emotion'].map(pm.label_dict)
    df = df[df['Label'].notna()].copy()
    df['Label'] = df['Label'].astype(int)

    # 3. Drop known non-character speaker entries
    # df = df[~df['Speaker'].isin(SPEAKER_FILTER)].copy()

    # 4. Build file path and filename columns
    df['Path'] = df.apply(
        lambda r: f"{data_path}/waves/dia{r.Dialogue_ID}_utt{r.Utterance_ID}.wav",
        axis=1
    )
    df['Filename'] = df.apply(
        lambda r: f"dia{r.Dialogue_ID}_utt{r.Utterance_ID}",
        axis=1
    )

    # 5. Group by Dialogue_ID, sort by Utterance_ID → scenes
    #    Each utterance row: [Filename, Path, Label, Utterance_text]
    #    Feature extraction will later fill in a 5th slot for the feature array;
    #    we append None as a placeholder to match the expected list structure.
    min_len = getattr(args, 'min_scene_len', 2)
    scenes = []
    for dialogue_id, group in df.groupby('Dialogue_ID'):
        group_sorted = group.sort_values('Utterance_ID')
        utt_ids = group_sorted['Utterance_ID'].tolist()

        expected = list(range(utt_ids[0], utt_ids[-1] + 1))
        if utt_ids != expected:
            print(f'[{split}] Dialogue {dialogue_id} has missing Utterance_IDs: {expected} - {utt_ids}')
            continue

        utterances = [
            [row['Filename'], row['Path'], row['Label'], row['Utterance'], None]
            for _, row in group_sorted.iterrows()
        ]
        if len(utterances) >= min_len:
            scenes.append(utterances)

    print(f'[{split}] {len(scenes)} scenes after filtering '
          f'(min_scene_len={min_len}).')
    return scenes


def dirichlet_partition_scenes(
    scenes: list,
    num_clients: int,
    alpha: float,
    seed: int = 42,
    min_scene_size: int = 1,
) -> dict:
    """
    Partition scenes across num_clients using a Dirichlet distribution over
    the dominant emotion label of each scene.

    The dominant label is the most common emotion label among utterances in the
    scene (ties broken by first occurrence).

    Args:
        scenes:          list of scenes from load_split (train only).
        num_clients:     number of FL clients.
        alpha:           Dirichlet concentration parameter.
                         Lower => more skewed; higher => more uniform.
        seed:            RNG seed for reproducibility.
        min_scene_size:  retry threshold — minimum scenes per client.

    Returns:
        dict mapping client_id (int) -> list of scenes assigned to that client.
    """
    # Compute dominant label per scene
    def dominant_label(scene):
        counts = defaultdict(int)
        for utt in scene:
            counts[utt[2]] += 1           # utt[2] is the integer label
        return max(counts, key=counts.get)

    scene_labels = [dominant_label(s) for s in scenes]
    unique_labels = sorted(set(scene_labels))
    K = len(unique_labels)
    N = len(scenes)

    np.random.seed(seed)
    min_size = 0

    while min_size < min_scene_size:
        client_indices = [[] for _ in range(num_clients)]

        for label in unique_labels:
            # indices of scenes with this dominant label
            idx_k = np.where(np.array(scene_labels) == label)[0]
            np.random.shuffle(idx_k)

            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            # Balance: don't over-fill clients that already have many scenes
            proportions = np.array([
                p * (len(c) < N / num_clients)
                for p, c in zip(proportions, client_indices)
            ])
            # Renormalise (may be all-zero if all clients are full — rare)
            prop_sum = proportions.sum()
            if prop_sum == 0:
                proportions = np.ones(num_clients) / num_clients
            else:
                proportions = proportions / prop_sum

            splits = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            for client_id, chunk in enumerate(np.split(idx_k, splits)):
                client_indices[client_id].extend(chunk.tolist())

        min_size = min(len(c) for c in client_indices)

    # Build output dict: client_id -> list of scenes
    client_scenes = {}
    for client_id, indices in enumerate(client_indices):
        client_scenes[client_id] = [scenes[i] for i in indices]

    return client_scenes

from collections import Counter

def label_dist(scenes):
    counts = Counter()
    for scene in scenes:
        for utt in scene:
            counts[utt[2]] += 1
    return dict(sorted(counts.items()))

if __name__ == '__main__':

    path_conf = {}
    cfg_path = Path(os.path.realpath(__file__)).parents[1].joinpath('system.cfg')
    with open(str(cfg_path)) as f:
        for line in f:
            key, val = line.strip().split('=')
            path_conf[key] = val.replace('"', '')

    if path_conf['data_dir'] == '.':
        path_conf['data_dir'] = str(
            Path(os.path.realpath(__file__)).parents[3].joinpath('data')
        )
    if path_conf['output_dir'] == '.':
        path_conf['output_dir'] = str(
            Path(os.path.realpath(__file__)).parents[3].joinpath('output')
        )

    parser = argparse.ArgumentParser(
        description='Scene-level MELD data partition for Level-1 reintegration experiments.'
    )
    parser.add_argument(
        '--raw_data_dir',
        type=str,
        default=path_conf['data_dir'],
        help='Root directory containing meld/MELD.Raw/...',
    )
    parser.add_argument(
        '--output_partition_path',
        type=str,
        default=path_conf['output_dir'],
        help='Root directory for writing partition/meld/partition.json',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='meld',
        help='Dataset name (only meld supported here).',
    )
    parser.add_argument(
        '--num_clients',
        type=int,
        default=10,
        help='Number of FL training clients.',
    )
    parser.add_argument(
        '--alpha',
        type=float,
        # default=0.5,
        default=100,
        help='Dirichlet concentration parameter controlling label heterogeneity.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='RNG seed for Dirichlet partitioning.',
    )
    parser.add_argument(
        '--min_scene_len',
        type=int,
        default=9,
        help='Minimum number of valid utterances required to keep a scene.',
    )
    args = parser.parse_args()

    output_data_path = Path(args.output_partition_path).joinpath(
        'partition', args.dataset
    )
    output_data_path.mkdir(parents=True, exist_ok=True)

    pm = PartitionManager(args)
    pm.fetch_label_dict()

    train_scenes = load_split(args, split='train', pm=pm)
    dev_scenes   = load_split(args, split='dev',   pm=pm)
    test_scenes  = load_split(args, split='test',  pm=pm)

    # Partition training scenes across clients (Dirichlet)
    client_scenes = dirichlet_partition_scenes(
        scenes=train_scenes,
        num_clients=args.num_clients,
        alpha=args.alpha,
        seed=args.seed,
        min_scene_size=1,
    )

    # Assemble partition_dict
    # Training clients: int keys 0..num_clients-1
    # Dev / test:       string keys 'dev' and 'test'
    #                   preserved as lists of scenes (not distributed to clients)
    partition_dict = {}

    for client_id, scenes in client_scenes.items():
        partition_dict[client_id] = scenes

    # Dev and test keep all scenes; eval runs over every scene in order
    partition_dict['dev']  = dev_scenes
    partition_dict['test'] = test_scenes

    total_train_scenes = sum(len(v) for k, v in partition_dict.items()
                             if k not in ('dev', 'test'))
    print(f'\nPartition summary:')
    print(f'  Training clients : {args.num_clients}')
    print(f'  Total train scenes distributed: {total_train_scenes} '
          f'(original: {len(train_scenes)})')
    print(f'  Dev scenes  : {len(partition_dict["dev"])}')
    print(f'  Test scenes : {len(partition_dict["test"])}')

    scenes_per_client = [len(partition_dict[i]) for i in range(args.num_clients)]
    print(f'  Scenes per client — '
          f'min: {min(scenes_per_client)}, '
          f'max: {max(scenes_per_client)}, '
          f'mean: {np.mean(scenes_per_client):.1f}')


    print("Train:", label_dist(train_scenes))
    print("Dev:",   label_dist(dev_scenes))
    print("Test:",  label_dist(test_scenes))

    # Verify utterance ordering within a sample scene
    sample_scene = partition_dict[0][0]
    print(f'\n  Sample scene (client 0, scene 0):')
    for utt in sample_scene:
        print(f'    {utt[0]}  label={utt[2]}  text="{utt[3][:40]}"')

    # Write partition
    out_path = output_data_path.joinpath('partition.json')
    with open(out_path, 'w') as f:
        json.dump(partition_dict, f, indent=2)
    print(f'\nPartition written to: {out_path}')