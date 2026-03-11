"""
extract_audio_features_scene.py
================================
MFCC feature extraction for the MELD reintegration experiments.

Reads the scene-nested partition.json produced by data_partition.py:
    partition[client_id][scene_idx][utt_idx] = [Filename, Path, Label, Utterance, None]

Saves one filename-keyed dict per client/split:
    output_dir/feature/audio/mfcc/meld/{client_id}.pkl
    = { 'dia64_utt3': np.ndarray(T_frames, 80), ... }

This format is consumed directly by SceneDataset in scene_dataloader.py.

Speaker normalisation:
    Original MELD extractor normalises per-speaker, but the Speaker column is
    not stored in the partition rows. We use per-client global normalisation for
    training clients (all utterances assigned to that client treated as one group)
    and per-split global normalisation for dev/test. This removes DC offset and
    is consistent across the entire pipeline.

Usage:
    python extract_audio_features_scene.py \
        --output_dir /path/to/output \
        --dataset meld
"""

import os
import json
import pickle
import logging
import argparse
import warnings
import numpy as np

from tqdm import tqdm
from pathlib import Path

from .feature_manager import FeatureManager

warnings.filterwarnings('ignore')
logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)


def parse_args():
    path_conf = {}
    cfg_path = Path(os.path.realpath(__file__)).parents[1].joinpath('system.cfg')
    with open(str(cfg_path)) as f:
        for line in f:
            key, val = line.strip().split('=')
            path_conf[key] = val.replace('"', '')
    if path_conf['data_dir'] == '.':
        # path_conf['data_dir'] = str(Path(os.path.realpath(__file__)).parents[3].joinpath('data'))
        path_conf['data_dir'] = str(Path(os.path.realpath(__file__)).parents[1])
    if path_conf['output_dir'] == '.':
        path_conf['output_dir'] = str(Path(os.path.realpath(__file__)).parents[1].joinpath('output'))

    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_dir', default=path_conf['data_dir'], type=str)
    parser.add_argument('--output_dir',   default=path_conf['output_dir'], type=str)
    parser.add_argument('--feature_type', default='mfcc', type=str)
    parser.add_argument('--dataset',      default='meld', type=str)
    return parser.parse_args()


def iter_utterances(scenes):
    """Yield every utterance from a nested list of scenes.
    utt layout: [Filename, Path, Label, Utterance_text, None]
    """
    for scene in scenes:
        for utt in scene:
            yield utt


if __name__ == '__main__':
    args = parse_args()

    output_path = Path(args.output_dir).joinpath(
        'feature', 'audio', args.feature_type, args.dataset
    )
    output_path.mkdir(parents=True, exist_ok=True)

    partition_path = Path(args.output_dir).joinpath(
        'partition', args.dataset, 'partition.json'
    )
    with open(str(partition_path)) as f:
        partition = json.load(f)

    logging.info(f'Clients/splits found: {list(partition.keys())}')
    fm = FeatureManager(args)

    for client_id, scenes in partition.items():
        out_file = output_path.joinpath(f'{client_id}.pkl')
        if out_file.exists():
            logging.info(f'  {client_id}: already exists, skipping.')
            continue

        n_scenes = len(scenes)
        n_utts   = sum(len(s) for s in scenes)
        logging.info(f'  {client_id}: {n_scenes} scenes, {n_utts} utterances')

        # Pass 1: extract raw MFCC features
        raw        = {}   # filename -> np.ndarray(T_frames, 80)
        all_frames = []

        for utt in tqdm(iter_utterances(scenes), total=n_utts,
                        desc=f'{client_id} extract'):
            filename  = utt[0]   # e.g. 'dia64_utt3'
            file_path = utt[1]   # wav path

            feats = fm.extract_mfcc_features(
                audio_path   = file_path,
                frame_length = 25,
                frame_shift  = 10,
                max_len      = 1000,
                en_znorm     = False,
            )
            if feats is None or feats.shape[0] == 0:
                logging.warning(f'    Empty features for {filename}, using zeros.')
                feats = np.zeros((1, 80), dtype=np.float32)

            raw[filename] = feats
            all_frames.append(feats)

        # Pass 2: global normalisation across all utterances for this client/split
        all_concat  = np.concatenate(all_frames, axis=0)
        global_mean = np.mean(all_concat, axis=0)
        global_std  = np.std(all_concat,  axis=0)

        feat_dict = {
            fname: (f - global_mean) / (global_std + 1e-5)
            for fname, f in raw.items()
        }

        with open(str(out_file), 'wb') as handle:
            pickle.dump(feat_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info(f'  {client_id}: saved {len(feat_dict)} utterances → {out_file}')

    logging.info('Audio extraction complete.')