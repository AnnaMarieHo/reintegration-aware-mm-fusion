"""
extract_text_features_scene.py
================================
MobileBERT feature extraction for the MELD reintegration experiments.

Reads the scene-nested partition.json produced by data_partition.py:
    partition[client_id][scene_idx][utt_idx] = [Filename, Path, Label, Utterance, None]

Saves one filename-keyed dict per client/split:
    output_dir/feature/text/mobilebert/meld/{client_id}.pkl
    = { 'dia64_utt3': np.ndarray(T_tokens, 512), ... }

This format is consumed directly by SceneDataset in scene_dataloader.py.

No normalisation is applied — BERT-family embeddings are already well-scaled.

Usage:
    python extract_text_features_scene.py \
        --output_dir /path/to/output \
        --dataset meld
"""

import os
import json
import pickle
import logging
import argparse

from tqdm import tqdm
from pathlib import Path

from .feature_manager import FeatureManager

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
        path_conf['data_dir'] = str(Path(os.path.realpath(__file__)).parents[1])
    if path_conf['output_dir'] == '.':
        path_conf['output_dir'] = str(Path(os.path.realpath(__file__)).parents[1].joinpath('output'))

    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_dir', default=path_conf['data_dir'], type=str)
    parser.add_argument('--output_dir',   default=path_conf['output_dir'], type=str)
    parser.add_argument('--feature_type', default='mobilebert', type=str)
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
        'feature', 'text', args.feature_type, args.dataset
    )
    output_path.mkdir(parents=True, exist_ok=True)

    partition_path = Path(args.output_dir).joinpath(
        'partition', args.dataset, 'partition.json'
    )
    with open(str(partition_path)) as f:
        partition = json.load(f)

    logging.info(f'Clients/splits found: {list(partition.keys())}')
    fm = FeatureManager(args)   # loads MobileBERT once here

    for client_id, scenes in partition.items():
        out_file = output_path.joinpath(f'{client_id}.pkl')
        if out_file.exists():
            logging.info(f'  {client_id}: already exists, skipping.')
            continue

        n_utts = sum(len(s) for s in scenes)
        logging.info(f'  {client_id}: {len(scenes)} scenes, {n_utts} utterances')

        feat_dict = {}

        for utt in tqdm(iter_utterances(scenes), total=n_utts,
                        desc=f'{client_id} extract'):
            filename = utt[0]   # e.g. 'dia64_utt3'
            text_str = utt[3]   # utterance text (index 3, NOT speaker)

            if not isinstance(text_str, str) or not text_str.strip():
                logging.warning(f'    Empty text for {filename}, skipping.')
                continue

            feats = fm.extract_text_feature(input_str=text_str)
            # feats: np.ndarray(T_tokens, 512) from MobileBERT last_hidden_state

            feat_dict[filename] = feats

        with open(str(out_file), 'wb') as handle:
            pickle.dump(feat_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info(f'  {client_id}: saved {len(feat_dict)} utterances → {out_file}')

    logging.info('Text extraction complete.')