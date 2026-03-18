"""
scene_dataloader.py
===================
Scene-level dataset and dataloader for reintegration experiments.

Yielded batch per scene (batch_size always 1):
    scene_x_a:    list[T] of (1, T_frames, D_audio)
    scene_x_b:    list[T] of (1, T_tokens, D_text)
    scene_len_a:  list[T] of (1,)
    scene_len_b:  list[T] of (1,)
    scene_labels: (T,)  LongTensor — per-utterance emotion labels
    scene_mask:   (T,)  LongTensor — per-utterance audio availability {0,1}
"""

import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# Markov mask generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_markov_mask(
    scene_len: int,
    p_stay_absent: float,
    p_stay_present: float,
    seed: int,
) -> np.ndarray:
    """
    Generate a deterministic per-utterance audio availability mask.
    Seed derived from scene_id ensures the same mask every run for the same scene.

    State:  1 = audio present,  0 = audio absent
    P(present → present) = p_stay_present
    P(absent  → absent)  = p_stay_absent
    P(absent  → present) = 1 - p_stay_absent  ← reintegration event
    """
    rng = np.random.default_rng(seed)
    mask = np.ones(scene_len, dtype=np.int64)
    state = 1 if rng.random() < p_stay_present else 0
    mask[0] = state
    for t in range(1, scene_len):
        if state == 1:
            state = 1 if rng.random() < p_stay_present else 0
        else:
            state = 0 if rng.random() < p_stay_absent else 1
        mask[t] = state
    return mask


def scene_id_to_seed(scene_id: str) -> int:
    """Stable integer seed from scene id string, e.g. 'dia64' → int."""
    return abs(hash(scene_id)) % (2**31)


# ─────────────────────────────────────────────────────────────────────────────
# Scene-level Dataset
# ─────────────────────────────────────────────────────────────────────────────

class SceneDataset(Dataset):
    """
    Args:
        scenes:          list of scenes from partition.json
                         scene = list of utterances
                         utt   = [filename, path, label, text, None]
        audio_feat_dict: filename-keyed dict  {'dia64_utt3': np.ndarray}
        text_feat_dict:  filename-keyed dict  {'dia64_utt3': np.ndarray}
        default_shape_a: fallback shape (T_frames, D_audio) when filename missing
        default_shape_b: fallback shape (T_tokens, D_text)  when filename missing
        p_stay_absent:   P(absent→absent) Markov parameter
        p_stay_present:  P(present→present) Markov parameter
        apply_mask:      True  → generate Markov mask per scene
                         False → all-ones mask (stable condition)
    """

    def __init__(
        self,
        scenes: list,
        audio_feat_dict: dict,
        text_feat_dict: dict,
        default_shape_a: np.ndarray,
        default_shape_b: np.ndarray,
        p_stay_absent: float = 0.7,
        p_stay_present: float = 0.75,
        apply_mask: bool = True,
    ):
        self.scenes          = scenes
        self.audio_feat_dict = audio_feat_dict
        self.text_feat_dict  = text_feat_dict
        self.default_shape_a = tuple(default_shape_a)
        self.default_shape_b = tuple(default_shape_b)
        self.p_stay_absent   = p_stay_absent
        self.p_stay_present  = p_stay_present
        self.apply_mask      = apply_mask

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene = self.scenes[idx]
        T = len(scene)

        # Scene ID from first utterance filename: 'dia64_utt0' → 'dia64'
        scene_id  = scene[0][0].rsplit('_utt', 1)[0]
        mask_seed = scene_id_to_seed(scene_id)

        if self.apply_mask:
            mask_np = generate_markov_mask(
                T, self.p_stay_absent, self.p_stay_present, seed=mask_seed
            )
        else:
            mask_np = np.ones(T, dtype=np.int64)

        scene_mask   = torch.tensor(mask_np, dtype=torch.long)  # (T,)
        scene_x_a    = []
        scene_x_b    = []
        scene_len_a  = []
        scene_len_b  = []
        scene_labels = []

        for utt in scene:
            filename = utt[0]
            label    = utt[2]


            # Audio
            feat_a = self.audio_feat_dict.get(filename, None)
            if feat_a is not None:
                if feat_a.ndim == 3:
                    feat_a = feat_a[0]
                # Treat utterances too short to survive conv stack as missing
                if feat_a.shape[0] < 8:
                    feat_a = None

            if feat_a is not None:
                x_a = torch.tensor(feat_a, dtype=torch.float32)
                l_a = x_a.shape[0]
            else:
                x_a = torch.zeros(self.default_shape_a, dtype=torch.float32)
                l_a = 0

            # feat_a = self.audio_feat_dict.get(filename, None)
            # if feat_a is not None:
            #     if feat_a.ndim == 3:
            #         feat_a = feat_a[0]
            #     x_a = torch.tensor(feat_a, dtype=torch.float32)
            #     l_a = x_a.shape[0]
            # else:
            #     x_a = torch.zeros(self.default_shape_a, dtype=torch.float32)
            #     l_a = 0

            # Text
            feat_b = self.text_feat_dict.get(filename, None)
            if feat_b is not None:
                if feat_b.ndim == 3:
                    feat_b = feat_b[0]
                x_b = torch.tensor(feat_b, dtype=torch.float32)
                l_b = x_b.shape[0]
            else:
                x_b = torch.zeros(self.default_shape_b, dtype=torch.float32)
                l_b = 0

            scene_x_a.append(x_a.unsqueeze(0))      # (1, T_frames, D_audio)
            scene_x_b.append(x_b.unsqueeze(0))      # (1, T_tokens, D_text)
            scene_len_a.append(torch.tensor([l_a]))  # (1,)
            scene_len_b.append(torch.tensor([l_b]))  # (1,)
            scene_labels.append(label)

        scene_labels = torch.tensor(scene_labels, dtype=torch.long)  # (T,)

        return scene_x_a, scene_x_b, scene_len_a, scene_len_b, scene_labels, scene_mask


# ─────────────────────────────────────────────────────────────────────────────
# Collate — batch_size is always 1; just unwrap the outer list
# ─────────────────────────────────────────────────────────────────────────────

def collate_scene_fn(batch):
    assert len(batch) == 1, (
        "SceneDataset requires batch_size=1 — scenes have variable T and cannot be stacked."
    )
    return batch[0]


# ─────────────────────────────────────────────────────────────────────────────
# Factory — called by DataloadManager.set_scene_dataloader()
# ─────────────────────────────────────────────────────────────────────────────

def build_scene_dataloader(
    scenes: list,
    audio_feat_dict: dict,
    text_feat_dict: dict,
    default_shape_a: np.ndarray,
    default_shape_b: np.ndarray,
    p_stay_absent: float = 0.7,
    p_stay_present: float = 0.75,
    apply_mask: bool = True,
    shuffle: bool = False,
) -> DataLoader:
    """
    Build a scene-level DataLoader.

    Args:
        scenes:          list of scenes from partition.json
        audio_feat_dict: filename-keyed dict from extract_audio_features_scene.py
        text_feat_dict:  filename-keyed dict from extract_text_features_scene.py
        default_shape_a: fallback audio shape (T_frames, D_audio)
        default_shape_b: fallback text shape  (T_tokens, D_text)
        p_stay_absent:   P(absent→absent)
        p_stay_present:  P(present→present)
        apply_mask:      True applies Markov mask; False gives all-ones
        shuffle:         True for training clients, False for dev/test
    """
    dataset = SceneDataset(
        scenes          = scenes,
        audio_feat_dict = audio_feat_dict,
        text_feat_dict  = text_feat_dict,
        default_shape_a = default_shape_a,
        default_shape_b = default_shape_b,
        p_stay_absent   = p_stay_absent,
        p_stay_present  = p_stay_present,
        apply_mask      = apply_mask,
    )

    return DataLoader(
        dataset,
        batch_size  = 1,
        shuffle     = shuffle,
        num_workers = 0,
        collate_fn  = collate_scene_fn,
    )