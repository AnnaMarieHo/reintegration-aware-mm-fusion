"""
Unit tests for DataloadManager.set_scene_dataloader.

Documents the contract between DataloadManager → build_scene_dataloader →
SceneDataset and what the training code receives from each DataLoader step.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from reintegration.dataloader.dataload_manager import DataloadManager


def _dm_for_scene_tests() -> DataloadManager:
    """DataloadManager whose __init__ does not touch feature paths."""
    args = SimpleNamespace(dataset="__scene_test__", data_dir="/unused")
    return DataloadManager(args)


def _scene_row(filename: str, label: int = 0):
    """One partition utterance row: [Filename, Path, Label, Utterance, None]."""
    return [filename, "/dummy/path", label, "dummy text", None]


class TestSetSceneDataloader(unittest.TestCase):
    """Batch layout matches scene_dataloader.SceneDataset + collate_scene_fn."""

    def test_returns_dataloader_and_batch_has_six_fields(self):
        scenes = [
            [
                _scene_row("dia1_utt0", label=2),
                _scene_row("dia1_utt1", label=3),
            ]
        ]
        Ta, Da, Tb, Db = 12, 80, 5, 512
        audio_feat_dict = {
            "dia1_utt0": np.random.randn(Ta, Da).astype(np.float32),
            "dia1_utt1": np.random.randn(Ta, Da).astype(np.float32),
        }
        text_feat_dict = {
            "dia1_utt0": np.random.randn(Tb, Db).astype(np.float32),
            "dia1_utt1": np.random.randn(Tb, Db).astype(np.float32),
        }
        dm = _dm_for_scene_tests()
        loader = dm.set_scene_dataloader(
            scenes=scenes,
            audio_feat_dict=audio_feat_dict,
            text_feat_dict=text_feat_dict,
            default_feat_shape_a=np.array([1000, 80]),
            default_feat_shape_b=np.array([10, 512]),
            shuffle=False,
            apply_mask=False,
        )
        self.assertIsInstance(loader, DataLoader)
        self.assertEqual(loader.batch_size, 1)

        batch = next(iter(loader))
        self.assertEqual(len(batch), 6)
        scene_x_a, scene_x_b, scene_len_a, scene_len_b, scene_labels, scene_mask = batch

        T = 2
        self.assertEqual(len(scene_x_a), T)
        self.assertEqual(len(scene_x_b), T)
        self.assertEqual(len(scene_len_a), T)
        self.assertEqual(len(scene_len_b), T)
        self.assertEqual(scene_labels.shape, (T,))
        self.assertEqual(scene_mask.shape, (T,))

        for t in range(T):
            self.assertEqual(scene_x_a[t].shape, (1, Ta, Da))
            self.assertEqual(scene_x_b[t].shape, (1, Tb, Db))
            self.assertEqual(scene_len_a[t].shape, (1,))
            self.assertEqual(scene_len_b[t].shape, (1,))
            self.assertEqual(int(scene_len_a[t].item()), Ta)
            self.assertEqual(int(scene_len_b[t].item()), Tb)

        self.assertEqual(scene_labels.dtype, torch.long)
        self.assertTrue(torch.all(scene_mask == 1))

    def test_missing_features_use_default_shapes_and_zero_lengths(self):
        scenes = [[_scene_row("missing_utt0", label=1)]]
        dm = _dm_for_scene_tests()
        default_a = np.array([7, 80])
        default_b = np.array([4, 512])
        loader = dm.set_scene_dataloader(
            scenes=scenes,
            audio_feat_dict={},
            text_feat_dict={},
            default_feat_shape_a=default_a,
            default_feat_shape_b=default_b,
            apply_mask=False,
        )
        scene_x_a, scene_x_b, scene_len_a, scene_len_b, scene_labels, scene_mask = next(
            iter(loader)
        )
        self.assertEqual(scene_x_a[0].shape, (1, default_a[0], default_a[1]))
        self.assertEqual(scene_x_b[0].shape, (1, default_b[0], default_b[1]))
        self.assertEqual(int(scene_len_a[0].item()), 0)
        self.assertEqual(int(scene_len_b[0].item()), 0)
        self.assertEqual(int(scene_labels[0].item()), 1)

    def test_audio_shorter_than_eight_frames_treated_as_missing(self):
        """SceneDataset drops audio with T_frames < 8 (conv stack constraint)."""
        scenes = [[_scene_row("short_utt0", label=0)]]
        audio_feat_dict = {"short_utt0": np.zeros((5, 80), dtype=np.float32)}
        text_feat_dict = {"short_utt0": np.ones((3, 512), dtype=np.float32)}
        dm = _dm_for_scene_tests()
        default_a = np.array([9, 80])
        loader = dm.set_scene_dataloader(
            scenes=scenes,
            audio_feat_dict=audio_feat_dict,
            text_feat_dict=text_feat_dict,
            default_feat_shape_a=default_a,
            default_feat_shape_b=np.array([10, 512]),
            apply_mask=False,
        )
        scene_x_a, scene_x_b, scene_len_a, scene_len_b, _, _ = next(iter(loader))
        self.assertEqual(int(scene_len_a[0].item()), 0)
        self.assertEqual(scene_x_a[0].shape, (1, default_a[0], default_a[1]))
        self.assertEqual(int(scene_len_b[0].item()), 3)

    def test_apply_mask_true_yields_binary_long_mask(self):
        scenes = [[_scene_row("dia99_utt0", label=0), _scene_row("dia99_utt1", label=1)]]
        Ta, Da, Tb, Db = 10, 80, 2, 512
        audio_feat_dict = {f"dia99_utt{i}": np.ones((Ta, Da), dtype=np.float32) for i in range(2)}
        text_feat_dict = {f"dia99_utt{i}": np.ones((Tb, Db), dtype=np.float32) for i in range(2)}
        dm = _dm_for_scene_tests()
        _, _, _, _, _, scene_mask = next(
            iter(
                dm.set_scene_dataloader(
                    scenes=scenes,
                    audio_feat_dict=audio_feat_dict,
                    text_feat_dict=text_feat_dict,
                    apply_mask=True,
                )
            )
        )
        self.assertEqual(scene_mask.dtype, torch.long)
        self.assertTrue(((scene_mask == 0) | (scene_mask == 1)).all())

    def test_apply_mask_false_all_ones(self):
        scenes = [[_scene_row("dia7_utt0", label=0)]]
        dm = _dm_for_scene_tests()
        *_, scene_mask = next(
            iter(
                dm.set_scene_dataloader(
                    scenes=scenes,
                    audio_feat_dict={"dia7_utt0": np.ones((10, 80), dtype=np.float32)},
                    text_feat_dict={"dia7_utt0": np.ones((2, 512), dtype=np.float32)},
                    apply_mask=False,
                )
            )
        )
        self.assertTrue(torch.all(scene_mask == 1))

    def test_ndim_three_features_squeeze_leading_dim(self):
        scenes = [[_scene_row("dia3d_utt0", label=0)]]
        Ta, Da = 11, 80
        audio_feat_dict = {"dia3d_utt0": np.ones((1, Ta, Da), dtype=np.float32)}
        text_feat_dict = {"dia3d_utt0": np.ones((1, 4, 512), dtype=np.float32)}
        dm = _dm_for_scene_tests()
        scene_x_a, scene_x_b, scene_len_a, scene_len_b, _, _ = next(
            iter(
                dm.set_scene_dataloader(
                    scenes=scenes,
                    audio_feat_dict=audio_feat_dict,
                    text_feat_dict=text_feat_dict,
                    apply_mask=False,
                )
            )
        )
        self.assertEqual(scene_x_a[0].shape, (1, Ta, Da))
        self.assertEqual(int(scene_len_a[0].item()), Ta)
        self.assertEqual(int(scene_len_b[0].item()), 4)


if __name__ == "__main__":
    unittest.main()
