import json
import glob
import torch
import pickle
import random
import pdb, os
import torchaudio
import numpy as np
import os.path as osp
# import pickle5 as pickle

from tqdm import tqdm
from pathlib import Path
from collections import Counter
from torch.utils.data import DataLoader, Dataset
from .scene_dataloader import build_scene_dataloader



def pad_tensor(vec, pad):
    pad_size = list(vec.shape)
    pad_size[0] = pad - vec.size(0)
    return torch.cat([vec, torch.zeros(*pad_size)], dim=0)

#------------------------------------------------------------------------------------------------
## ADDED FOR REINTEGRATION EXPERIMENTS

def pad_mask(mask, pad_length):
    """Pad boolean mask to pad_length with False; mask is 1D (T,)."""
    if mask is None or mask.numel() == 0:
        return torch.zeros(pad_length, dtype=torch.bool)
    if mask.shape[0] >= pad_length:
        return mask[:pad_length]
    pad = torch.zeros(pad_length - mask.shape[0], dtype=torch.bool)
    return torch.cat([mask, pad], dim=0)
#------------------------------------------------------------------------------------------------

def collate_mm_fn_padd(batch):
    # find longest sequence
    if batch[0][0] is not None: max_a_len = max(map(lambda x: x[0].shape[0], batch))
    if batch[0][1] is not None: max_b_len = max(map(lambda x: x[1].shape[0], batch))
    #------------------------------------------------------------------------------------------------
    ## ADDED FOR REINTEGRATION EXPERIMENTS
    has_masks = len(batch[0]) >= 7
    #------------------------------------------------------------------------------------------------

    # batch items are (data_a, data_b, len_a, len_b, label, mask_a, mask_b)
    x_a, x_b, len_a, len_b, ys = list(), list(), list(), list(), list()

    #------------------------------------------------------------------------------------------------
    ## ADDED FOR REINTEGRATION EXPERIMENTS
    mask_a_list, mask_b_list = list(), list()
    #------------------------------------------------------------------------------------------------

    for idx in range(len(batch)):
        x_a.append(pad_tensor(batch[idx][0], pad=max_a_len))
        x_b.append(pad_tensor(batch[idx][1], pad=max_b_len))
        len_a.append(torch.tensor(batch[idx][2]))
        len_b.append(torch.tensor(batch[idx][3]))


        # ys.append(batch[idx][-1])
        #------------------------------------------------------------------------------------------------
        ## ADDED FOR REINTEGRATION EXPERIMENTS
        ys.append(batch[idx][4])
        #------------------------------------------------------------------------------------------------


        #------------------------------------------------------------------------------------------------
        ## ADDED FOR REINTEGRATION EXPERIMENTS
        if has_masks:
            mask_a_list.append(pad_mask(batch[idx][5], max_a_len))
            mask_b_list.append(pad_mask(batch[idx][6], max_b_len))
        else:
            mask_a_list.append(None)
            mask_b_list.append(None)
        #------------------------------------------------------------------------------------------------

    x_a = torch.stack(x_a, dim=0)
    x_b = torch.stack(x_b, dim=0)
    len_a = torch.stack(len_a, dim=0)
    len_b = torch.stack(len_b, dim=0)
    ys = torch.stack(ys, dim=0)

    #------------------------------------------------------------------------------------------------
    ## ADDED FOR REINTEGRATION EXPERIMENTS
    if has_masks:
        mask_a = torch.stack(mask_a_list, dim=0)
        mask_b = torch.stack(mask_b_list, dim=0)
        return x_a, x_b, len_a, len_b, ys, mask_a, mask_b
    #------------------------------------------------------------------------------------------------
    else:
        ## original return
        return x_a, x_b, len_a, len_b, ys

def collate_unimodal_fn_padd(batch):
    # find longest sequence
    if batch[0][0] is not None: max_a_len = max(map(lambda x: x[0].shape[0], batch))
    
    # pad according to max_len
    x_a, len_a, ys = list(), list(), list()
    for idx in range(len(batch)):
        x_a.append(pad_tensor(batch[idx][0], pad=max_a_len))
        len_a.append(torch.tensor(batch[idx][1]))
        ys.append(batch[idx][-1])
    
    # stack all
    x_a = torch.stack(x_a, dim=0)
    len_a = torch.stack(len_a, dim=0)
    ys = torch.stack(ys, dim=0)
    return x_a, len_a, ys


class MMDatasetGenerator(Dataset):
    def __init__(
        self, 
        modalityA, 
        modalityB, 
        default_feat_shape_a,
        default_feat_shape_b,
        data_len: int, 
        simulate_feat=None,
        #------------------------------------------------------------------------------------------------
        ## ADDED FOR REINTEGRATION EXPERIMENTS
        dataset: str='',
        availability_schedule=None,
        schedule_index_map=None,
        #------------------------------------------------------------------------------------------------
    ):
        self.data_len = data_len
        self.modalityA = modalityA
        self.modalityB = modalityB
        self.simulate_feat = simulate_feat
        self.default_feat_shape_a = default_feat_shape_a
        self.default_feat_shape_b = default_feat_shape_b
        self.dataset = dataset
        #------------------------------------------------------------------------------------------------
        ## ADDED FOR REINTEGRATION EXPERIMENTS
        self.availability_schedule = availability_schedule
        self.schedule_index_map = schedule_index_map
        #------------------------------------------------------------------------------------------------

    def __len__(self):
        return self.data_len

    #------------------------------------------------------------------------------------------------
    ## ADDED FOR REINTEGRATION EXPERIMENTS
    def _get_schedule_key(self, item):
        if self.schedule_index_map is not None:
            return self.schedule_index_map[item]
        return item
    #------------------------------------------------------------------------------------------------

    def __getitem__(self, item):
        # read modality
        data_a = self.modalityA[item][-1]
        data_b = self.modalityB[item][-1]
        label = torch.tensor(self.modalityA[item][-2])
        
        # modality A, if missing replace with 0s, and mask
        if data_a is not None: 
            if len(data_a.shape) == 3: data_a = data_a[0]
            data_a = torch.tensor(data_a)
            len_a = len(data_a)
        else: 
            data_a = torch.tensor(np.zeros(self.default_feat_shape_a))
            len_a = 0

        # modality B, if missing replace with 0s
        if data_b is not None:
            if len(data_b.shape) == 3: data_b = data_b[0]
            data_b = torch.tensor(data_b)
            len_b = len(data_b)
        else: 
            data_b = torch.tensor(np.zeros(self.default_feat_shape_b))
            len_b = 0

        #------------------------------------------------------------------------------------------------
        ## ADDED FOR REINTEGRATION EXPERIMENTS
        # per-timestep availability masks (all True for valid positions when no schedule)
        key = self._get_schedule_key(item)
        sched = self.availability_schedule.get(key) if self.availability_schedule else None
        
        if sched is not None:
            mask_a = torch.from_numpy(np.asarray(sched['mask_a'], dtype=bool))[:len_a]
            mask_b = torch.from_numpy(np.asarray(sched['mask_b'], dtype=bool))[:len_b]
        else:
            mask_a = torch.ones(len_a, dtype=torch.bool) if len_a > 0 else torch.zeros(0, dtype=torch.bool)
            mask_b = torch.ones(len_b, dtype=torch.bool) if len_b > 0 else torch.zeros(0, dtype=torch.bool)
        
        return data_a, data_b, len_a, len_b, label, mask_a, mask_b
    #------------------------------------------------------------------------------------------------

class DataloadManager():
    def __init__(
        self, 
        args: dict
    ):
        self.args = args
        self.label_dist_dict = dict()
        # Initialize video feature paths
        if self.args.dataset in ['ucf101', 'mit10', 'mit51', 'mit101', 'crema_d', "ego4d-ttm"]:
            self.get_video_feat_path()
        if self.args.dataset in ['hateful_memes', 'crisis-mmd']:
            self.get_image_feat_path()
            self.get_text_feat_path()
        # Initialize audio feature paths
        if self.args.dataset in ['ucf101', 'mit10', 'mit51', 'mit101', 'meld', 'iemocap', 'crema_d', "ego4d-ttm"]:
            self.get_audio_feat_path()
        
        #------------------------------------------------------------------------------------------------
        ## ADDED FOR REINTEGRATION EXPERIMENTS
        # Lazy-loaded temporal availability sidecar (markov); None for bernoulli
        self._availability_sidecar = None
        #------------------------------------------------------------------------------------------------

        
    def get_audio_feat_path(self):
        """
        Load audio feature path.
        """
        self.audio_feat_path = Path(self.args.data_dir).joinpath(
            'feature', 
            'audio', 
            self.args.audio_feat, 
            self.args.dataset
        )
        return Path(self.audio_feat_path)
    
    def get_video_feat_path(self):
        """
        Load frame-wise video feature path.
        """
        self.video_feat_path = Path(self.args.data_dir).joinpath(
            'feature', 
            'video', 
            self.args.video_feat, 
            self.args.dataset
        )
        return Path(self.video_feat_path)
    
    def get_image_feat_path(self):
        """
        Load image feature path.
        """
        self.img_feat_path = Path(self.args.data_dir).joinpath(
            'feature', 
            'img', 
            self.args.img_feat, 
            self.args.dataset
        )
        return Path(self.img_feat_path)

    def get_text_feat_path(self):
        """
        Load text feature path.
        """
        self.text_feat_path = Path(self.args.data_dir).joinpath(
            'feature', 
            'text', 
            self.args.text_feat, 
            self.args.dataset
        )
        return Path(self.text_feat_path)
    
    def get_client_ids(
            self, 
            fold_idx: int=1
        ):
        """
        Load client ids.
        :param fold_idx: fold index
        :return: None
        """
        if self.args.dataset in ("meld", "iemocap"):
            data_path = self.text_feat_path
        else:
            data_path = self.text_feat_path
        self.client_ids = [id.split('.pkl')[0] for id in os.listdir(str(data_path))]
        self.client_ids.sort()
        
    def load_audio_feat(
            self, 
            client_id: str, 
            fold_idx: int=1
        ) -> dict:
        """
        Load audio feature data different applications.
        :param client_id: client id
        :param fold_idx: fold index
        :return: data_dict: [key, path, label, feature_array]
        """
        if self.args.dataset in ("meld", "iemocap"):
            data_path = self.audio_feat_path.joinpath(f'{client_id}.pkl')
        else:
            data_path = self.audio_feat_path.joinpath(f'{client_id}.pkl')
        with open(str(data_path), "rb") as f: 
            data_dict = pickle.load(f)
        return data_dict
    
    
    
    def load_text_feat(
            self, 
            client_id: str, 
            fold_idx: int=1
        ) -> dict:
        """
        Load text feature data.
        :param client_id: client id
        :param fold_idx: fold index
        :return: data_dict: [key, path, label, feature_array]
        """
        if self.args.dataset in ("meld", "iemocap"):
            data_path = self.text_feat_path.joinpath(f'{client_id}.pkl')
        else:
            data_path = self.text_feat_path.joinpath(f'{client_id}.pkl')
        with open(str(data_path), "rb") as f: 
            data_dict = pickle.load(f)
        return data_dict

    def get_client_sim_dict(
            self, 
            client_id
        ):
        """
        Set dataloader for training/dev/test.
        :param client_id: client_id
        :return: dataloader: torch dataloader
        """
        if self.sim_data:
            return self.sim_data[client_id]
        return None

    #------------------------------------------------------------------------------------------------
    ## ADDED FOR REINTEGRATION EXPERIMENTS
    def get_availability_schedule(self, client_id):
        """
        Return per-sample temporal availability for a client when using Markov.
        If availability_process != "markov" or no sidecar path, returns None
        (caller should treat as all-ON masks).
        :param client_id: client id (e.g. "0", "dev", "test")
        :return: dict idx -> {mask_a, mask_b, events_a, events_b, r_a, r_b} or None
        """
        process = getattr(self.args, 'availability_process', None)
        path = getattr(self.args, 'availability_sidecar_path', None)
        if process != 'markov' or not path:
            return None
        if self._availability_sidecar is None:
            with open(str(Path(path)), 'rb') as f:
                self._availability_sidecar = pickle.load(f)
        return self._availability_sidecar.get(client_id)
    #------------------------------------------------------------------------------------------------
    
    def get_label_dist(
        self,
        scenes: list,
        client_id: str
    ):
        """
        Compute label distribution from a list of scenes.
        :param scenes: list of scenes, each scene is a list of utterance rows
                       utterance row: [Filename, Path, Label, Utterance, None]
        :param client_id: client identifier
        """
        label_list = []
        for scene in scenes:
            for utt in scene:
                label_list.append(utt[2])   # Label is at index 2
        self.label_dist_dict[client_id] = Counter(label_list)
        
    def set_scene_dataloader(
        self,
        scenes: list,
        audio_feat_dict: dict,
        text_feat_dict: dict,
        default_feat_shape_a: np.array = np.array([1000, 80]),
        default_feat_shape_b: np.array = np.array([10, 512]),
        p_stay_absent: float = 0.7,
        p_stay_present: float = 0.75,
        shuffle: bool = False,
        apply_mask: bool = True,
    ) -> DataLoader:
        """
        Scene-level dataloader for reintegration experiments.

        Args:
            scenes:           list of scenes from partition.json
                              partition[client_id] = list of scenes
                              scene = list of utterances
                              utt   = [Filename, Path, Label, Utterance, None]
            audio_feat_dict:  filename-keyed dict from extract_audio_features_scene.py
                              e.g. {'dia64_utt3': np.ndarray(T_frames, 80)}
            text_feat_dict:   filename-keyed dict from extract_text_features_scene.py
                              e.g. {'dia64_utt3': np.ndarray(T_tokens, 512)}
            default_feat_shape_a: fallback shape if filename missing (T_frames, D_audio)
            default_feat_shape_b: fallback shape if filename missing (T_tokens, D_text)
            p_stay_absent:    P(absent→absent) Markov parameter
            p_stay_present:   P(present→present) Markov parameter
            shuffle:          True for training clients, False for dev/test
            apply_mask:       True applies Markov mask; False yields all-ones mask
                              (stable condition). Training always uses True since
                              SceneGRUWrapper.forward_two_pass() handles the stable
                              pass internally.
        Returns:
            DataLoader yielding one scene per iteration.
            Each batch: (scene_x_a, scene_x_b, scene_len_a, scene_len_b,
                         scene_labels, scene_mask)
        """
        return build_scene_dataloader(
            scenes          = scenes,
            audio_feat_dict = audio_feat_dict,
            text_feat_dict  = text_feat_dict,
            default_shape_a = default_feat_shape_a,
            default_shape_b = default_feat_shape_b,
            p_stay_absent   = p_stay_absent,
            p_stay_present  = p_stay_present,
            apply_mask      = apply_mask,
            shuffle         = shuffle,
        )

    def set_dataloader(
            self, 
            data_a: dict,
            data_b: dict,
            default_feat_shape_a: np.array=np.array([0, 0]),
            default_feat_shape_b: np.array=np.array([0, 0]),
            client_sim_dict: dict=None,
            shuffle: bool=False,
            #------------------------------------------------------------------------------------------------
            ## ADDED FOR REINTEGRATION EXPERIMENTS
            client_id=None,
            #------------------------------------------------------------------------------------------------
        ) -> (DataLoader):
        """
        Set dataloader for training/dev/test.
        :param data_a: modality A data
        :param data_b: modality B data
        :param default_feat_shape_a: default input shape for modality A, fill 0 in missing modality case
        :param default_feat_shape_b: default input shape for modality B, fill 0 in missing modality case
        :param shuffle: shuffle flag for dataloader, True for training; False for dev and test
        :param client_id: optional; when set and availability_process==markov, per-sample masks are loaded from sidecar
        :return: dataloader: torch dataloader
        """
        # modify data based on simulation
        labeled_data_idx, unlabeled_data_idx = list(), list()
        if client_sim_dict is not None:
            for idx in range(len(client_sim_dict)):
                # read simulate feature
                sim_data = client_sim_dict[idx][-1]
                # read modality A
                if sim_data[0] == 1: data_a[idx][-1] = None
                # read modality B
                if sim_data[1] == 1: data_b[idx][-1] = None
                # label noise
                data_a[idx][-2] = sim_data[2]
                # missing label
                if sim_data[-1] == 0: labeled_data_idx.append(idx)
                else: unlabeled_data_idx.append(idx)
            
            # return None when both modalities are missing
            if sim_data[0] == 1 and sim_data[1] == 1:
                return None
            
            labeled_data_a, unlabeled_data_a = list(), list()
            labeled_data_b, unlabeled_data_b = list(), list()
            if len(unlabeled_data_idx) > 0:
                for idx in labeled_data_idx:
                    labeled_data_a.append(data_a[idx])
                    labeled_data_b.append(data_b[idx])
                for idx in unlabeled_data_idx:
                    unlabeled_data_a.append(data_a[idx])
                    unlabeled_data_b.append(data_b[idx])
                data_a = labeled_data_a
                data_b = labeled_data_b

        if len(data_a) == 0: return None
        #------------------------------------------------------------------------------------------------
        ## ADDED FOR REINTEGRATION EXPERIMENTS
        availability_schedule = self.get_availability_schedule(client_id) if client_id is not None else None
        schedule_index_map = labeled_data_idx if (client_sim_dict is not None and len(unlabeled_data_idx) > 0) else None
        #------------------------------------------------------------------------------------------------
        data_ab = MMDatasetGenerator(
            data_a, 
            data_b,
            default_feat_shape_a,
            default_feat_shape_b,
            len(data_a),
            dataset=self.args.dataset,
            #------------------------------------------------------------------------------------------------
            ## ADDED FOR REINTEGRATION EXPERIMENTS
            availability_schedule=availability_schedule,
            schedule_index_map=schedule_index_map,
            #------------------------------------------------------------------------------------------------
        )
        if shuffle:
            # we use args input batch size for train, typically set as 16 in FL setup
            dataloader = DataLoader(
                data_ab, 
                batch_size=int(self.args.batch_size), 
                num_workers=0, 
                shuffle=shuffle, 
                collate_fn=collate_mm_fn_padd
            )
        else:
            # we use a larger batch size for validation and testing
            dataloader = DataLoader(
                data_ab, 
                batch_size=64, 
                num_workers=0, 
                shuffle=shuffle, 
                collate_fn=collate_mm_fn_padd
            )
        return dataloader


    def get_simulation_setting(self, alpha=None):
        """
        Load get simulation setting string.
        :param alpha: alpha in manual split
        :return: None
        """
        self.setting_str = ''
        # 1. missing modality
        if self.args.missing_modality == True:
            self.setting_str += 'mm'+str(self.args.missing_modailty_rate).replace('.', '')
        # 2. label nosiy
        if self.args.label_nosiy == True:
            if len(self.setting_str) != 0: self.setting_str += '_'
            self.setting_str += 'ln'+str(self.args.label_nosiy_level).replace('.', '')
        # 3. missing labels
        if self.args.missing_label == True:
            if len(self.setting_str) != 0: self.setting_str += '_'
            self.setting_str += 'ml'+str(self.args.missing_label_rate).replace('.', '')
        # 4. alpha for manual split
        if len(self.setting_str) != 0:
            if alpha is not None:
                alpha_str = str(self.args.alpha).replace('.', '')
                self.setting_str += f'_alpha{alpha_str}'