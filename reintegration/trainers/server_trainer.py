import json
import math
import numpy as np
import pandas as pd
import copy, pdb, time, warnings, torch
import torch.nn.functional as F

from pathlib import Path
from copy import deepcopy
from typing import Optional
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import recall_score, f1_score

from reintegration.evaluation import EvalMetric

import logging


def sanitize_for_json(obj):
    """
    Recursively convert nested dicts/lists to JSON-safe values: float NaN/Inf -> None,
    dict keys -> str, numpy scalars/arrays -> Python types.
    """
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        x = float(obj)
        return None if math.isnan(x) or math.isinf(x) else x
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    return obj


logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)


class Server(object):
    def __init__(self, args, model, device, criterion, client_ids):
        self.args         = args
        self.device       = device
        self.result_dict  = dict()
        self.global_model = model
        self.criterion    = criterion
        self.client_ids   = client_ids
        self.multilabel   = True if args.dataset == 'ptb-xl' else False
        self.model_setting_str = self.get_model_setting()

        if self.args.fed_alg == 'scaffold':
            self.server_control  = self.init_control(model)
            self.set_control_device(self.server_control, True)
            self.client_controls = {
                client_id: self.init_control(model) for client_id in self.client_ids
            }
        elif self.args.fed_alg == 'fed_opt':
            self.global_optimizer = self._initialize_global_optimizer()

    def _initialize_global_optimizer(self):
        return torch.optim.SGD(
            self.global_model.parameters(),
            lr=self.args.global_learning_rate,
            momentum=0.9, weight_decay=0.0
        )

    def set_client_control(self, client_id, client_control):
        self.client_controls[client_id] = client_control

    def initialize_log(self, fold_idx: int=1):
        self.fold_idx   = fold_idx
        _rn = getattr(self.args, "run_name", None)
        _run = (_rn.strip(),) if _rn and str(_rn).strip() else ()
        self.log_path   = Path(self.args.data_dir).joinpath(
            'log', self.args.fed_alg, self.args.dataset,
            self.feature, self.att, self.model_setting_str,
            *_run,
            f'fold{fold_idx}', 'raw_log'
        )
        self.result_path = Path(self.args.data_dir).joinpath(
            'log', self.args.fed_alg, self.args.dataset,
            self.feature, self.att, self.model_setting_str,
            *_run,
            f'fold{fold_idx}'
        )
        Path.mkdir(self.log_path, parents=True, exist_ok=True)
        self.log_writer = SummaryWriter(
            str(self.log_path),
            filename_suffix=f'_{self.model_setting_str}'
        )
        self.best_test_dict = list()

    def get_model_setting(self):
        model_setting_str = ""
        if self.args.dataset in ['meld', 'iemocap']:
            if self.args.modality == "multimodal":
                self.feature = f'{self.args.audio_feat}_{self.args.text_feat}'
            elif self.args.modality == "audio":
                self.feature = f'{self.args.audio_feat}'
            elif self.args.modality == "text":
                self.feature = f'{self.args.text_feat}'

        if len(model_setting_str) == 0:
            model_setting_str = 'hid' + str(self.args.hid_size)
        else:
            model_setting_str += '_hid' + str(self.args.hid_size)
        model_setting_str += '_le'  + str(self.args.local_epochs)
        model_setting_str += '_lr'  + str(self.args.learning_rate).replace('.', '')
        if self.args.fed_alg == 'fed_opt':
            model_setting_str += '_gl' + str(self.args.global_learning_rate).replace('.', '')
        model_setting_str += '_bs' + str(self.args.batch_size)
        model_setting_str += '_sr' + str(self.args.sample_rate).replace('.', '')
        model_setting_str += '_ep' + str(self.args.num_epochs)
        if self.args.fed_alg == 'fed_prox':
            model_setting_str += '_mu' + str(self.args.mu).replace('.', '')

        self.att = f'{self.args.att_name}' if self.args.att else 'no_att'

        if self.args.missing_modality:
            model_setting_str += '_mm' + str(self.args.missing_modailty_rate).replace('.', '')
        if self.args.label_nosiy:
            model_setting_str += '_ln' + str(self.args.label_nosiy_level).replace('.', '')
        if self.args.missing_label:
            model_setting_str += '_ml' + str(self.args.missing_label_rate).replace('.', '')
        return model_setting_str

    def sample_clients(self, num_of_clients, sample_rate=0.1, fold_idx: int = 1):
        """
        Build per-round client index lists for FedAvg partial participation.

        Seeding uses both fold_idx and epoch so each outer-loop fold gets a
        different subsample schedule; epoch-only seeding would repeat the same
        schedule on every fold.
        """
        self.clients_list = list()
        # Large stride keeps (fold, epoch) pairs from colliding for typical num_epochs.
        _fold_stride = 100_000
        _sched = getattr(self.args, "client_schedule_seed", None)
        for epoch in range(int(self.args.num_epochs)):
            if _sched is not None:
                np.random.seed(int(_sched) + _fold_stride * int(fold_idx) + epoch)
            else:
                np.random.seed(_fold_stride * int(fold_idx) + epoch)
            idxs_clients = np.random.choice(
                range(num_of_clients),
                int(sample_rate * num_of_clients),
                replace=False
            )
            self.clients_list.append(idxs_clients)
        self.num_of_clients = num_of_clients

    def initialize_epoch_updates(self, epoch):
        self.epoch = epoch
        self.model_updates    = list()
        self.num_samples_list = list()
        self.delta_controls   = list()
        self.result_dict[self.epoch] = dict()
        self.result_dict[self.epoch]['train'] = list()
        self.result_dict[self.epoch]['dev']   = list()
        self.result_dict[self.epoch]['test']  = list()

        # Prune result_dict to current + best epoch only.
        keep = {self.epoch}
        if hasattr(self, 'best_epoch'):
            keep.add(self.best_epoch)
        for old_epoch in list(self.result_dict.keys()):
            if old_epoch not in keep:
                del self.result_dict[old_epoch]

        # Prune result_dict: keep only current epoch and best epoch.
        # Without this, result_dict accumulates entries for all 200+ rounds,
        # holding logit arrays and metric dicts in memory for the entire run.
        keep = {self.epoch}
        if hasattr(self, 'best_epoch'):
            keep.add(self.best_epoch)
        for old_epoch in list(self.result_dict.keys()):
            if old_epoch not in keep:
                del self.result_dict[old_epoch]

    def get_parameters(self):
        return self.global_model.state_dict()

    def get_model_result(self):
        return self.result

    def inference(self, dataloader):
        """
        Scene-level inference for dev/test evaluation.

        always evaluates with the stable (all-ones) audio mask.
        The model was trained on full audio; dev/test UAR is measured in
        the same condition to track learning progress across FL rounds.

        The reintegration contrast is measured separately at the end of
        training via run_reintegration_eval(), which runs both the stable
        and masked passes on the test set using the best checkpoint.
        """
        self.global_model.eval()
        self.eval = EvalMetric(self.multilabel)

        for batch_idx, batch_data in enumerate(dataloader):
            if self.args.modality == "multimodal":
                (scene_x_a, scene_x_b,
                 scene_len_a, scene_len_b,
                 scene_labels, _) = batch_data   # scene_mask unused during inference

                scene_labels = scene_labels.to(self.device)
                T = scene_labels.shape[0]

                # Stable mask: audio always present, matches training condition
                stable_mask = torch.ones(T, device=self.device, dtype=torch.long)

                preds, _ = self.global_model(
                    scene_x_a, scene_x_b,
                    scene_len_a, scene_len_b,
                    stable_mask,
                    self.device
                )
                log_preds = torch.log_softmax(preds, dim=-1)
                loss = self.criterion(log_preds, scene_labels)

                if not self.multilabel:
                    self.eval.append_classification_results(
                        scene_labels, log_preds, loss
                    )
                else:
                    self.eval.append_multilabel_results(
                        scene_labels, log_preds, loss
                    )
            else:
                x, l, y = batch_data
                x, l, y = x.to(self.device), l.to(self.device), y.to(self.device)
                outputs, _ = self.global_model(x.float(), l)
                if not self.multilabel:
                    outputs = torch.log_softmax(outputs, dim=1)
                loss = self.criterion(outputs, y)
                if not self.multilabel:
                    self.eval.append_classification_results(y, outputs, loss)
                else:
                    self.eval.append_multilabel_results(y, outputs, loss)

        if not self.multilabel:
            self.result = self.eval.classification_summary()
        else:
            self.result = self.eval.multilabel_summary()

    def run_reintegration_eval(
        self,
        dataloader,
        recovery_window: int = 4,
        split_label: Optional[str] = None,
    ):
        """
        Per-timestep reintegration evaluation. the primary result.

        Runs on the best checkpoint after FL training completes.
        For each test scene, the model is forwarded twice:
            - Stable pass  (all-ones mask): model in its trained condition
            - Masked pass  (Markov mask):   audio absent then returns

        Reintegration events are identified where mask[t-1]==0, mask[t]==1.
        At each event boundary and for recovery_window utterances after it,
        delta = stable_correct - masked_correct is recorded.

        A positive mean_delta at offset 0 means the model makes more correct
        predictions when audio was continuously present than when audio just
        returned after an absence run the reintegration phenomenon exists.

        A decaying recovery curve (delta shrinks at offsets 1,2,3,4) means
        the cost is localised to the boundary. A flat curve means the hidden
        state divergence persists across subsequent utterances.

        Window UAR (uar_*_window): macro recall on the union of post-return
        timesteps (+0..+recovery_window per event), with each timestep counted
        once per scene if overlapping windows share indices. Excludes absent
        timesteps so global delta_uar is not confounded with long-absence drag.

        Soft belief metrics (same timesteps as the binary recovery curve): for each
        offset k, log p(y*) gap and KL(P_stable||P_masked) use full logits; they
        measure how much the stable vs masked distributions differ, not only argmax.

        Args:
            dataloader:      scene-level DataLoader (apply_mask=True)
            recovery_window: utterances after t_reint to track (default 4)
            split_label:     optional tag (e.g. dev / test) prefixed in log lines

        Returns dict with keys:
            mean_delta          : float        — mean delta at offset 0
            delta_by_offset     : dict[int, list[float]]
            mean_delta_by_offset: dict[int, float] — the recovery curve
            n_reint_events      : int
            uar_stable          : float        — macro recall, stable (%)
            uar_masked          : float        — macro recall, masked (%)
            delta_uar           : float        — uar_stable - uar_masked
            uar_stable_window   : float|None   — macro UAR on post-return window timesteps only (stable)
            uar_masked_window   : float|None   — same timesteps, masked pass
            delta_uar_window      : float|None   — uar_stable_window - uar_masked_window
            n_window_timesteps    : int          — unique timesteps in union of windows (per-scene dedup)
            mean_logp_gap_by_offset : dict[int, float] — mean log p_s(y*) − log p_m(y*) at each offset (nats)
            mean_kl_forward_by_offset: dict[int, float] — mean KL(P_stable || P_masked) at each offset (nats)
            mean_disagree_by_offset : dict[int, float] — mean 1[argmax_s ≠ argmax_m] at each offset
            logp_gap_by_offset    : dict[int, list[float]] — raw values for aggregation / bootstrap
            kl_forward_by_offset  : dict[int, list[float]]
            disagree_by_offset    : dict[int, list[float]]
            split_label           : optional str — same tag passed in (stored for JSON)
        """
        self.global_model.eval()

        all_preds_stable = []
        all_preds_masked = []
        all_labels       = []
        n_reint_events   = 0
        delta_by_offset  = {k: [] for k in range(recovery_window + 1)}
        logp_gap_by_offset = {k: [] for k in range(recovery_window + 1)}
        kl_forward_by_offset = {k: [] for k in range(recovery_window + 1)}
        disagree_by_offset = {k: [] for k in range(recovery_window + 1)}
        win_labels = []
        win_preds_stable = []
        win_preds_masked = []

        for batch_data in dataloader:
            if self.args.modality != "multimodal":
                continue

            (scene_x_a, scene_x_b,
             scene_len_a, scene_len_b,
             scene_labels, scene_mask) = batch_data

            scene_labels = scene_labels.to(self.device)
            scene_mask   = scene_mask.to(self.device)
            T = scene_labels.shape[0]

            # Stable pass — model in its trained condition
            ones_mask = torch.ones(T, device=self.device, dtype=torch.long)
            with torch.no_grad():
                preds_stable, _ = self.global_model(
                    scene_x_a, scene_x_b,
                    scene_len_a, scene_len_b,
                    ones_mask, self.device
                )

            # Masked pass — Markov audio availability
            with torch.no_grad():
                preds_masked, _ = self.global_model(
                    scene_x_a, scene_x_b,
                    scene_len_a, scene_len_b,
                    scene_mask, self.device
                )

            pred_s    = preds_stable.argmax(dim=-1)   # (T,)
            pred_m    = preds_masked.argmax(dim=-1)   # (T,)
            labels    = scene_labels                  # (T,)

            all_preds_stable.extend(pred_s.cpu().tolist())
            all_preds_masked.extend(pred_m.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            mask_np   = scene_mask.cpu().numpy()
            pred_s_np = pred_s.cpu().numpy()
            pred_m_np = pred_m.cpu().numpy()
            labels_np = labels.cpu().numpy()

            log_p_s = F.log_softmax(preds_stable, dim=-1).detach().cpu().numpy()
            log_p_m = F.log_softmax(preds_masked, dim=-1).detach().cpu().numpy()

            seen_window_t = set()
            for t in range(1, T):
                if mask_np[t - 1] == 0 and mask_np[t] == 1:
                    n_reint_events += 1
                    for k in range(recovery_window + 1):
                        t_k = t + k
                        if t_k >= T:
                            break
                        if k > 0 and mask_np[t_k] == 0:
                            break   # another absence run stop this event's window
                        correct_s = int(pred_s_np[t_k] == labels_np[t_k])
                        correct_m = int(pred_m_np[t_k] == labels_np[t_k])
                        delta_by_offset[k].append(correct_s - correct_m)

                        y_idx = int(labels_np[t_k])
                        logp_gap = float(log_p_s[t_k, y_idx] - log_p_m[t_k, y_idx])
                        logp_gap_by_offset[k].append(logp_gap)

                        # KL(P_stable || P_masked) in nats; same support as softmax rows
                        p_row = np.exp(log_p_s[t_k])
                        kl_f = float(np.sum(p_row * (log_p_s[t_k] - log_p_m[t_k])))
                        kl_forward_by_offset[k].append(kl_f)

                        disagree_by_offset[k].append(
                            float(pred_s_np[t_k] != pred_m_np[t_k])
                        )

                        if t_k not in seen_window_t:
                            seen_window_t.add(t_k)
                            win_labels.append(int(labels_np[t_k]))
                            win_preds_stable.append(int(pred_s_np[t_k]))
                            win_preds_masked.append(int(pred_m_np[t_k]))

        uar_stable = recall_score(
            all_labels, all_preds_stable, average='macro', zero_division=0
        ) * 100
        uar_masked = recall_score(
            all_labels, all_preds_masked, average='macro', zero_division=0
        ) * 100

        if win_labels:
            uar_stable_window = recall_score(
                win_labels, win_preds_stable, average='macro', zero_division=0
            ) * 100
            uar_masked_window = recall_score(
                win_labels, win_preds_masked, average='macro', zero_division=0
            ) * 100
            delta_uar_window = uar_stable_window - uar_masked_window
        else:
            uar_stable_window = None
            uar_masked_window = None
            delta_uar_window = None

        def _mean_list(d):
            return {
                kk: float(np.mean(vv)) if vv else float('nan')
                for kk, vv in d.items()
            }

        mean_delta_by_offset = _mean_list(delta_by_offset)
        mean_delta = mean_delta_by_offset[0]

        mean_logp_gap_by_offset = _mean_list(logp_gap_by_offset)
        mean_kl_forward_by_offset = _mean_list(kl_forward_by_offset)
        mean_disagree_by_offset = _mean_list(disagree_by_offset)

        curve_str = ', '.join(
            f'+{k}:{mean_delta_by_offset[k]:.4f} (n={len(delta_by_offset[k])})'
            for k in range(recovery_window + 1)
        )
        logp_str = ', '.join(
            f'+{k}:{mean_logp_gap_by_offset[k]:.4f}'
            for k in range(recovery_window + 1)
        )
        kl_str = ', '.join(
            f'+{k}:{mean_kl_forward_by_offset[k]:.4f}'
            for k in range(recovery_window + 1)
        )
        dis_str = ', '.join(
            f'+{k}:{mean_disagree_by_offset[k]:.4f}'
            for k in range(recovery_window + 1)
        )
        lp = f'[{split_label}] ' if split_label else ''
        logging.info(
            f'{lp}Reintegration eval: n_events={n_reint_events}, '
            f'UAR_stable={uar_stable:.2f}%, UAR_masked={uar_masked:.2f}%'
        )
        logging.info(f'{lp}Recovery curve: {curve_str}')
        logging.info(f'{lp}Log-prob gap on true class (nats): {logp_str}')
        logging.info(f'{lp}KL(P_stable || P_masked) (nats): {kl_str}')
        logging.info(f'{lp}Argmax disagreement rate: {dis_str}')
        if n_reint_events == 0:
            logging.info(
                f'{lp}No reintegration events (no mask 0→1 transitions); per-offset '
                f'curves are empty. Scene-level UAR_stable / UAR_masked still summarize '
                f'the two full-scene passes.'
            )

        return {
            'split_label':          split_label,
            'mean_delta':           mean_delta,
            'delta_by_offset':      {k: v for k, v in delta_by_offset.items()},
            'mean_delta_by_offset': mean_delta_by_offset,
            'n_reint_events':       n_reint_events,
            'uar_stable':           uar_stable,
            'uar_masked':           uar_masked,
            'delta_uar':            uar_stable - uar_masked,
            'uar_stable_window':    uar_stable_window,
            'uar_masked_window':    uar_masked_window,
            'delta_uar_window':     delta_uar_window,
            'n_window_timesteps':   len(win_labels),
            'mean_logp_gap_by_offset': mean_logp_gap_by_offset,
            'mean_kl_forward_by_offset': mean_kl_forward_by_offset,
            'mean_disagree_by_offset': mean_disagree_by_offset,
            'logp_gap_by_offset':   {k: v for k, v in logp_gap_by_offset.items()},
            'kl_forward_by_offset': {k: v for k, v in kl_forward_by_offset.items()},
            'disagree_by_offset':   {k: v for k, v in disagree_by_offset.items()},
        }

    # Remaining methods unchanged 

    def get_num_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.global_model.parameters())
        num_params = sum([np.prod(p.size()) for p in model_parameters]) / 1000
        logging.info(f'Number of Parameters: {num_params} K')
        return num_params

    def log_classification_result(self, data_split: str, metric: str='uar'):
        if data_split == 'train':
            loss    = np.mean([d['loss']    for d in self.result_dict[self.epoch][data_split]])
            acc     = np.mean([d['acc']     for d in self.result_dict[self.epoch][data_split]])
            uar     = np.mean([d['uar']     for d in self.result_dict[self.epoch][data_split]])
            top5_acc= np.mean([d['top5_acc']for d in self.result_dict[self.epoch][data_split]])
            f1      = np.mean([d['f1']      for d in self.result_dict[self.epoch][data_split]])
        else:
            loss     = self.result_dict[self.epoch][data_split]['loss']
            acc      = self.result_dict[self.epoch][data_split]['acc']
            uar      = self.result_dict[self.epoch][data_split]['uar']
            top5_acc = self.result_dict[self.epoch][data_split]['top5_acc']
            f1       = self.result_dict[self.epoch][data_split]['f1']

        if data_split == 'train': logging.info(f'Current Round: {self.epoch}')
        if metric == 'uar':
            logging.info(f'{data_split} set, Loss: {loss:.3f}, UAR: {uar:.2f}%, Acc: {acc:.2f}%')
        elif metric == 'f1':
            logging.info(f'{data_split} set, Loss: {loss:.3f}, Macro-F1: {f1:.2f}%, Acc: {acc:.2f}%')

        self.log_writer.add_scalar(f'Loss/{data_split}',     loss,     self.epoch)
        self.log_writer.add_scalar(f'Acc/{data_split}',      acc,      self.epoch)
        self.log_writer.add_scalar(f'UAR/{data_split}',      uar,      self.epoch)
        self.log_writer.add_scalar(f'F1/{data_split}',       f1,       self.epoch)
        self.log_writer.add_scalar(f'Top5_Acc/{data_split}', top5_acc, self.epoch)

    def save_result(self, file_path):
        jsonString = json.dumps(self.result_dict, indent=4)
        with open(str(file_path), "w") as f:
            f.write(jsonString)

    def save_train_updates(self, model_updates, num_sample, result, delta_control=None):
        # Move state dict tensors to CPU before storing. keeps GPU memory free
        # between client training and aggregation.
        cpu_updates = {k: v.cpu() for k, v in model_updates.items()}
        self.model_updates.append(cpu_updates)
        self.num_samples_list.append(num_sample)
        self.result_dict[self.epoch]['train'].append(result)
        self.delta_controls.append(delta_control)

    def log_epoch_result(self, metric: str='acc'):
        if len(self.best_test_dict) == 0:
            self.best_epoch     = self.epoch
            self.best_dev_dict  = self.result_dict[self.epoch]['dev']
            self.best_test_dict = self.result_dict[self.epoch]['test']

        if self.result_dict[self.epoch]['dev'][metric] > self.best_dev_dict[metric]:
            self.best_epoch     = self.epoch
            self.best_dev_dict  = self.result_dict[self.epoch]['dev']
            self.best_test_dict = self.result_dict[self.epoch]['test']
            torch.save(
                deepcopy(self.global_model.state_dict()),
                str(self.result_path.joinpath('model.pt'))
            )

        best_dev_uar  = self.best_dev_dict['uar']
        best_dev_acc  = self.best_dev_dict['acc']
        best_test_uar = self.best_test_dict['uar']
        best_test_acc = self.best_test_dict['acc']

        logging.info(f'Best epoch {self.best_epoch}')
        logging.info(f'Best dev  UAR {best_dev_uar:.2f}%,  Acc {best_dev_acc:.2f}%')
        logging.info(f'Best test UAR {best_test_uar:.2f}%, Acc {best_test_acc:.2f}%')

    def summarize_dict_results(self):
        result = dict()
        result['acc']      = self.best_test_dict['acc']
        result['top5_acc'] = self.best_test_dict['top5_acc']
        result['uar']      = self.best_test_dict['uar']
        result['f1']       = self.best_test_dict['f1']
        return result

    def average_weights(self):
        if len(self.num_samples_list) == 0:
            return
        total_num_samples = np.sum(self.num_samples_list)
        # model_updates are on CPU (offloaded in save_train_updates).
        # Aggregate on CPU, then load into model (which moves to device).
        w_avg = copy.deepcopy(self.model_updates[0])

        for key in w_avg.keys():
            if self.args.fed_alg == 'scaffold':
                w_avg[key] = torch.div(self.model_updates[0][key], len(self.model_updates))
            else:
                w_avg[key] = self.model_updates[0][key] * (self.num_samples_list[0] / total_num_samples)

        for key in w_avg.keys():
            for i in range(1, len(self.model_updates)):
                if self.args.fed_alg == 'scaffold':
                    w_avg[key] += torch.div(self.model_updates[i][key], len(self.model_updates))
                else:
                    w_avg[key] += torch.div(
                        self.model_updates[i][key] * self.num_samples_list[i],
                        total_num_samples
                    )

        if self.args.fed_alg == 'fed_opt':
            self.update_global(copy.deepcopy(w_avg))
        else:
            self.global_model.load_state_dict(copy.deepcopy(w_avg))

        if self.args.fed_alg == 'scaffold':
            self.update_server_control()

        # Free per-client state_dict copies immediately after aggregation.
        # Holding them until initialize_epoch_updates doubles peak memory.
        del w_avg
        self.model_updates.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def save_json_file(self, data_dict, data_path):
        jsonString = json.dumps(data_dict, indent=4)
        with open(str(data_path), "w") as f:
            f.write(jsonString)

    def init_control(self, model):
        return {
            name: torch.zeros_like(p.data).cpu()
            for name, p in model.state_dict().items()
        }

    def set_control_device(self, control, device=True):
        for name in control.keys():
            control[name] = control[name].to(self.device) if device else control[name].cpu()
