import collections
import numpy as np
import copy, pdb, time, warnings, torch

from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score

from .optimizer import FedProxOptimizer

warnings.filterwarnings('ignore')
from reintegration.evaluation import EvalMetric


class ClientFedAvg(object):
    """
    FedAvg client trainer — Phase 1 (establish reintegration phenomenon).

    Training is stable-only: audio is always present during training.
    The model never sees audio absence, so it learns full (A,T) fusion
    without any strategy for handling missingness.

    At test time, run_reintegration_eval() presents the trained model with
    scenes where audio is absent for a run then returns (Markov mask).
    Any delta at t_reint is the unmitigated reintegration cost — the
    phenomenon in its natural form, unaffected by robustness training.

    Batch data contract (multimodal):
        scene_x_a      : list of T audio tensors, each (1, T_frames, D_audio)
        scene_x_b      : list of T text tensors,  each (1, T_tokens, D_text)
        scene_len_a    : list of T length tensors, each (1,)
        scene_len_b    : list of T length tensors, each (1,)
        scene_labels   : (T,) label tensor
        scene_mask     : (T,) int tensor — Markov mask (unused during training,
                         present in batch because dataloader always yields it)
    """

    def __init__(
        self,
        args,
        device,
        criterion,
        dataloader,
        model,
        label_dict=None,
        num_class=None
    ):
        self.args       = args
        self.model      = model
        self.device     = device
        self.criterion  = criterion
        self.dataloader = dataloader
        self.multilabel = True if args.dataset == 'ptb-xl' else False

    def get_parameters(self):
        return self.model.state_dict()

    def get_model_result(self):
        return self.result

    def update_weights(self):
        self.model.train()
        self.eval = EvalMetric(self.multilabel)

        if self.args.fed_alg in ['fed_avg', 'fed_opt']:
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.args.learning_rate,
                momentum=0.9,
                weight_decay=1e-5
            )
        else:
            optimizer = FedProxOptimizer(
                self.model.parameters(),
                lr=self.args.learning_rate,
                momentum=0.9,
                weight_decay=1e-5,
                mu=self.args.mu
            )

        for iter in range(int(self.args.local_epochs)):
            for batch_idx, batch_data in enumerate(self.dataloader):

                self.model.zero_grad()
                optimizer.zero_grad()

                if self.args.modality == "multimodal":
                    # ── Unpack scene-level batch ───────────────────────────
                    (scene_x_a, scene_x_b,
                     scene_len_a, scene_len_b,
                     scene_labels, _) = batch_data   # scene_mask unused in Phase 1

                    scene_labels = scene_labels.to(self.device)   # (T,)
                    T = scene_labels.shape[0]

                    # ── Stable-only forward pass ───────────────────────────
                    # Audio always present: all-ones mask.
                    # The model learns full (A,T) fusion exclusively.
                    # No masked pass — the model must not learn any strategy
                    # for handling absence, so the reintegration dip at test
                    # time reflects the unmitigated phenomenon.
                    stable_mask = torch.ones(T, device=self.device, dtype=torch.long)

                    preds, _ = self.model(
                        scene_x_a, scene_x_b,
                        scene_len_a, scene_len_b,
                        stable_mask,
                        self.device
                    )
                    # preds: (T, num_classes)

                    log_preds = torch.log_softmax(preds, dim=-1)   # (T, C)
                    loss = self.criterion(log_preds, scene_labels)

                else:
                    # ── Unimodal path — unchanged ─────────────────────────
                    x, l, y = batch_data
                    x, l, y = x.to(self.device), l.to(self.device), y.to(self.device)
                    outputs, _ = self.model(x.float(), l)
                    if not self.multilabel:
                        outputs = torch.log_softmax(outputs, dim=1)
                    loss = self.criterion(outputs, y)

                # ── Backward ──────────────────────────────────────────────
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                optimizer.step()

                # ── Metrics ───────────────────────────────────────────────
                if self.args.modality == "multimodal":
                    if not self.multilabel:
                        self.eval.append_classification_results(
                            scene_labels, log_preds, loss
                        )
                else:
                    if not self.multilabel:
                        self.eval.append_classification_results(y, outputs, loss)
                    else:
                        self.eval.append_multilabel_results(y, outputs, loss)

        if not self.multilabel:
            self.result = self.eval.classification_summary()
        else:
            self.result = self.eval.multilabel_summary()
