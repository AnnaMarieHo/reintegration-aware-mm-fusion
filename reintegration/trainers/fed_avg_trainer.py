import collections
import numpy as np
import copy, pdb, time, warnings, torch

from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score

from .optimizer import FedProxOptimizer

warnings.filterwarnings('ignore')
from my_extensions.reintegration.evaluation import EvalMetric


class ClientFedAvg(object):
    """
    FedAvg client trainer modified for scene-level reintegration experiments.

    Key changes from original:
        1. update_weights processes scenes one at a time (scene loop).
           Each scene is forwarded twice per iteration:
               - stable pass  (all-ones audio mask)
               - masked pass  (Markov audio availability mask)
           Loss = loss_stable + loss_masked, backprop once.
           This trains the model on both conditions without splitting the dataset.

        2. Batch data contract changed:
               scene_x_a      : list of T audio tensors
               scene_x_b      : list of T text tensors
               scene_len_a    : list of T length tensors
               scene_len_b    : list of T length tensors
               scene_labels   : (T,) label tensor
               scene_mask     : (T,) int tensor — Markov audio availability mask
                                1 = audio present, 0 = audio absent

        3. Per-timestep predictions are flattened across all utterances in
           the scene before being passed to EvalMetric, so training metrics
           remain comparable to the original utterance-level evaluation.

        4. SceneGRUWrapper.forward_two_pass() is called during training.
           For inference (server.inference), forward() is called directly
           with the appropriate mask.
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
        self.args        = args
        self.model       = model
        self.device      = device
        self.criterion   = criterion
        self.dataloader  = dataloader
        self.multilabel  = True if args.dataset == 'ptb-xl' else False

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
                    # Dataloader yields one scene per iteration.
                    # scene_x_a:   list[T] of (1, T_frames, D_audio)
                    # scene_x_b:   list[T] of (1, T_tokens, D_text)
                    # scene_len_a: list[T] of (1,)
                    # scene_len_b: list[T] of (1,)
                    # scene_labels:(T,)   — per-utterance labels
                    # scene_mask:  (T,)   — per-utterance Markov audio mask
                    (scene_x_a, scene_x_b,
                     scene_len_a, scene_len_b,
                     scene_labels, scene_mask) = batch_data

                    scene_labels = scene_labels.to(self.device)   # (T,)
                    scene_mask   = scene_mask.to(self.device)     # (T,)

                    # ── Two-pass forward ──────────────────────────────────
                    # Pass 1: stable (audio always present)
                    # Pass 2: masked (Markov audio availability)
                    # Both passes run on the SAME scene content.
                    # No dataset splitting needed.
                    preds_stable, preds_masked = self.model.forward_two_pass(
                        scene_x_a, scene_x_b,
                        scene_len_a, scene_len_b,
                        scene_mask,
                        self.device
                    )
                    # preds_stable: (T, num_classes)
                    # preds_masked: (T, num_classes)

                    # ── Loss ──────────────────────────────────────────────
                    # NLLLoss expects log-probabilities
                    log_stable = torch.log_softmax(preds_stable, dim=-1)  # (T, C)
                    log_masked = torch.log_softmax(preds_masked, dim=-1)  # (T, C)

                    loss_stable = self.criterion(log_stable, scene_labels)
                    loss_masked = self.criterion(log_masked, scene_labels)

                    # Equal weighting of both conditions.
                    # The stable pass teaches full fusion;
                    # the masked pass teaches absence robustness and reintegration.
                    loss = loss_stable + loss_masked

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
                # Use the masked pass predictions for training metrics
                # (more informative than stable since it reflects the harder task).
                # Flatten per-timestep predictions for EvalMetric compatibility.
                if self.args.modality == "multimodal":
                    if not self.multilabel:
                        self.eval.append_classification_results(
                            scene_labels,
                            log_masked,
                            loss
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