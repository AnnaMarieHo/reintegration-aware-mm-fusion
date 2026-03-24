"""
EvalMetric: accumulates per-batch predictions and computes epoch-level metrics.

classification_summary() returns:
    loss     : float  — mean NLLLoss
    acc      : float  — top-1 accuracy (%)
    top5_acc : float  — top-5 accuracy (%) [capped at num_classes]
    uar      : float  — unweighted average recall (%) = macro recall
    f1       : float  — macro F1 (%)
    sample   : int    — total number of predictions accumulated

These keys are all accessed by server_trainer.py and train.py.
"""

import torch
import numpy as np
from sklearn.metrics import recall_score, f1_score


class EvalMetric:
    def __init__(self, multilabel: bool = False):
        self.multilabel = multilabel
        self._reset()

    def _reset(self):
        self.all_labels  = []   # list of int labels
        self.all_preds   = []   # list of int top-1 predictions
        self.all_logits  = []   # list of 1-D np arrays (num_classes,)
        self.losses      = []   # list of scalar losses

    """
    Classification (single-label)
    """

    def append_classification_results(
        self,
        labels,    # (B,) or (T,) LongTensor
        logits,    # (B, C) or (T, C) log-probability tensor
        loss,      # scalar tensor or float
    ):
        """
        Accumulate predictions for one batch/scene.
        Works for both (B, C) utterance batches and (T, C) scene outputs.
        """
        if isinstance(loss, torch.Tensor):
            loss = loss.item()
        self.losses.append(loss)

        labels = labels.detach().cpu()
        logits = logits.detach().cpu()

        preds = logits.argmax(dim=-1)   # (B,) or (T,)

        self.all_labels.extend(labels.tolist())
        self.all_preds.extend(preds.tolist())
        self.all_logits.append(logits.numpy())   # (B, C) or (T, C)

    def classification_summary(self) -> dict:
        """Compute and return epoch-level metrics dict."""
        if len(self.all_labels) == 0:
            return {'loss': 0.0, 'acc': 0.0, 'top5_acc': 0.0,
                    'uar': 0.0, 'f1': 0.0, 'sample': 0}

        labels = np.array(self.all_labels)
        preds  = np.array(self.all_preds)
        logits = np.concatenate(self.all_logits, axis=0)   # (N, C)

        n      = len(labels)
        C      = logits.shape[1]

        # Top-1 accuracy
        acc = float((preds == labels).mean()) * 100.0

        # Top-5 accuracy (capped at num_classes)
        k = min(5, C)
        top5_indices = np.argsort(logits, axis=1)[:, -k:]
        top5_correct = np.array([labels[i] in top5_indices[i] for i in range(n)])
        top5_acc = float(top5_correct.mean()) * 100.0

        # UAR — unweighted average recall = macro recall
        uar = recall_score(labels, preds, average='macro', zero_division=0) * 100.0

        # Macro F1
        f1 = f1_score(labels, preds, average='macro', zero_division=0) * 100.0

        # Mean loss
        loss = float(np.mean(self.losses))

        self._reset()

        return {
            'loss':     loss,
            'acc':      acc,
            'top5_acc': top5_acc,
            'uar':      uar,
            'f1':       f1,
            'sample':   n,
        }

    """
    Multilabel (ptb-xl only kept for completeness, not used for MELD)
    """

    def append_multilabel_results(self, labels, logits, loss):
        if isinstance(loss, torch.Tensor):
            loss = loss.item()
        self.losses.append(loss)
        self.all_labels.append(labels.detach().cpu().numpy())
        self.all_preds.append((logits.detach().cpu().numpy() > 0.5).astype(int))

    def multilabel_summary(self) -> dict:
        if len(self.all_labels) == 0:
            return {'loss': 0.0, 'acc': 0.0, 'macro_f': 0.0, 'sample': 0}

        labels = np.concatenate(self.all_labels, axis=0)
        preds  = np.concatenate(self.all_preds,  axis=0)

        acc     = float((preds == labels).all(axis=1).mean()) * 100.0
        macro_f = f1_score(labels, preds, average='macro', zero_division=0) * 100.0
        loss    = float(np.mean(self.losses))
        n       = labels.shape[0]

        self._reset()

        return {'loss': loss, 'acc': acc, 'macro_f': macro_f, 'sample': n}