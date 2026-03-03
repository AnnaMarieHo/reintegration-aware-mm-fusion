
import collections
import numpy as np
import pandas as pd
import copy, pdb, time, warnings, torch


from torch import nn
from torch.utils import data
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, recall_score

# import optimizer
from .optimizer import FedProxOptimizer

warnings.filterwarnings('ignore')
from .evaluation import EvalMetric


class ClientFedAvg(object):
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
        self.args = args
        self.model = model
        self.device = device
        self.criterion = criterion
        self.dataloader = dataloader
        self.multilabel = True if args.dataset == 'ptb-xl' else False
        
    def get_parameters(self):
        # Return model parameters
        return self.model.state_dict()
    
    def get_model_result(self):
        # Return model results
        return self.result
    
    def get_test_true(self):
        # Return test labels
        return self.test_true
    
    def get_test_pred(self):
        # Return test predictions
        return self.test_pred
    
    def get_train_groundtruth(self):
        # Return groundtruth used for training
        return self.train_groundtruth

    def update_weights(self):
        # Set mode to train model
        self.model.train()

        # initialize eval
        self.eval = EvalMetric(self.multilabel)
        
        # optimizer
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
            
        # last global model
        last_global_model = copy.deepcopy(self.model)
        
        for iter in range(int(self.args.local_epochs)):
            for batch_idx, batch_data in enumerate(self.dataloader):
                if self.args.dataset == 'extrasensory' and batch_idx > 20: continue
                self.model.zero_grad()
                optimizer.zero_grad()
                aux_logits = None  # set only by multimodal forward; ensures no stale aux loss in unimodal
                if self.args.modality == "multimodal":
                    #------------------------------------------------------------------------------------------------
                    ## ADDED FOR REINTEGRATION EXPERIMENTS
                    # x_a, x_b, l_a, l_b, y = batch_data
                    x_a, x_b, l_a, l_b, y, mask_a, mask_b = batch_data
                    x_a, x_b, y = x_a.to(self.device), x_b.to(self.device), y.to(self.device)
                    #------------------------------------------------------------------------------------------------
                    l_a, l_b = l_a.to(self.device), l_b.to(self.device)
                    mask_a = mask_a.to(self.device) if mask_a is not None else None
                    mask_b = mask_b.to(self.device) if mask_b is not None else None

                    #------------------------------------------------------------------------------------------------
                    # ## ADDED FOR REINTEGRATION EXPERIMENTS
                    # m = mask_b[:, :mask_b.shape[1]].detach().cpu().numpy().astype(bool)
                    # lens = l_b.detach().cpu().numpy()

                    # flips = []
                    # reintegration = []
                    # means = []
                    # for i in range(m.shape[0]):
                    #     mi = m[i, :lens[i]]
                    #     flips.append((mi[1:] != mi[:-1]).sum())
                    #     reintegration.append(((~mi[:-1]) & mi[1:]).sum())
                    #     means.append(mi.mean() if len(mi) else 0.0)

                    # print("flips(avg/min/max):", np.mean(flips), np.min(flips), np.max(flips))
                    # print("reintegration(avg/min/max):", np.mean(reintegration), np.min(reintegration), np.max(reintegration))
                    # print("mean(avg/min/max):", np.mean(means), np.min(means), np.max(means))
                    # #------------------------------------------------------------------------------------------------



                    #------------------------------------------------------------------------------------------------
                    ## ADDED FOR REINTEGRATION EXPERIMENTS
                    # forward
                    outputs, x_mm, aux_logits = self.model(
                        x_a.float(), x_b.float(), l_a, l_b,
                        mask_a=mask_a, mask_b=mask_b,
                        return_aux=True,
                    )
                    # print(f'aux_logits: {aux_logits.shape}')
                    
                    #------------------------------------------------------------------------------------------------
                    # # forward
                    # outputs, _ = self.model(
                    #     x_a.float(), x_b.float(), l_a, l_b,
                    #     mask_a=mask_a, mask_b=mask_b,
                    # )
                else:
                    x, l, y = batch_data
                    x, l, y = x.to(self.device), l.to(self.device), y.to(self.device)
                    
                    # forward
                    outputs, _ = self.model(
                        x.float(), l
                    )
                
                if not self.multilabel: 
                    outputs = torch.log_softmax(outputs, dim=1)
                    
                # backward
                loss = self.criterion(outputs, y)
                # Aux loss for reintegration (per-timestep): train aux_head (must run when multimodal + return_aux)
                if not self.multilabel and aux_logits is not None:
                    B, T_aux, C = aux_logits.shape
                    aux_labels = y.unsqueeze(1).expand(B, T_aux).reshape(-1)
                    aux_loss = self.criterion(
                        torch.log_softmax(aux_logits, dim=-1).reshape(B * T_aux, C),
                        aux_labels,
                    )
                    loss = loss + 0.3 * aux_loss
                # backward
                loss.backward()
                
                # clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    10.0
                )
                optimizer.step()
                
                # save results
                if not self.multilabel: 
                    self.eval.append_classification_results(
                        y, 
                        outputs, 
                        loss
                    )
                else:
                    self.eval.append_multilabel_results(
                        y, 
                        outputs, 
                        loss
                    )
                
        # epoch train results
        if not self.multilabel:
            self.result = self.eval.classification_summary()
        else:
            self.result = self.eval.multilabel_summary()
