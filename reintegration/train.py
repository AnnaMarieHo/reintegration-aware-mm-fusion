"""
Train temporal missingness model for the MELD dataset.
Usage:

python -m reintegration.train \
  --dataset iemocap \
  --data_dir /mnt/c/Users/aymie/Documents/UK_projects/masters-proj/my_extensions/reintegration/output \
  --modality multimodal \
  --audio_feat mfcc --text_feat mobilebert \
  --fed_alg fed_avg --availability_process markov \
  --num_epochs 200 --local_epochs 1 --sample_rate 1.0 --batch_size 16 \
  --hid_size 128 --learning_rate 0.01 --en_att --att_name fuse_base \
  --eval_only \
  --ckpt_path "/mnt/c/Users/aymie/Documents/UK_projects/masters-proj/my_extensions/reintegration/output/log/fed_avg/iemocap/mfcc_mobilebert/fuse_base/hid128_le1_lr001_bs16_sr10_ep100/fold2/model.pt"

python -m my_extensions.reintegration.train \
    --dataset meld \
    --modality multimodal \
    --audio_feat mfcc     
    --text_feat mobilebert \
    --fed_alg fed_avg \
    --num_epochs 200 \
    --local_epochs 1 \
    --sample_rate 1.0 \
    --batch_size 16 \
    --hid_size 128 \
    --learning_rate 0.01 \
    --en_att \
    --att_name fuse_base \
    --availability_process markov
"""
import torch
import json
import random
import numpy as np
import torch.nn as nn
import argparse, logging
import torch.multiprocessing
import copy, time, shutil, sys, os, pdb, gc
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


from tqdm import tqdm
from pathlib import Path


from reintegration.constants import constants
from reintegration.trainers.server_trainer import Server
from reintegration.model.mm_models import SERClassifier, SceneGRUWrapper
from reintegration.dataloader.dataload_manager import DataloadManager

# from my_extensions.reintegration.trainers.fed_rs_trainer import ClientFedRS
from reintegration.trainers.fed_avg_trainer import ClientFedAvg
# from my_extensions.reintegration.trainers.scaffold_trainer import ClientScaffold

import sys
from pathlib import Path

# define logging console
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s', 
    level=logging.INFO, 
    datefmt='%Y-%m-%d %H:%M:%S'
)


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def parse_args():
    # read path config files
    path_conf = dict()
    with open(str(Path(os.path.realpath(__file__)).parents[0].joinpath('system.cfg'))) as f:
        for line in f:
            key, val = line.strip().split('=')
            path_conf[key] = val.replace("\"", "")

    # If default setting
    if path_conf["data_dir"] == ".":
        path_conf["data_dir"] = str(Path(os.path.realpath(__file__)).parents[0].joinpath('MELD.Raw'))
    if path_conf["output_dir"] == ".":
        path_conf["output_dir"] = str(Path(os.path.realpath(__file__)).parents[0].joinpath('output'))

    parser = argparse.ArgumentParser(description='FedMultimoda experiments')
    parser.add_argument(
        '--data_dir', 
        default=path_conf["output_dir"],
        type=str, 
        help='output feature directory'
    )
    
    parser.add_argument(
        '--audio_feat', 
        default='mfcc',
        type=str,
        help="audio feature name",
    )
    
    parser.add_argument(
        '--text_feat', 
        default='mobilebert',
        type=str,
        help="text embedding feature name",
    )
    
    parser.add_argument(
        '--att', 
        type=bool, 
        default=False,
        help='self attention applied or not'
    )
    
    parser.add_argument(
        "--en_att",
        dest='att',
        action='store_true',
        help="enable self-attention"
    )

    parser.add_argument(
        '--att_name',
        type=str, 
        default='multihead',
        help='attention name'
    )
    
    parser.add_argument(
        '--learning_rate', 
        default=0.01,
        type=float,
        help="learning rate",
    )

    parser.add_argument(
        '--global_learning_rate', 
        default=0.05,
        type=float,
        help="learning rate",
    )

    parser.add_argument(
        '--mu',
        type=float, 
        default=0.001,
        help='Fed prox term'
    )
    
    parser.add_argument(
        '--sample_rate', 
        default=0.1,
        type=float,
        help="client sample rate",
    )
    
    parser.add_argument(
        '--num_epochs', 
        default=300,
        type=int,
        help="total training rounds",
    )

    parser.add_argument(
        '--test_frequency', 
        default=1,
        type=int,
        help="perform test frequency",
    )
    
    parser.add_argument(
        '--local_epochs', 
        default=1,
        type=int,
        help="local epochs",
    )
    
    parser.add_argument(
        '--optimizer', 
        default='sgd',
        type=str,
        help="optimizer",
    )
    
    parser.add_argument(
        '--fed_alg', 
        default='fed_avg',
        type=str,
        help="federated learning aggregation algorithm",
    )
    
    parser.add_argument(
        '--batch_size',
        default=16,
        type=int,
        help="training batch size",
    )
    
    parser.add_argument(
        '--hid_size',
        type=int, 
        default=64,
        help='RNN hidden size dim'
    )

    
    parser.add_argument(
        "--missing_modality",
        type=bool, 
        default=False,
        help="missing modality simulation",
    )
    
    parser.add_argument(
        "--en_missing_modality",
        dest='missing_modality',
        action='store_true',
        help="enable missing modality simulation",
    )
    
    parser.add_argument(
        "--missing_modailty_rate",
        type=float, 
        default=0.5,
        help='missing rate for modality; 0.9 means 90%% missing'
    )
    
    parser.add_argument(
        "--missing_label",
        type=bool, 
        default=False,
        help="missing label simulation",
    )
    
    parser.add_argument(
        "--en_missing_label",
        dest='missing_label',
        action='store_true',
        help="enable missing label simulation",
    )
    
    parser.add_argument(
        "--missing_label_rate",
        type=float, 
        default=0.5,
        help='missing rate for modality; 0.9 means 90%% missing'
    )
    
    parser.add_argument(
        '--label_nosiy', 
        type=bool, 
        default=False,
        help='clean label or nosiy label'
    )
    
    parser.add_argument(
        "--en_label_nosiy",
        dest='label_nosiy',
        action='store_true',
        help="enable label noise simulation",
    )

    parser.add_argument(
        '--label_nosiy_level', 
        type=float, 
        default=0.1,
        help='nosiy level for labels; 0.9 means 90% wrong'
    )
    
    parser.add_argument(
        '--modality', 
        type=str, 
        default='multimodal',
        help='modality type'
    )
    
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="meld",
        help='data set name'
    )
    #------------------------------------------------------------------------------------------------
    ## ADDED FOR REINTEGRATION EXPERIMENTS
    parser.add_argument(
        "--availability_process",
        type=str,
        # default="bernoulli",
        default="markov",
        choices=("bernoulli", "markov"),
        help="Availability process: bernoulli (per-client whole-sequence) or markov (per-timestep from sidecar).",
    )
    parser.add_argument(
        "--availability_sidecar_path",
        type=str,
        default=None,
        help="Path to precomputed availability .pkl for markov; required when availability_process=markov.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to model.pt for reintegration eval (e.g. best epoch 44). Overrides result_path/model.pt when set.",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip training; load checkpoint and run reintegration eval only (use with --ckpt_path).",
    )
    #------------------------------------------------------------------------------------------------
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    # argument parser
    args = parse_args()

    # data manager
    dm = DataloadManager(args)
    dm.get_text_feat_path()
    # find device
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available(): print('GPU available, use GPU')
    save_result_dict = dict()

    if args.fed_alg in ['fed_avg', 'fed_prox', 'fed_opt']:
        Client = ClientFedAvg
    # elif args.fed_alg in ['scaffold']:
    #     Client = ClientScaffold
    # elif args.fed_alg in ['fed_rs']:
        # Client = ClientFedRS

    # load simulation feature
    # Load partition so we have scene structure for label dist and dataloaders
    partition_path = Path(args.data_dir).joinpath('partition', args.dataset, 'partition.json')
    with open(str(partition_path)) as f:
        partition = json.load(f)

    # load client ids
    dm.get_client_ids()
    # set dataloaders
    dataloader_dict = dict()
    logging.info('Reading Data')

    for client_id in tqdm(dm.client_ids):
        audio_feat_dict = dm.load_audio_feat(client_id=client_id)
        # When running audio-only sanity checks, skip real text features and
        # pass an empty dict; the scene dataloader will synthesize zero text.
        if args.modality == 'audio_only':
            text_feat_dict = {}
        else:
            text_feat_dict  = dm.load_text_feat(client_id=client_id)

        # scenes for this client/split from partition (nested: scenes → utterances)
        scenes = partition[str(client_id)]

        dm.get_label_dist(scenes, client_id)

        shuffle    = client_id not in ['dev', 'test']
        # apply_mask=True for all splits so that dev/test dataloaders carry Markov
        # masks. Training ignores scene_mask (stable-only, Phase 1). Dev/test need
        # the masks so run_reintegration_eval() can locate reintegration events
        # (mask[t-1]==0, mask[t]==1) and run the stable vs masked contrast.
        # server.inference() uses the stable condition for UAR tracking throughout.
        apply_mask = True

        dataloader_dict[client_id] = dm.set_scene_dataloader(
            scenes          = scenes,
            audio_feat_dict = audio_feat_dict,
            text_feat_dict  = text_feat_dict,
            default_feat_shape_a = np.array([1000, constants.feature_len_dict["mfcc"]]),
            default_feat_shape_b = np.array([10,   constants.feature_len_dict["mobilebert"]]),
            p_stay_absent   = 0.7,
            p_stay_present  = 0.75,
            shuffle         = shuffle,
            apply_mask      = apply_mask,
        )
        # All-zeros audio ablation: same test scenes with p_stay_absent=1.0 so audio
        # never transitions to present. run_reintegration_eval gives preds_stable vs
        # preds_masked (all-zeros) and delta_uar = uar_stable - uar_masked.
        if client_id == 'test':
            dataloader_dict['test_all_zeros_audio'] = dm.set_scene_dataloader(
                scenes          = scenes,
                audio_feat_dict = audio_feat_dict,
                text_feat_dict  = text_feat_dict,
                default_feat_shape_a = np.array([1000, constants.feature_len_dict["mfcc"]]),
                default_feat_shape_b = np.array([10,   constants.feature_len_dict["mobilebert"]]),
                p_stay_absent   = 1.0,
                p_stay_present  = 0.75,
                shuffle         = False,
                apply_mask      = apply_mask,
            )
        
    # pdb.set_trace()
    # We perform 5 fold experiments with 5 seeds
    # for fold_idx in range(1, 6):
    for fold_idx in range(1, 6):
        # number of clients
        client_ids = [client_id for client_id in dm.client_ids if client_id not in ['dev', 'test']]
        num_of_clients = len(client_ids)
        # set seeds
        set_seed(8*fold_idx)
        # loss function
        criterion = nn.NLLLoss().to(device)
        # Define the model
        # SERClassifier: utterance-level encoder (intra-utterance features)
        # SceneGRUWrapper: cross-utterance GRU wrapper
        # The scene GRU hidden state carries absence history across utterances,
        # which is the mechanism through which reintegration effects manifest.
        # Multimodal vs audio-only configuration. For audio-only sanity checks,
        # we keep the same classifier head size but feed zeros in place of
        # text embeddings inside SERClassifier.
        if args.modality == 'audio_only':
            utterance_encoder = SERClassifier(
                num_classes=constants.num_class_dict[args.dataset],
                audio_input_dim=constants.feature_len_dict[args.audio_feat],
                text_input_dim=0,
                d_hid=args.hid_size,
                en_att=args.att,
                att_name=args.att_name,
            )
        else:
            utterance_encoder = SERClassifier(
                num_classes=constants.num_class_dict[args.dataset],
                audio_input_dim=constants.feature_len_dict[args.audio_feat],
                text_input_dim=constants.feature_len_dict[args.text_feat],
                d_hid=args.hid_size,
                en_att=args.att,
                att_name=args.att_name,
            )
        global_model = SceneGRUWrapper(
            utterance_encoder=utterance_encoder,
            num_classes=constants.num_class_dict[args.dataset],
            d_hid=args.hid_size,
        )
        global_model = global_model.to(device)

        # initialize server
        server = Server(
            args, 
            global_model, 
            device=device, 
            criterion=criterion,
            client_ids=client_ids
        )

        server.initialize_log(fold_idx)
        server.sample_clients(
            num_of_clients, 
            sample_rate=args.sample_rate
        )

        # save json path
        save_json_path = Path(os.path.realpath(__file__)).parents[2].joinpath(
            'result', 
            args.fed_alg,
            args.dataset,
            server.feature,
            server.att,
            server.model_setting_str
        )
        Path.mkdir(
            save_json_path, 
            parents=True, 
            exist_ok=True
        )

        server.save_json_file(
            dm.label_dist_dict, 
            save_json_path.joinpath('label.json')
        )
        
        # set seeds again
        set_seed(8*fold_idx)

        if args.eval_only:
            if not getattr(args, 'ckpt_path', None):
                raise ValueError("--eval_only requires --ckpt_path (path to model.pt).")
            server.result_path = Path(args.ckpt_path).parent
            save_result_dict[f'fold{fold_idx}'] = {}

        # Training steps (skipped when --eval_only)
        for epoch in range(0 if args.eval_only else int(args.num_epochs)):
            # define list varibles that saves the weights, loss, num_sample, etc.
            server.initialize_epoch_updates(epoch)
            # 1. Local training, return weights in fed_avg, return gradients in fed_sgd
            skip_client_ids = list()
            for idx in server.clients_list[epoch]:
                # Local training
                client_id = client_ids[idx]
                dataloader = dataloader_dict[client_id]
                if dataloader is None:
                    skip_client_ids.append(client_id)
                    continue
                
                # initialize client object
                client = Client(
                    args, 
                    device, 
                    criterion, 
                    dataloader, 
                    model=copy.deepcopy(server.global_model),
                    label_dict=dm.label_dist_dict[client_id],
                    num_class=constants.num_class_dict[args.dataset]
                )

                if args.fed_alg == 'scaffold':
                    client.set_control(
                        server_control=copy.deepcopy(server.server_control), 
                        client_control=copy.deepcopy(server.client_controls[client_id])
                    )
                    client.update_weights()

                    # server append updates
                    server.set_client_control(client_id, copy.deepcopy(client.client_control))
                    server.save_train_updates(
                        copy.deepcopy(client.get_parameters()), 
                        client.result['sample'], 
                        client.result,
                        delta_control=copy.deepcopy(client.delta_control)
                    )
                else:
                    client.update_weights()
                    # server append updates
                    server.save_train_updates(
                        copy.deepcopy(client.get_parameters()), 
                        client.result['sample'], 
                        client.result
                    )
                del client
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # logging skip client
            logging.info(f'Client Round: {epoch}, Skip client {skip_client_ids}')

            # 2. aggregate, load new global weights
            if len(server.num_samples_list) == 0: continue
            server.average_weights()
            logging.info('---------------------------------------------------------')
            server.log_classification_result(
                data_split='train', 
                metric='uar'
            )
            if epoch % args.test_frequency == 0:
                with torch.no_grad():
                    # 3. Perform the validation on dev set
                    server.inference(dataloader_dict['dev'])
                    server.result_dict[epoch]['dev'] = server.result
                    server.log_classification_result(
                        data_split='dev', 
                        metric='uar'
                    )

                    # 4. Perform the test on holdout set
                    server.inference(dataloader_dict['test'])
                    server.result_dict[epoch]['test'] = server.result
                    server.log_classification_result(
                        data_split='test', 
                        metric='uar'
                    )
                
                logging.info('---------------------------------------------------------')
                server.log_epoch_result(
                    metric='uar'
                )
            logging.info('---------------------------------------------------------')

        # Performance save code (skip when eval_only; already set to {} above)
        if not args.eval_only:
            save_result_dict[f'fold{fold_idx}'] = server.summarize_dict_results()

        # Free the per-epoch result accumulator before reintegration eval.
        server.result_dict.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Free the per-epoch result accumulator — it has served its purpose
        # (best checkpoint already saved to disk by log_epoch_result).
        # Holding all 200 rounds of train/dev/test result dicts in memory
        # through the reintegration eval is unnecessary.
        server.result_dict.clear()

        #------------------------------------------------------------------------------------------------
        ## REINTEGRATION EVAL — Phase 1 primary result
        # Runs once after FL training completes, on the best checkpoint
        # (selected by dev UAR). The model was trained stable-only and has
        # never seen audio absence. Any positive mean_delta at offset 0
        # is the unmitigated reintegration cost.
        _reint_ok = (
            getattr(args, 'availability_process', None) == 'markov'
            and dataloader_dict is not None
            and dataloader_dict.get('dev') is not None
            and dataloader_dict.get('test') is not None
        )
        logging.info(
            "Reintegration: availability_process=%s has_dev=%s has_test=%s => run=%s",
            getattr(args, 'availability_process', None),
            dataloader_dict.get('dev') is not None if dataloader_dict else False,
            dataloader_dict.get('test') is not None if dataloader_dict else False,
            _reint_ok,
        )
        if _reint_ok:
            try:
                # Load best checkpoint before running reintegration eval
                best_ckpt_path = Path(args.ckpt_path) if getattr(args, 'ckpt_path', None) else server.result_path.joinpath('model.pt')
                if best_ckpt_path.exists():
                    server.global_model.load_state_dict(
                        torch.load(str(best_ckpt_path), map_location=device)
                    )
                    logging.info("Reintegration eval: loaded best checkpoint from %s", best_ckpt_path)
                else:
                    logging.warning("Reintegration eval: no checkpoint found, using final model weights")

                with torch.no_grad():
                    reint_dev  = server.run_reintegration_eval(dataloader_dict['dev'])
                    reint_test = server.run_reintegration_eval(dataloader_dict['test'])
                    reint_test_all_zeros = server.run_reintegration_eval(
                        dataloader_dict['test_all_zeros_audio']
                    )
                save_result_dict[f'fold{fold_idx}']['reintegration_dev']  = reint_dev
                save_result_dict[f'fold{fold_idx}']['reintegration_test'] = reint_test
                save_result_dict[f'fold{fold_idx}']['reintegration_test_all_zeros_audio'] = reint_test_all_zeros
                logging.info(
                    "Reintegration dev:  mean_delta=%.4f, n_events=%d, "
                    "UAR_stable=%.2f%%, UAR_masked=%.2f%%",
                    reint_dev['mean_delta'], reint_dev['n_reint_events'],
                    reint_dev['uar_stable'], reint_dev['uar_masked']
                )
                dev_curve = reint_dev.get('mean_delta_by_offset', {})
                if dev_curve:
                    curve_str = ', '.join(
                        f'+{k}:{v:.4f}' for k, v in sorted(dev_curve.items())
                    )
                    logging.info("Dev  recovery curve: %s", curve_str)

                logging.info(
                    "Reintegration test: mean_delta=%.4f, n_events=%d, "
                    "UAR_stable=%.2f%%, UAR_masked=%.2f%%",
                    reint_test['mean_delta'], reint_test['n_reint_events'],
                    reint_test['uar_stable'], reint_test['uar_masked']
                )
                logging.info(
                    "Reintegration test (all-zeros audio ablation): n_events=%d, "
                    "UAR_stable=%.2f%%, UAR_masked=%.2f%%, delta_uar=%.2f%%",
                    reint_test_all_zeros['n_reint_events'],
                    reint_test_all_zeros['uar_stable'], reint_test_all_zeros['uar_masked'],
                    reint_test_all_zeros['delta_uar']
                )
                test_curve = reint_test.get('mean_delta_by_offset', {})
                if test_curve:
                    curve_str = ', '.join(
                        f'+{k}:{v:.4f}' for k, v in sorted(test_curve.items())
                    )
                    logging.info("Test recovery curve: %s", curve_str)
            except Exception as e:
                logging.exception("Reintegration eval failed: %s", e)
                print("Reintegration eval FAILED:", e)
        else:
            logging.info("Reintegration: skipped (need markov + dev + test loaders).")
            print("Reintegration: skipped (need markov + dev + test loaders).")
        #------------------------------------------------------------------------------------------------
        # output to results
        server.save_json_file(
            save_result_dict, 
            save_json_path.joinpath('result.json')
        )

    # Calculate the average of the 5-fold experiments
    save_result_dict['average'] = dict()
    for metric in ['uar', 'acc', 'top5_acc']:
        result_list = list()
        for key in save_result_dict:
            if metric not in save_result_dict[key]: continue
            result_list.append(save_result_dict[key][metric])
        save_result_dict['average'][metric] = np.nanmean(result_list)
    
    # dump the dictionary
    server.save_json_file(
        save_result_dict, 
        save_json_path.joinpath('result.json')
    )
