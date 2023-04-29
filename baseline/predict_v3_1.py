import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
from model.modules import MLP, CGBlock, MCGBlock, HistoryEncoder
from model.multipathpp import MultiPathPP
from model.data import get_dataloader, dict_to_cuda, normalize
from model.losses import pytorch_neg_multi_log_likelihood_batch, nll_with_covariances
from prerender.utils.utils import data_to_numpy, get_config
import subprocess
from matplotlib import pyplot as plt
import os
import glob
import sys
import random

from google.protobuf import text_format
from waymo_open_dataset.protos import motion_metrics_pb2
from waymo_open_dataset.metrics.ops import py_metrics_ops

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class MotionMetrics():
  """Wrapper for motion metrics computation."""

  def __init__(self, config):
    super().__init__()
    self._prediction_trajectory = []
    self._prediction_score = []
    self._ground_truth_trajectory = []
    self._ground_truth_is_valid = []
    self._prediction_ground_truth_indices = []
    self._prediction_ground_truth_indices_mask = []
    self._object_type = []
    self._metrics_config = config

  def update_state(self, prediction_trajectory, prediction_score,
                   ground_truth_trajectory, ground_truth_is_valid,
                   prediction_ground_truth_indices,
                   prediction_ground_truth_indices_mask, object_type):
    self._prediction_trajectory.append(prediction_trajectory)
    self._prediction_score.append(prediction_score)
    self._ground_truth_trajectory.append(ground_truth_trajectory)
    self._ground_truth_is_valid.append(ground_truth_is_valid)
    self._prediction_ground_truth_indices.append(
        prediction_ground_truth_indices)
    self._prediction_ground_truth_indices_mask.append(
        prediction_ground_truth_indices_mask)
    self._object_type.append(object_type)

  def result(self):
    # [batch_size, num_preds, 1, 1, steps, 2].
    # The ones indicate top_k = 1, num_agents_per_joint_prediction = 1.
    prediction_trajectory = np.vstack(self._prediction_trajectory)
    # print(prediction_trajectory.shape)
    # [batch_size, num_preds, 1].
    prediction_score = np.vstack(self._prediction_score)   #np.concatenate((self._prediction_score, 0))
    # [batch_size, num_agents, gt_steps, 7].
    ground_truth_trajectory = np.vstack(self._ground_truth_trajectory)     #np.concatenate((self._ground_truth_trajectory, 0))
    # [batch_size, num_agents, gt_steps].
    ground_truth_is_valid = np.vstack(self._ground_truth_is_valid)        #np.concatenate((self._ground_truth_is_valid, 0))
    # [batch_size, num_preds, 1].
    prediction_ground_truth_indices = np.vstack(self._prediction_ground_truth_indices) #np.concatenate((
        #self._prediction_ground_truth_indices, 0))
    # [batch_size, num_preds, 1].
    prediction_ground_truth_indices_mask = np.vstack(self._prediction_ground_truth_indices_mask) #np.concatenate((
        #self._prediction_ground_truth_indices_mask, 0))
    # [batch_size, num_agents].
    object_type = np.vstack(self._object_type).astype(int)#, dtype = int) #np.concatenate((self._object_type, 0), dtype=int)

    # We are predicting more steps than needed by the eval code. Subsample.
    interval = (
        self._metrics_config.track_steps_per_second //
        self._metrics_config.prediction_steps_per_second)
    prediction_trajectory = prediction_trajectory[...,
                                                  (interval - 1)::interval, :]

    return py_metrics_ops.motion_metrics(
        config=self._metrics_config.SerializeToString(),
        prediction_trajectory=prediction_trajectory,
        prediction_score=prediction_score,
        ground_truth_trajectory=ground_truth_trajectory,
        ground_truth_is_valid=ground_truth_is_valid,
        prediction_ground_truth_indices=prediction_ground_truth_indices,
        prediction_ground_truth_indices_mask=prediction_ground_truth_indices_mask,
        object_type=object_type)

def get_git_revision_short_hash():
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()

def get_last_file(path):
    list_of_files = glob.glob(f'{path}/*')
    if len(list_of_files) == 0:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def _default_metrics_config():
    config = motion_metrics_pb2.MotionMetricsConfig()
    config_text = """
    track_steps_per_second: 10
    prediction_steps_per_second: 2
    track_history_samples: 10
    track_future_samples: 80
    speed_lower_bound: 1.4
    speed_upper_bound: 11.0
    speed_scale_lower: 0.5
    speed_scale_upper: 1.0
    step_configurations {
        measurement_step: 5
        lateral_miss_threshold: 1.0
        longitudinal_miss_threshold: 2.0
    }
    step_configurations {
        measurement_step: 9
        lateral_miss_threshold: 1.8
        longitudinal_miss_threshold: 3.6
    }
    step_configurations {
        measurement_step: 15
        lateral_miss_threshold: 3.0
        longitudinal_miss_threshold: 6.0
    }
    max_predictions: 6
    """
    text_format.Parse(config_text, config)
    return config

config = get_config(sys.argv[1])
alias = sys.argv[1].split("/")[-1].split(".")[0]
try:
    models_path = "../baseline_models"
    os.mkdir(models_path)
except:
    pass
last_checkpoint = get_last_file(models_path)

val_dataloader = get_dataloader(config["val"]["data_config"])
test_dataloader = get_dataloader(config["test"]["data_config"])

model = MultiPathPP(config["model"])
model.cuda()

num_steps = 0
if last_checkpoint is not None:
    model.load_state_dict(torch.load(last_checkpoint)["model_state_dict"])
    num_steps = torch.load(last_checkpoint)["num_steps"]
    print("LOADED ", last_checkpoint)

this_num_steps = 0
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("N PARAMS =", params)

metrics_config = _default_metrics_config()
motion_metrics = MotionMetrics(metrics_config)

model.eval()
with torch.no_grad():
    
    losses = []
    min_ades = []
    
    first_batch = True

    # run the metrics on validation data

    # for val_data in tqdm(val_dataloader):
    #     if config["train"]["normalize"]:
    #         data = normalize(data, config)
        
    #     dict_to_cuda(data)

    for data in tqdm(test_dataloader):
        if config["train"]["normalize"]:
            data = normalize(data, config)
        dict_to_cuda(data)

        probas, coordinates, cov_mat, loss_coeff = model(data, num_steps)

        # coordinates shape ([batch_size, num_agents, steps, 2]) --> ([42, 6, 80, 2])
        # probabas shape () --> ([42, 6])
        # covariance matrix () --> ([42, 6, 80, 2, 2])
        pred_trajectory = coordinates

        prediction_start = metrics_config.track_history_samples + 1
        
        # batch_size = 42
        # num_agents = 6
        # steps = 80

        
        #### NEEDS TO BE FIXED
        # ([42, 6, 80, 7])
        # print(data["target/gt_future_states"].shape)   ## --> ([42, 80, 7])
        gt_trajectory = data["target/gt_future_states"][:, np.newaxis, :, :]
        gt_trajectory = gt_trajectory.repeat(1, 6, 1, 1)
        gt_targets = gt_trajectory[..., prediction_start:, :2]

        # [batch_size, num_agents, 1, 1, steps, 2] --> ([42, 6, 1, 1, 80, 2])
        pred_trajectory = pred_trajectory[:, :, np.newaxis, np.newaxis].cpu()
        
        # TODO: Check this. for now, faking this score
        pred_score = np.ones(shape=pred_trajectory.shape[:3])

        batch_size = data['state/tracks_to_predict'].reshape(-1, 128).shape[0]
        num_samples = 128
        steps = 80

        # TODO: check after new prerenders
        object_type = data["target/agent_type"]
        if object_type.numel() < batch_size:
            suppl = torch.ones(batch_size - object_type.numel())
            object_type = torch.cat([object_type, suppl])

        # batch_size = data['state/tracks_to_predict'].shape[0]
        # num_samples = data['state/tracks_to_predict'].shape[1]
        # print(batch_size)
        # print(num_samples)

        # [batch_size, num_agents, steps]
        # gt_is_valid = np.ones(shape=(batch_size, num_samples, steps))
        gt_is_valid = data["target/future/valid"].permute(0, 2, 1).cpu()
        gt_is_valid = gt_is_valid.repeat(1, 6, 1)

        pred_gt_indices = np.arange(data['state/tracks_to_predict'].reshape(-1, 128).shape[1], dtype=np.int64)
        # ([batch_size, num_agents, 1])  --> ([42, 6, 1])
        pred_gt_indices = np.tile(pred_gt_indices[np.newaxis, :, np.newaxis], (batch_size, 1, 1))

        # [batch_size, num_agents, 1]  --> ([42, 6, 1])
        pred_gt_indices_mask = data['state/tracks_to_predict'].reshape((-1, 128))[..., np.newaxis]
        # print(pred_gt_indices_mask.shape)

        # if config["test"]["normalize_output"]:
        #     coordinates = coordinates * 10. + torch.Tensor([1.4715e+01, 4.3008e-03]).cuda()

        # errors = np.abs(coordinates.permute(0, 2, 1, 3) - gt_traj)  # calculate absolute errors
        # mae = np.mean(errors)  # calculate mean of errors
        # print(f'MAE: {mae}')

        motion_metrics.update_state(pred_trajectory, pred_score, gt_trajectory,
                              gt_is_valid, pred_gt_indices,
                              pred_gt_indices_mask, object_type)

        # Compute minimum ADE across all predicted trajectories
        # min_ade = min(ade_list)
        # min_ades.append(min_ade)


print(motion_metrics.result())
