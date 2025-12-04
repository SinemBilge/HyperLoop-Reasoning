from torch import optim
import os
import torch
from datetime import datetime
from transformers import Adafactor
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
import sys
import math
from geoopt.optim import RiemannianAdam
from src.models import SoftPromptModel

# Get the current directory (train)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory (one level up from the current directory)
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to the system path
sys.path.append(parent_dir)

# Now you can import the config module
from config import BaseTrainingConfig

train_dir = current_dir
sys.path.append(train_dir)


from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(
        self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1.0, last_epoch=-1
    ):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [
                (self.eta_max - base_lr) * self.T_cur / self.T_up + base_lr
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr
                + (self.eta_max - base_lr)
                * (
                    1
                    + math.cos(
                        math.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up)
                    )
                )
                / 2
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(
                        math.log(
                            (epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult
                        )
                    )
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult**n - 1) / (
                        self.T_mult - 1
                    )
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr
def get_optimizer(parameters, trainer_config : BaseTrainingConfig):
    lr = trainer_config.learning_rate
    print(f"{parameters = }")
    print(f"Training with optimizer {trainer_config.optimizer} and Learning Rate {trainer_config.learning_rate}")
    if trainer_config.optimizer == 'Adam':
        optimizer = optim.Adam(parameters, lr= lr)
    elif  trainer_config.optimizer == 'AdamW':
        optimizer = optim.AdamW(parameters, lr=lr, weight_decay= trainer_config.optimizer_param)
    elif  trainer_config.optimizer == 'AdaFactor':
        optimizer = Adafactor(parameters, lr=lr, weight_decay= trainer_config.optimizer_param, relative_step=False, scale_parameter=False)
    elif  trainer_config.optimizer == 'Hyperbolic':
        optimizer = RiemannianAdam(parameters, lr=lr, weight_decay= trainer_config.optimizer_param)
    else:
        raise ValueError(f"Unsupported optimizer: {trainer_config.optimizer}")
    return optimizer

def setup_directories(trainer_config : BaseTrainingConfig, t5_config):
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    optimizer = trainer_config.optimizer
    final_string = f"{current_time}_{optimizer}_{trainer_config.additional_log_info}"
    log_dir = os.path.join(trainer_config.log_dir, final_string)
    model_dir = os.path.join(trainer_config.model_save_path, final_string) 
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)  
    return log_dir, model_dir

def load_model_checkpoint(model : nn.Module,
                          checkpoint_path,
                          device = 'cuda' if torch.cuda.is_available() else 'cpu',
                          with_model_state_dict = True,
                          gpu_parallelization = False
                          ):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if gpu_parallelization:
        new_checkpoint = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
        missing, unexpected = model.load_state_dict(new_checkpoint, strict=False)
        
        print(f"{missing = }")
        print(f"{unexpected = }")
        print(f"Loaded checkpoint from {checkpoint_path}")
        return model
    if with_model_state_dict:
        missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    print(f"{missing = }")
    print(f"{unexpected = }")
    print(f"Loaded checkpoint from {checkpoint_path}")
    return model
                    
def log_tensorboard(writer : SummaryWriter,
                    value : float,
                    idx,
                    phase,
                    eval_metric = 'loss'):
    if eval_metric not in ['loss', 'em', 'f1']:
        raise ValueError(f'Unsupported eval_metric: {eval_metric}. Supported are [loss, em, f1]')
    else:
        writer.add_scalar(f'{phase}/{eval_metric}', value, idx)
        
def load_soft_prompt_and_additional_layer(model : SoftPromptModel,
                    soft_prompt_path : str,
                    gpu_parallelization : bool,
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'):
    checkpoint = torch.load(soft_prompt_path, map_location=device) 
    soft_prompt = checkpoint['soft_prompt_state_dict']
    additional_layer = checkpoint['additional_linear_layer']

    if gpu_parallelization:
        model.module.soft_prompt = soft_prompt
        model.module.knit5.hyperbolic_layer.load_state_dict(additional_layer)
    else:
        model.soft_prompt = soft_prompt
        model.knit5.hyperbolic_layer.load_state_dict(additional_layer)

    model.to(device)

    print(f'Loading Soft Prompt and Additional Layer Checkpoint from {soft_prompt_path}')
def load_optimizer_and_start_epoch(optimizer_hyperbolic : optim.Optimizer,
                                   optimizer_soft_prompt : optim.Optimizer,
                                   checkpoint_path):
    checkpoint = torch.load(checkpoint_path) 
    if 'optimizer_state_dict' in checkpoint:
        optimizer_hyperbolic.load_state_dict(checkpoint['optimizer_hyperbolic_state_dict'])
        optimizer_soft_prompt.load_state_dict(checkpoint['optimizer_softprompt_state_dict'])
        print(f'Loading Optimizer Checkpoint from {checkpoint_path}')
        start_epoch = checkpoint['epoch']
    
        return optimizer_hyperbolic, optimizer_soft_prompt, start_epoch
    else:
        return optimizer_hyperbolic,optimizer_soft_prompt, 0
    
    
def geodesic_distance(u, v, c=1.0):
    norm_u = torch.norm(u, p=2, dim=-1)
    norm_v = torch.norm(v, p=2, dim=-1)
    numerator = torch.norm(u - v, p=2, dim=-1)**2
    denominator = (1 - c * norm_u**2) * (1 - c * norm_v**2)
    distance = torch.acosh(1 + 2 * c * numerator / denominator)
    return distance

def geodesic_regularization(soft_prompt_input, min_distance=1.0, c=1.0):
    loss = 0
    num_components = soft_prompt_input.size(1)
    for i in range(num_components):
        for j in range(i+1, num_components):
            distance = geodesic_distance(soft_prompt_input[:, i], soft_prompt_input[:, j], c)
            loss += torch.clamp(min_distance - distance, min=0).mean()
    return loss
    
    