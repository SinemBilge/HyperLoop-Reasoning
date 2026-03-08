import os
import sys
from typing import Union

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer

from src.config import Config
from src.eval import exact_match_score, f1_score
from src.utils.trainer_utils import *
from src.models import SoftPromptModel
from src.utils.trainer_utils import CosineAnnealingWarmUpRestarts


"""
Triggering Multi-Hop Reasoning for Question Answering
in Language Models using Soft Prompts and Random Walks: https://arxiv.org/pdf/2306.04009
"""




class SoftPromptTrainer:
    def __init__(self,
                 model: Union[SoftPromptModel, DDP],
                 tokenizer: AutoTokenizer,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 config: Config,
                 device: str = 'cpu',
                 checkpoint_path: str = None,
                 tboard_checkpoint_path: str = None,
                 validation_step: int = 1,
                 method: str = 'random_walk_training',
                 retrain: bool = False,
                 gpu_parallelization: bool = False,
                 rank: int = 0):
        # ----- Core state -----
        self.gpu_parallelization = gpu_parallelization
        self.rank = rank
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.tboard_checkpoint_path = tboard_checkpoint_path
        self.validation_step = validation_step

        # Move model to device and (optionally) wrap in DDP
        self.model = model.to(device)
        if gpu_parallelization:
            device_ids = None
            idx = device_index_from_str(device)
            if idx is not None:
                device_ids = [idx]  # single-process-per-GPU
            self.model = DDP(self.model, device_ids=device_ids, find_unused_parameters=False)

        # Always use self.net for attributes/params; use self.model(...) for forward
        self.net = unwrap_model(self.model)

        # Tokenizer / env
        self.tokenizer = tokenizer
        self.tokenizer.model_max_length = config.t5_model.tokenizer_max_length
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Data
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Configs
        self.config = config
        self.supported_methods = ['random_walk_training', 'parse_then_hop_training']
        if method not in self.supported_methods:
            raise ValueError(f"Unsupported phase: {method} Supported Phases are: {self.supported_methods}")
        self.method = method

        if self.method == 'random_walk_training':
            self.training_config = config.random_walk_training
        elif self.method == 'parse_then_hop_training':
            self.training_config = config.parse_then_hop_training

        # Training state
        self.start_epoch = 0
        self.patience = self.training_config.epochs
        self.best_loss = float('inf')
        self.best_em = 0.0
        self.early_stop_counter = 0
        self.best_model_path = None

        # Debug: hyperbolic layer learnable?
        print(f"Hyperbolic Layer learnable: "
              f"{all(p.requires_grad for p in self.net.knit5.additional_layer.parameters()) if self.net.knit5.additional_layer_type != 'identity' else False}")

        # ----- Optimizer (built once from self.net) -----
        optimizer_params = []
        if self.training_config.use_soft_prompt:
            optimizer_params.append({
                'params': self.net.soft_prompt,
                'lr': self.training_config.learning_rate
            })
        optimizer_params.append({
            'params': self.net.knit5.additional_layer.parameters(),
            'lr': config.single_hop_training.learning_rate
        })
        self.optimizer = get_optimizer(optimizer_params, self.training_config)

        # ----- Scheduler (optional) -----
        self.scheduler = None
        if self.training_config.use_scheduler:
            # Use your custom scheduler on the single optimizer
            self.scheduler = CosineAnnealingWarmUpRestarts(
                self.optimizer, T_0=150, T_mult=1, eta_max=0.001, T_up=10, gamma=0.5
            )
            print("Using Scheduler!")

        # ----- Curvature -----
        from math import log, exp
        c = 0.0
        hl = getattr(self.net.knit5, 'additional_layer', None)
        if hl is not None and hasattr(hl, 'manifold') and hasattr(hl.manifold, 'c'):
            c_val = hl.manifold.c.item()
            c = log(exp(c_val) + 1.0)
        self.training_config.curvature = c

        # ----- Directories / TB Writer -----
        self.log_dir, self.model_dir = setup_directories(self.training_config, self.config.t5_model)
        if self.tboard_checkpoint_path is not None:
            self.log_dir = tboard_checkpoint_path
            print(f"Continue writing to {tboard_checkpoint_path}")
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        if self.checkpoint_path is not None and retrain:
            load_soft_prompt_and_additional_layer(self.model, checkpoint_path, gpu_parallelization=gpu_parallelization)
            # If you previously saved optimizer state, load it here;
            # aligning with the new unified optimizer:
            _, _, self.start_epoch = load_optimizer_and_start_epoch(self.optimizer, None, checkpoint_path)

    # ---------- Logging ----------

    def log_tensorboard(self, value, idx, phase, eval_metric='loss'):
        if eval_metric not in ['loss', 'em', 'f1']:
            raise ValueError(f'Unsupported eval_metric: {eval_metric}. Supported are [loss, em, f1]')
        self.writer.add_scalar(f'{phase}/{eval_metric}', value, idx)

    # ---------- Train ----------

    def train(self, epochs: int):
        if self.start_epoch != 0:
            print(f'Starting training from epoch {self.start_epoch}')

        for epoch in range(self.start_epoch, epochs):
            # Freeze everything except hyperbolic layer (as per your original intent)
            self.net.knit5.eval()
            self.net.knit5.additional_layer.train()

            # DistributedSampler epoch bump (if present)
            if hasattr(self.train_dataloader.sampler, "set_epoch"):
                self.train_dataloader.sampler.set_epoch(epoch)

            progress_bar = tqdm(
                self.train_dataloader,
                leave=True,
                desc=f"Epoch {epoch} - Training - {self.method}",
                file=sys.stdout,
                dynamic_ncols=True
            )

            total_loss = 0.0

            for batch_idx, batch in enumerate(progress_bar):
                self.optimizer.zero_grad()

                input_batch, label_batch = batch
                inputs = self.tokenizer(
                    input_batch, padding=True, truncation=True, return_tensors='pt'
                )
                input_ids = inputs.input_ids.to(self.device)
                attention_mask = inputs.attention_mask.to(self.device)

                labels = self.tokenizer(
                    label_batch, padding=True, truncation=True, return_tensors='pt'
                )
                labels_input_ids = labels.input_ids.to(self.device)
                labels_input_ids[labels_input_ids == self.tokenizer.pad_token_id] = -100

                # Forward through (possibly) DDP-wrapped model
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels_input_ids
                )
                loss = outputs.loss

                loss.backward()
                self.optimizer.step()

                # Optional quick debug
                if batch_idx <= 5 and is_main_process():
                    print(f"{input_batch[0] = }")
                    print(f"{label_batch[0] = }")

                # Log local loss (rank-0 only to avoid spam)
                total_loss += loss.item()
                if is_main_process():
                    self.log_tensorboard(
                        loss.item(),
                        epoch * len(self.train_dataloader) + batch_idx,
                        'Training'
                    )

                progress_bar.set_description(
                    f"Epoch {epoch} - Training - {self.method} - Loss: {loss.item():.4f}"
                )

                # VRAM logging (rank-0)
                if is_main_process() and torch.cuda.is_available() and self.device.startswith("cuda"):
                    vram_allocated = torch.cuda.memory_allocated(torch.device(self.device)) / (1024 ** 2)
                    vram_reserved = torch.cuda.memory_reserved(torch.device(self.device)) / (1024 ** 2)
                    step_idx = epoch * len(self.train_dataloader) + batch_idx
                    self.writer.add_scalar('VRAM/Training/Allocated', vram_allocated, step_idx)
                    self.writer.add_scalar('VRAM/Training/Reserved', vram_reserved, step_idx)

                    # Curvature logging
                    hl = self.net.knit5.additional_layer
                    if isinstance(hl, nn.Sequential) and len(hl) > 1:
                        for layer_idx, layer in enumerate(hl):
                            if hasattr(layer, 'manifold') and hasattr(layer.manifold, 'c'):
                                self.writer.add_scalar(
                                    f'Training/Curvature_Layer_{layer_idx}',
                                    layer.manifold.c.item(),
                                    step_idx
                                )
                    else:
                        if hasattr(hl, 'manifold') and hasattr(hl.manifold, 'c'):
                            self.writer.add_scalar(
                                'Training/Curvature',
                                hl.manifold.c.item(),
                                step_idx
                            )

            # Epoch average (local)
            avg_loss_local = total_loss / max(1, len(self.train_dataloader))
            if is_main_process():
                print(f"Epoch {epoch}, Loss: {avg_loss_local:.4f}")

            # Validation / Early stopping
            if (self.val_dataloader is not None) and (epoch % self.validation_step == 0):
                if self.evaluate(epoch=epoch):
                    break  # Early Stopping

    # ---------- Evaluate ----------

    def evaluate(self, epoch: int):
        self.model.eval()

        # Local accumulators
        local_loss_sum = 0.0
        local_em_sum = 0.0
        local_f1_sum = 0.0
        local_count = 0

        progress_bar = tqdm(
            self.val_dataloader,
            leave=True,
            desc=f"Epoch {epoch} - Validation - {self.method}",
            file=sys.stdout,
            dynamic_ncols=True
        )

        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                input_batch, label_batch = batch

                # Inputs to device (BatchEncoding has .to)
                inputs = self.tokenizer(
                    input_batch, padding=True, truncation=True, return_tensors='pt'
                ).to(self.device)

                # Labels -> only use input_ids
                label_ids = self.tokenizer(
                    label_batch, padding=True, truncation=True, return_tensors='pt'
                )['input_ids'].to(self.device)

                outputs = self.model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    labels=label_ids
                )

                loss = outputs.loss
                batch_size = len(input_batch)

                local_loss_sum += loss.item() * batch_size

                # Greedy decode of logits
                predictions = torch.argmax(outputs.logits, dim=-1)
                decoded_predictions = [
                    self.tokenizer.decode(pred, skip_special_tokens=True)
                    for pred in predictions
                ]

                batch_em = sum(
                    1 if exact_match_score(pred, truth) else 0
                    for pred, truth in zip(decoded_predictions, label_batch)
                )
                batch_f1 = sum(
                    f1_score(pred, truth)[0]
                    for pred, truth in zip(decoded_predictions, label_batch)
                )

                local_em_sum += batch_em
                local_f1_sum += batch_f1
                local_count += batch_size

                progress_bar.set_description(
                    f"Epoch {epoch} - Validation - {self.method} - Loss: {loss.item():.4f}"
                )

                # Sample text logging
                if batch_idx <= 5 and is_main_process():
                    self.writer.add_text(
                        f'Validation/Prediction_vs_Label_{epoch}',
                        f'Prediction: {decoded_predictions[0]}\nLabel: {label_batch[0]}',
                        epoch
                    )

        # ----- Distributed reduction (sum) -----
        if self.gpu_parallelization and dist.is_available() and dist.is_initialized():
            tensors = {
                'loss': torch.tensor([local_loss_sum], device=self.device),
                'em': torch.tensor([local_em_sum], device=self.device),
                'f1': torch.tensor([local_f1_sum], device=self.device),
                'cnt': torch.tensor([local_count], device=self.device)
            }
            for t in tensors.values():
                dist.all_reduce(t, op=dist.ReduceOp.SUM)

            global_loss_sum = tensors['loss'].item()
            global_em_sum = tensors['em'].item()
            global_f1_sum = tensors['f1'].item()
            global_count = max(1.0, tensors['cnt'].item())

            avg_loss = global_loss_sum / global_count
            avg_em_perc = global_em_sum / global_count
            avg_f1_perc = global_f1_sum / global_count
        else:
            # Single-process
            global_count = max(1, local_count)
            avg_loss = local_loss_sum / global_count
            avg_em_perc = local_em_sum / global_count
            avg_f1_perc = local_f1_sum / global_count

        if self.scheduler is not None:
            self.scheduler.step(avg_loss)
            if is_main_process():
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"New LR: {current_lr}")

        # ----- Logging / Early Stopping / Saving (rank-0 only) -----
        if is_main_process():
            self.log_tensorboard(avg_loss, epoch, 'Validation')
            self.log_tensorboard(avg_em_perc, epoch, 'Validation', eval_metric='em')
            self.log_tensorboard(avg_f1_perc, epoch, 'Validation', eval_metric='f1')
            print(f"Epoch {epoch} - Validation - AvgLoss: {avg_loss:.4f} | "
                  f"AvgEM: {avg_em_perc:.4f} | AvgF1: {avg_f1_perc:.4f}")

            soft_prompt_path = (
                f"{self.model_dir}/"
                f"soft_prompt_epoch_{epoch}_val_loss_{avg_loss:.4f}_em_{avg_em_perc:2f}.pth"
            )

            # Save best by EM
            if avg_em_perc >= self.best_em:
                if self.best_model_path and os.path.exists(self.best_model_path):
                    try:
                        os.remove(self.best_model_path)
                        print(f"Removed previous best model at {self.best_model_path}")
                    except OSError:
                        pass

                self.best_em = avg_em_perc
                self.best_model_path = soft_prompt_path

                curvature_value = 0.0
                hl = getattr(self.net.knit5, 'additional_layer', None)
                if hl is not None and hasattr(hl, 'manifold') and hasattr(hl.manifold, 'c'):
                    curvature_value = hl.manifold.c.item()

                savings = {
                    'soft_prompt_state_dict': self.net.soft_prompt,
                    'additional_linear_layer': self.net.knit5.additional_layer.state_dict(),
                    'curvature': getattr(self.net.knit5, 'curvature', curvature_value),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'num_loop_iterations': self.training_config.num_loop_iterations,
                }
                torch.save(savings, soft_prompt_path)
                print(f"New best model Saved with EM {avg_em_perc:2f} at {soft_prompt_path}")

            # Early stopping on loss
            stopped = False
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
                print(f"Early stopping counter: {self.early_stop_counter} / {self.patience}")
                if self.early_stop_counter >= self.patience:
                    print("Early stopping triggered. Stopping training.")
                    stopped = True

            return stopped

        # Non-main processes never early-stop the loop themselves
        return False
