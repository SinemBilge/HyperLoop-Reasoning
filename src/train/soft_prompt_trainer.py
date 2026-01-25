import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import sys
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from config import Config
from eval import exact_match_score, f1_score
from utils.trainer_utils import *
from src.models import SoftPromptModel
from typing import Union
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.utils.trainer_utils import CosineAnnealingWarmUpRestarts

"""Triggering Multi-Hop Reasoning for Question Answering
in Language Models using Soft Prompts and Random Walks: https://arxiv.org/pdf/2306.04009
"""
class SoftPromptTrainer:
    def __init__(self,
                 model : Union[SoftPromptModel, DDP],
                 tokenizer : AutoTokenizer,
                 train_dataloader: DataLoader,
                 val_dataloader : DataLoader,
                 config : Config,
                 device : str ='cpu',
                 checkpoint_path : str = None,
                 tboard_checkpoint_path : str = None,
                 validation_step : int = 1,
                 method : str = 'random_walk_training',
                 retrain : bool = False,
                 gpu_parallelization = False,
                 rank = 1):
        self.model = model.to(device)
        self.gpu_parallelization = gpu_parallelization
        self.rank = rank
        self.device = device
        if gpu_parallelization:
            self.model = DDP(self.model, device_ids=[self.device])
        self.tokenizer = tokenizer
        self.tokenizer.model_max_length = config.t5_model.tokenizer_max_length
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.tboard_checkpoint_path = tboard_checkpoint_path
        self.validation_step = validation_step
        
        self.supported_methods = ['random_walk_training', 'parse_then_hop_training']
        
        
        if method not in self.supported_methods:
            raise ValueError(f"Unsupported phase: {method} Supported Phases are: {self.supported_methods}")
        else:
            self.method = method
        

        if self.method == 'random_walk_training':
            self.training_config = config.random_walk_training
        elif self.method == 'parse_then_hop_training':
            self.training_config = config.parse_then_hop_training


        self.start_epoch = 0
        self.patience = self.training_config.epochs
        self.best_loss = float('inf')
        self.best_em = 0
        self.early_stop_counter=0
        self.best_model_path = None
        if self.gpu_parallelization:
            print(f"Hyperbolic Layer learnable: {all(param.requires_grad for param in self.model.module.knit5.hyperbolic_layer.parameters()) if self.model.module.knit5.additional_layer_type != 'identity' else False}")
        else:
            print(f"Hyperbolic Layer learnable: {all(param.requires_grad for param in self.model.knit5.hyperbolic_layer.parameters()) if self.model.knit5.additional_layer_type != 'identity' else False}")
        # optimizer_params = []
        # if self.training_config.use_soft_prompt:
        #     self.optimizer_soft_prompt = get_optimizer([{
        #         'params': self.model.module.soft_prompt if self.gpu_parallelization else self.model.soft_prompt,
        #         'lr': self.training_config.learning_rate
        #     }], self.training_config)
        # # Always add the hyperbolic layer
        # self.optimizer_hyperbolic_layer = get_optimizer([{
        #     'params': self.model.module.knit5.hyperbolic_layer.parameters() if self.gpu_parallelization else self.model.knit5.hyperbolic_layer.parameters(),
        #     'lr': config.single_hop_training.learning_rate
        # }], self.training_config)
        optimizer_params = []
        if self.training_config.use_soft_prompt:
            optimizer_params.append({
                'params': self.model.module.soft_prompt if self.gpu_parallelization else self.model.soft_prompt,
                'lr': self.training_config.learning_rate
            })

        knit5 = self.model.module.knit5 if self.gpu_parallelization else self.model.knit5

        # Collect all parameter ids already added to avoid duplicates
        added_param_ids = set()

        # Add hyperbolic layer parameters
        hyp_layer_params = list(knit5.hyperbolic_layer.parameters())
        optimizer_params.append({
            'params': hyp_layer_params,
            'lr': config.single_hop_training.learning_rate
        })
        for p in hyp_layer_params:
            added_param_ids.add(id(p))

        # Add alpha parameter if present (for looped hyperbolic models) and not already added
        if hasattr(knit5, 'alpha') and knit5.alpha is not None:
            if id(knit5.alpha) not in added_param_ids:
                optimizer_params.append({
                    'params': [knit5.alpha],
                    'lr': config.single_hop_training.learning_rate
                })
                added_param_ids.add(id(knit5.alpha))
                print(f"Added alpha parameter to optimizer (initial value: {knit5.alpha.item():.4f})")
            else:
                print(f"Alpha already in optimizer via hyperbolic_layer.parameters()")

        # Add manifold curvature parameter if learnable and not already added
        # Note: manifold might be shared with hyperbolic_layer, so check for duplicates
        if hasattr(knit5, 'manifold') and hasattr(knit5.manifold, 'isp_c') and knit5.manifold.learnable:
            if id(knit5.manifold.isp_c) not in added_param_ids:
                optimizer_params.append({
                    'params': [knit5.manifold.isp_c],
                    'lr': config.single_hop_training.learning_rate
                })
                added_param_ids.add(id(knit5.manifold.isp_c))
                print(f"Added manifold curvature parameter to optimizer")
            else:
                print(f"Manifold curvature already in optimizer via hyperbolic_layer.parameters()")

        self.optimizer = get_optimizer(optimizer_params, self.training_config)

        # Finally, create the optimizer
        # self.optimizer = get_optimizer(optimizer_params, self.training_config)
        if self.training_config.use_scheduler:
            self.scheduler = CosineAnnealingWarmUpRestarts(self.optimizer_hyperbolic_layer, T_0=150, T_mult=1, eta_max=0.001, T_up=10, gamma=0.5 )
            print(f"Using Scheduler!")
        from math import log, exp
        if hasattr(self.model.module.knit5.hyperbolic_layer, 'manifold') and hasattr(self.model.module.knit5.hyperbolic_layer.manifold, 'c'):
            c = self.model.module.knit5.hyperbolic_layer.manifold.c.item()
            c = log(exp(c) + 1)
        else:
            c = 0.0
        self.training_config.curvature = c
        self.log_dir, self.model_dir = setup_directories(self.training_config, self.config.t5_model)
        if self.tboard_checkpoint_path is not None:
            self.log_dir = tboard_checkpoint_path
            print(f"Continue writing to {tboard_checkpoint_path}")
        self.writer = SummaryWriter(log_dir=self.log_dir)
        print(f"{self.checkpoint_path = }")
        if self.checkpoint_path is not None and retrain:
            load_soft_prompt_and_additional_layer(self.model, checkpoint_path, gpu_parallelization=gpu_parallelization)
            self.optimizer_hyperbolic_layer, self.optimizer_soft_prompt, self.start_epoch = load_optimizer_and_start_epoch(self.optimizer_hyperbolic_layer, self.optimizer_soft_prompt, checkpoint_path)


        
            
        
        
    def log_tensorboard(self, loss, idx, phase, eval_metric = 'loss'):
        if eval_metric not in ['loss', 'em', 'f1']:
            raise ValueError(f'Unsupported eval_metric: {eval_metric}. Supported are [loss, em, f1]')
        else:
            self.writer.add_scalar(f'{phase}/{eval_metric}', loss, idx)
        
    
    def train(self, epochs: int):
        if self.start_epoch != 0:
            print(f'Starting training from epoch {self.start_epoch}')
        
        for epoch in range(self.start_epoch, epochs):
            if self.gpu_parallelization:
                self.model.module.knit5.eval()
                self.model.module.knit5.hyperbolic_layer.train()
            else:
                self.model.knit5.eval()
                self.model.knit5.hyperbolic_layer.train()
            
            if self.gpu_parallelization:
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
                # self.optimizer_hyperbolic_layer.zero_grad()
                # if self.training_config.use_soft_prompt:
                #     self.optimizer_soft_prompt.zero_grad()
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
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels_input_ids
                )
                loss = outputs.loss  # typically averaged over the local batch
                
                loss.backward()
                # self.optimizer_hyperbolic_layer.step()
                # if self.training_config.use_soft_prompt:
                #     self.optimizer_soft_prompt.step()
                self.optimizer.step()
                
                # For debugging/logging
                if batch_idx <= 5:
                    print(f"{input_batch[0] = }")
                    print(f"{label_batch[0] = }")
                
                # True global average (if needed)
                batch_size = input_ids.size(0)  # local batch size
                if self.gpu_parallelization:
                    loss_total_local = loss.item() * batch_size
                    
                    # Tensors to reduce
                    loss_tensor = torch.tensor([loss_total_local], device=self.device)
                    size_tensor = torch.tensor([batch_size], device=self.device)
                    
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                    dist.all_reduce(size_tensor, op=dist.ReduceOp.SUM)
                    
                    # global sums
                    global_loss_sum = loss_tensor.item()
                    global_size = size_tensor.item()
                    
                    # Global average loss across *this batch*
                    global_avg_loss = global_loss_sum / global_size
                    
                    # Log only on rank 0
                    if self.rank == 0:
                        total_loss += global_avg_loss
                        self.log_tensorboard(
                            global_avg_loss,
                            epoch * len(self.train_dataloader) + batch_idx,
                            'Training'
                        )
                else:
                    total_loss += loss.item()
                    self.log_tensorboard(
                        loss.item(),
                        epoch * len(self.train_dataloader) + batch_idx,
                        'Training'
                    )
                
                progress_bar.set_description(
                    f"Epoch {epoch} - Training - {self.method} - Loss: {loss.item():.4f}"
                )
                
                # Log VRAM usage on rank 0
                if self.rank == 0:
                    vram_allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
                    vram_reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 2)
                    self.writer.add_scalar(
                        'VRAM/Training/Allocated',
                        vram_allocated,
                        epoch * len(self.train_dataloader) + batch_idx
                    )
                    self.writer.add_scalar(
                        'VRAM/Training/Reserved',
                        vram_reserved,
                        epoch * len(self.train_dataloader) + batch_idx
                    )
                    # If there's a curvature attribute
                    if isinstance(self.model.module.knit5.hyperbolic_layer, nn.Sequential) and len(self.model.module.knit5.hyperbolic_layer) > 1:
                        # Iterate through each layer in the Sequential container
                        for layer_idx, layer in enumerate(self.model.module.knit5.hyperbolic_layer):
                            if not isinstance(layer, nn.ReLU):
                                if hasattr(layer, 'manifold') and hasattr(layer.manifold, 'c'):
                                    # Get the curvature value from the manifold
                                    c = layer.manifold.c.item()
                                else:
                                    c = 0.0

                                # Log the curvature for this specific layer
                                self.writer.add_scalar(
                                    f'Training/Curvature_Layer_{layer_idx}',
                                    c,
                                    epoch * len(self.train_dataloader) + batch_idx
                                )
                    else:
                        # Handle the case where hyperbolic_layer is a single layer
                        if hasattr(self.model.module.knit5.hyperbolic_layer, 'manifold') and hasattr(self.model.module.knit5.hyperbolic_layer.manifold, 'c'):
                            c = self.model.module.knit5.hyperbolic_layer.manifold.c.item()
                        else:
                            c = 0.0

                        # Log the curvature for the single layer
                        self.writer.add_scalar(
                            'Training/Curvature',
                            c,
                            epoch * len(self.train_dataloader) + batch_idx
                        )
            
            # Average loss over the entire epoch
            # If we're doing the "global" approach, 'total_loss' on rank 0 is the
            # sum of per-batch global_average_loss. So:
            if self.gpu_parallelization:
                if self.rank == 0:
                    avg_loss = total_loss / len(self.train_dataloader)
                    print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
            else:
                avg_loss = total_loss / len(self.train_dataloader)
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
            
            # Validation / Early stopping
            if (self.val_dataloader is not None) and (epoch % self.validation_step == 0):
                if self.evaluate(epoch=epoch):
                    break  # Early Stopping

            
    def evaluate(self, epoch):
        self.model.eval()

        # Local accumulators on THIS GPU
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
                inputs = self.tokenizer(
                    input_batch, padding=True, truncation=True, return_tensors='pt'
                ).to(self.device)

                # Tokenize labels separately, but do NOT move them to the GPU using .to(...) on the entire dict
                # because we only need input_ids on the GPU for the model:
                label_ids = self.tokenizer(
                    label_batch, padding=True, truncation=True, return_tensors='pt'
                )['input_ids'].to(self.device)

                outputs = self.model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    labels=label_ids
                )
                
                # 'outputs.loss' is typically the *average* loss per example in the batch.
                loss = outputs.loss

                # Number of examples in the current batch
                batch_size = len(input_batch)

                # Accumulate the *total* loss for this batch
                local_loss_sum += loss.item() * batch_size

                # Predictions
                predictions = torch.argmax(outputs.logits, dim=-1)
                decoded_predictions = [
                    self.tokenizer.decode(pred, skip_special_tokens=True)
                    for pred in predictions
                ]

                # Compute EM and F1 for each example; sum them locally
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

                # Update local count
                local_count += batch_size

                progress_bar.set_description(
                    f"Epoch {epoch} - Validation - {self.method} - Loss: {loss.item():.4f}"
                )

                # (Optional) Log a few predictions for debugging
                if batch_idx <= 5:
                    self.writer.add_text(
                        f'Validation/Prediction_vs_Label_{epoch}',
                        f'Prediction: {decoded_predictions[0]}\nLabel: {label_batch[0]}',
                        epoch
                    )

        # -- Distributed Reduction --
        if self.gpu_parallelization:
            # Convert local sums/counters to tensors
            total_loss_tensor = torch.tensor([local_loss_sum], device=self.device)
            total_em_tensor   = torch.tensor([local_em_sum],   device=self.device)
            total_f1_tensor   = torch.tensor([local_f1_sum],   device=self.device)
            count_tensor      = torch.tensor([local_count],    device=self.device)

            # Sum across all GPUs
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_em_tensor,   op=dist.ReduceOp.SUM)
            dist.all_reduce(total_f1_tensor,   op=dist.ReduceOp.SUM)
            dist.all_reduce(count_tensor,      op=dist.ReduceOp.SUM)

            # Get final global sums
            global_loss_sum = total_loss_tensor.item()
            global_em_sum   = total_em_tensor.item()
            global_f1_sum   = total_f1_tensor.item()
            global_count    = count_tensor.item()

            # Compute *global* averages
            # (If using a distributed sampler, global_count should match the full dataset size.)
            # If you replicate the dataset on each GPU, global_count = dataset_size * world_size.
            # So you adjust accordingly if needed.
            avg_loss   = global_loss_sum / global_count
            avg_em_perc= global_em_sum   / global_count
            avg_f1_perc= global_f1_sum   / global_count

        else:
            # Single-GPU or non-distributed:
            # Just compute local averages.
            avg_loss   = local_loss_sum / local_count
            avg_em_perc= local_em_sum   / local_count
            avg_f1_perc= local_f1_sum   / local_count
        if self.training_config.use_scheduler:
            self.scheduler.step(avg_loss)
            current_lr = self.optimizer_hyperbolic_layer.param_groups[0]['lr']
            print(f"New Hyperbolic LR: {current_lr}")

        # -- Logging / Early Stopping / Saving --
        if self.gpu_parallelization:
            if self.rank == 0:
                # Only rank 0 logs
                self.log_tensorboard(avg_loss, epoch, 'Validation')
                self.log_tensorboard(avg_em_perc, epoch, 'Validation', eval_metric='em')
                self.log_tensorboard(avg_f1_perc, epoch, 'Validation', eval_metric='f1')
                print(f"Epoch {epoch} - Validation - AvgLoss: {avg_loss:.4f} | "
                    f"AvgEM: {avg_em_perc:.4f} | AvgF1: {avg_f1_perc:.4f}")

                # Potential path for saving
                soft_prompt_path = (
                    f"{self.model_dir}/"
                    f"soft_prompt_epoch_{epoch}_val_loss_{avg_loss:.4f}_em_{avg_em_perc:2f}.pth"
                )

                # If this is the best EM so far, save model
                if avg_em_perc >= self.best_em:
                    if self.best_model_path and os.path.exists(self.best_model_path):
                        os.remove(self.best_model_path)
                        print(f"Removed previous best model at {self.best_model_path}")

                    self.best_em = avg_em_perc
                    self.best_model_path = soft_prompt_path
                    savings = {
                        'soft_prompt_state_dict': self.model.module.soft_prompt,
                        'additional_linear_layer': self.model.module.knit5.hyperbolic_layer.state_dict(),
                        'curvature': (self.model.module.knit5.curvature
                                    if hasattr(self.model.module.knit5.hyperbolic_layer, 'manifold')
                                    else 0.0),
                        # 'optimizer_hyperbolic_state_dict': self.optimizer_hyperbolic_layer.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'epoch': epoch
                    }
                    # Save alpha parameter if present (for looped hyperbolic models)
                    if hasattr(self.model.module.knit5, 'alpha'):
                        savings['alpha'] = self.model.module.knit5.alpha.data
                    # Save looping parameters if present
                    if hasattr(self.model.module.knit5, 'num_loops'):
                        savings['num_loops'] = self.model.module.knit5.num_loops
                    if hasattr(self.model.module.knit5, 'max_norm'):
                        savings['max_norm'] = self.model.module.knit5.max_norm
                    # if self.training_config.use_soft_prompt:
                    #     savings["optimizer_softprompt_state_dict"] = self.optimizer_soft_prompt.state_dict()
                    torch.save(savings, soft_prompt_path)
                    print(f"New best model Saved with EM {avg_em_perc:2f} at {soft_prompt_path}")

                # Early stopping check based on loss
                if avg_loss < self.best_loss:
                    self.best_loss = avg_loss
                    self.early_stop_counter = 0
                else:
                    self.early_stop_counter += 1
                    print(f"Early stopping counter: {self.early_stop_counter} / {self.patience}")
                    if self.early_stop_counter >= self.patience:
                        print("Early stopping triggered. Stopping training.")
                        return True
        else:
            # Non-distributed case: always log
            self.log_tensorboard(avg_loss, epoch, 'Validation')
            self.log_tensorboard(avg_em_perc, epoch, 'Validation', eval_metric='em')
            self.log_tensorboard(avg_f1_perc, epoch, 'Validation', eval_metric='f1')
            print(f"Epoch {epoch} - Validation - AvgLoss: {avg_loss:.4f} | "
                f"AvgEM: {avg_em_perc:.4f} | AvgF1: {avg_f1_perc:.4f}")

            soft_prompt_path = (
                f"{self.model_dir}/"
                f"soft_prompt_epoch_{epoch}_val_loss_{avg_loss:.4f}_em_{avg_em_perc:2f}.pth"
            )

            # Save model if EM is best
            if avg_em_perc > self.best_em:
                if self.best_model_path and os.path.exists(self.best_model_path):
                    os.remove(self.best_model_path)
                    print(f"Removed previous best model at {self.best_model_path}")

                self.best_em = avg_em_perc
                self.best_model_path = soft_prompt_path
                savings = {
                    'soft_prompt_state_dict': self.model.soft_prompt,
                    'additional_linear_layer': self.model.knit5.hyperbolic_layer.state_dict(),
                    'curvature': (self.model.knit5.curvature
                                if hasattr(self.model.knit5.hyperbolic_layer, 'manifold')
                                else 0.0),
                    # 'optimizer_hyperbolic_state_dict': self.optimizer_hyperbolic_layer.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch
                }
                # Save alpha parameter if present (for looped hyperbolic models)
                if hasattr(self.model.knit5, 'alpha'):
                    savings['alpha'] = self.model.knit5.alpha.data
                # Save looping parameters if present
                if hasattr(self.model.knit5, 'num_loops'):
                    savings['num_loops'] = self.model.knit5.num_loops
                if hasattr(self.model.knit5, 'max_norm'):
                    savings['max_norm'] = self.model.knit5.max_norm
                # if self.training_config.use_soft_prompt:
                #         savings["optimizer_softprompt_state_dict"] = self.optimizer_soft_prompt.state_dict()
                torch.save(savings, soft_prompt_path)
                print(f"New best model Saved with EM {avg_em_perc:2f} at {soft_prompt_path}")
            # Early stopping check
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
                print(f"Early stopping counter: {self.early_stop_counter} / {self.patience}")
                if self.early_stop_counter >= self.patience:
                    print("Early stopping triggered. Stopping training.")
                    return True

        return False


   
    
            
            
            
            
            
    
    
    
    
    
