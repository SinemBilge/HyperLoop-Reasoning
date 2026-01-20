import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import sys
from datetime import datetime
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from transformers import Adafactor
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from config import Config
from eval import exact_match_score, f1_score
from utils.trainer_utils import *
from typing import Union
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from src.models import T5ModelWithAdditionalLayer
import signal
import sys

"""Triggering Multi-Hop Reasoning for Question Answering
in Language Models using Soft Prompts and Random Walks: https://arxiv.org/pdf/2306.04009
"""
class ModelTrainer:
    def __init__(self,
                 model : Union[nn.Module, T5ModelWithAdditionalLayer, DDP],
                 tokenizer : AutoTokenizer,
                 list_train_dataloader: list,
                 val_dataloader : DataLoader,
                 config : Config,
                 device : str ='cpu',
                 checkpoint_path : str = None,
                 tboard_checkpoint_path : str = None,
                 validation_step : int = 1,
                 method : str = 'single_hop_training',
                 load_optimizer = True,
                 gpu_parallelization = False,
                 rank = 0):
        self.device = device
        self.model = model.to(device)
        self.gpu_parallelization = gpu_parallelization
        self.rank = rank
        if gpu_parallelization:
            self.model = DDP(self.model, device_ids=[rank])
        self.tokenizer = tokenizer
        self.tokenizer.model_max_length = config.t5_model.tokenizer_max_length
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.train_dataloader = list_train_dataloader
        if len(list_train_dataloader) == 1:
            self.train_dataloader = list_train_dataloader[0]
        elif len(list_train_dataloader) > 1:
            self.single_hop_train_dataloader = list_train_dataloader[0]
            self.c4_train_dataloader = list_train_dataloader[1]
        else:
            raise ValueError('Length of List for Train Dataloaders must be >= 1')
        self.val_dataloader = val_dataloader
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.tboard_checkpoint_path = tboard_checkpoint_path
        self.validation_step = validation_step
        self.load_optimizer = load_optimizer
        self.gradient_accumulation_steps = 2
        
        
        
        self.supported_methods = ['single_hop_training', 'one_hop_wiki_training']
        
        
        if method not in self.supported_methods:
            raise ValueError(f"Unsupported phase: {method} Supported Phases are: {self.supported_methods}")
        else:
            self.method = method
        
        if self.method == 'single_hop_training':
            self.training_config = config.single_hop_training
        elif self.method == 'one_hop_wiki_training':
            self.training_config = config.one_hop_wiki_training
        self.optimizer = get_optimizer(self.model.parameters(), self.training_config)
        

        self.start_epoch = 0
        self.patience = self.training_config.epochs
        self.best_loss = float('inf')
        self.best_em = 0
        self.early_stop_counter=0
        self.best_model_path = None

        from transformers import get_linear_schedule_with_warmup
        num_steps = len(self.train_dataloader) * self.training_config.epochs
        if self.training_config.scheduler is not None:
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_training_steps=num_steps, num_warmup_steps=int(0.1 * num_steps))
        else:
            self.scheduler = None
        print(f"Using Scheduler: {self.scheduler}")
        
        self.log_dir, self.model_dir = setup_directories(self.training_config, self.config.t5_model)
        if self.tboard_checkpoint_path is not None:
            self.log_dir = tboard_checkpoint_path
            print(f"Continue writing to {tboard_checkpoint_path}")
        self.writer = SummaryWriter(log_dir=self.log_dir)
        if self.checkpoint_path is not None:
            self.model = load_model_checkpoint(self.model, checkpoint_path)
            self.optimizer, self.start_epoch = load_optimizer_and_start_epoch(self.optimizer, checkpoint_path)  
        
    def train(self,
            epochs : int):
        """Trains Knowledge Integration part. Given 'e1 ; r1' predict 'e2'
        """
        if hasattr(self.model, 'module'):
            self.model.module.gradient_checkpointing_enable()
        else:
            self.model.gradient_checkpointing_enable()
        if self.start_epoch != 0:
            print(f'Starting training from epoch {self.start_epoch}')


        # if self.train_dataloader is not None:
        #     print(f"Training without C4...")
        #     self._train_without_c4(epochs)
        # else:
        print(f"Training with C4...")
        self._train_with_c4(epochs)
                
                    
    def _train_with_c4(self, epochs : int):
        min_len = min(len(self.single_hop_train_dataloader), len(self.c4_train_dataloader))
        num_iterations = 2 * min_len
        for epoch in range(epochs):
            self.model.train()
            if self.gpu_parallelization:
                self.single_hop_train_dataloader.sampler.set_epoch(epoch)
                self.c4_train_dataloader.sampler.set_epoch(epoch)



            total_c4_loss = 0
            total_single_hop_loss = 0
            single_hop_iter = iter(self.single_hop_train_dataloader)
            c4_iter = iter(self.c4_train_dataloader)
            progress_bar = tqdm(range(2*min(len(self.single_hop_train_dataloader), len(self.c4_train_dataloader))), leave=True, desc=f"Epoch {epoch} - Training - {self.method}", file=sys.stdout, dynamic_ncols=True)
            # progress_bar = tqdm(self.train_dataloader, leave=True, desc=f"Epoch {epoch} - Training - {self.method}", file=sys.stdout, dynamic_ncols=True)
            for batch_idx in progress_bar:
                #dataset type either: knowledge_integration or c4
                #Train in 50:50 Mixture
                if batch_idx % 2 == 0:
                    try:
                        batch = next(single_hop_iter)
                    except StopIteration:
                        single_hop_iter = iter(self.single_hop_train_dataloader)
                        batch = next(single_hop_iter)
                else:
                    try:
                        batch = next(c4_iter)
                    except StopIteration:
                        c4_iter = iter(self.c4_train_dataloader)
                        batch = next(c4_iter)
                
                        
                input_str, label = batch
                
                tokenized_inputs = self.tokenizer(input_str, padding=True, truncation=True, return_tensors='pt')
                tokenized_labels = self.tokenizer(label, padding=True, truncation=True, return_tensors='pt')
                tokenized_labels[tokenized_labels == self.tokenizer.pad_token_id] = -100
                input_ids = tokenized_inputs['input_ids'].to(self.device)
                attention_mask = tokenized_inputs['attention_mask'].to(self.device)
                labels = tokenized_labels['input_ids'].to(self.device)
                
                if batch_idx % self.gradient_accumulation_steps == 0:
                    self.optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels = labels, use_cache=False)
                loss = outputs.loss / self.gradient_accumulation_steps  # Scale loss

                loss.backward()

                # Only step optimizer every N accumulation steps
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    
                # if batch_idx <= 10 and batch_idx % 2 == 0:
                #     print(f"{input_str = }")
                #     print(f"{label = }")
                global_step = epoch* num_iterations + batch_idx
                if self.gpu_parallelization:
                    # After loss.backward() and self.optimizer.step()
                    loss_tensor = torch.tensor([loss.item()], device=self.device)
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                    global_loss = loss_tensor.item() / dist.get_world_size()

                    if self.rank == 0:
                        # Log only from rank 0
                        if batch_idx % 2 == 0:
                            total_single_hop_loss += global_loss
                            log_tensorboard(self.writer, global_loss, global_step, 'Training/SingleHop', eval_metric='loss')
                        else:
                            total_c4_loss += global_loss
                            log_tensorboard(self.writer, global_loss, global_step, 'Training/C4', eval_metric='loss')
                else:
                    if batch_idx % 2 == 0:
                        total_single_hop_loss += loss.item()
                        log_tensorboard(self.writer, loss.item(), global_step, 'Training/SingleHop', eval_metric='loss')
                    else:
                        total_c4_loss += loss.item()
                        log_tensorboard(self.writer, loss.item(), global_step, 'Training/C4', eval_metric='loss')
                
                progress_bar.set_description(f"Epoch {epoch} - Training - {self.method} - Loss: {loss.item():.4f}")
                
                
                if self.rank == 0:
                    vram_allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
                    vram_reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 2)
                    self.writer.add_scalar('VRAM/Training/Allocated', vram_allocated, global_step)
                    self.writer.add_scalar('VRAM/Training/Reserved', vram_reserved, global_step)
                    actual_model = self.model.module if hasattr(self.model, 'module') else self.model
                    if actual_model.additional_layer_type == 'hyperbolic':
                        c = actual_model.hyperbolic_layer.manifold.c.item()
                    else:
                        c = 0.0
                    self.writer.add_scalar('Training/Curvature', c, global_step)

            avg_single_hop_loss = total_single_hop_loss / min_len
            avg_c4_loss = total_c4_loss / min_len
            print(f"Epoch {epoch} - Training - AVGLoss for SingleHop: {avg_single_hop_loss:.4f} | AVGLoss for C4: {avg_c4_loss:.4f}")
            
            if (self.val_dataloader is not None) and (epoch % self.validation_step == 0):
                if self.evaluate(epoch=epoch):
                    break #Early Stopping
        
        
        
    def _train_without_c4(self, epochs : int):
        for epoch in range(self.start_epoch, epochs):
            progress_bar = tqdm(self.train_dataloader, leave=True, desc=f"Epoch {epoch} - Training - {self.method}", file=sys.stdout, dynamic_ncols=True)
            total_loss = 0
            for batch_idx, batch in enumerate(progress_bar):
                self.optimizer.zero_grad()
                
                
                input_str, label = batch
                
                
                tokenized_inputs = self.tokenizer(input_str, padding=True, truncation=True, return_tensors='pt').to(self.device)
                tokenized_labels = self.tokenizer(label, padding=True, truncation=True, return_tensors='pt').to(self.device)
                input_ids = tokenized_inputs['input_ids']
                attention_mask = tokenized_inputs['attention_mask']
                labels = tokenized_labels['input_ids']
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels = labels)
                loss = outputs.loss
                loss.backward()         
                
                self.optimizer.step()

                total_loss += loss.item()
                progress_bar.set_description(f"Epoch {epoch} - Training - {self.method} - Loss: {loss.item():.4f}")
                log_tensorboard(self.writer, loss.item(), epoch*len(self.train_dataloader) + batch_idx, 'Training', eval_metric='loss')
                
                vram_allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 2)  # Convert to MB
                vram_reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 2)  # Convert to MB
                self.writer.add_scalar('VRAM/Training/Allocated', vram_allocated, epoch*len(self.train_dataloader) + batch_idx)
                self.writer.add_scalar('VRAM/Training/Reserved', vram_reserved, epoch*len(self.train_dataloader) + batch_idx)
            avg_loss = total_loss / len(self.train_dataloader)
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

            if (self.val_dataloader is not None) and (epoch % self.validation_step == 0):
                if self.evaluate(epoch=epoch):
                    break #Early Stopping
    
            
    def evaluate(self,
                epoch : int):
        self.model.eval()
        total_loss = 0
        total_em = 0
        total_f1 = 0
        progress_bar = tqdm(self.val_dataloader, leave=True, desc=f"Epoch {epoch} - Validation - {self.method}", file=sys.stdout, dynamic_ncols=True)
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                input_str, label = batch
                inputs = self.tokenizer(input_str, padding=True, truncation=True, return_tensors = 'pt').to(self.device)
                labels = self.tokenizer(label, padding=True, truncation=True, return_tensors = 'pt')['input_ids'].to(self.device)
                
                outputs = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=labels)
                loss = outputs.loss
                
                total_loss += loss.item()
                
                predictions = torch.argmax(outputs.logits, dim=-1)
                decoded_predictions = [self.tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions]
                _f1_score = sum([f1_score(pred, truth)[0] for pred, truth, in zip(decoded_predictions, label)])
                em_score = sum([1 if exact_match_score(pred, truth) else 0 for pred, truth in zip(decoded_predictions, label)])
                
                
                total_em += em_score
                total_f1 += _f1_score
                progress_bar.set_description(f"Epoch {epoch} - Validation - {self.method} - Loss: {loss.item():.4f}")
                if batch_idx <= 5: 
                    self.writer.add_text(f'Validation/Prediction_vs_Label_{epoch}', 
                                     f'Prediction: {decoded_predictions[0]}\nLabel: {label[0]}', epoch)
                
                #progress_bar.set_description(f"Epoch {epoch} - Validation - Random Walk Training - Loss: {loss.item():.4f}")
            if self.gpu_parallelization:
                # Convert totals to tensors and reduce
                total_loss_tensor = torch.tensor([total_loss], device=self.device)
                dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
                avg_loss = (total_loss_tensor.item() / dist.get_world_size()) / len(self.val_dataloader)

                total_em_tensor = torch.tensor([total_em], device=self.device)
                dist.all_reduce(total_em_tensor, op=dist.ReduceOp.SUM)
                avg_em_perc = total_em_tensor.item() / len(self.val_dataloader.dataset)

                total_f1_tensor = torch.tensor([total_f1], device=self.device)
                dist.all_reduce(total_f1_tensor, op=dist.ReduceOp.SUM)
                avg_f1_perc = total_f1_tensor.item() / len(self.val_dataloader.dataset)

                
                # Log only aggregated metrics from the main process
            else:
                avg_loss = total_loss / len(self.val_dataloader)
                avg_em_perc = total_em / len(self.val_dataloader.dataset)
                avg_f1_perc = total_f1 / len(self.val_dataloader.dataset)
        
        
        if self.gpu_parallelization:
            if self.rank == 0:  # Only main process logs and handles model saving
                log_tensorboard(self.writer, avg_loss, epoch, 'Validation', eval_metric='loss')
                log_tensorboard(self.writer, avg_em_perc, epoch, 'Validation', eval_metric='em')
                log_tensorboard(self.writer, avg_f1_perc, epoch, 'Validation', eval_metric='f1')
                print(f"Epoch {epoch} - Validation - AvgLoss: {avg_loss:.4f} | AvgEM: {avg_em_perc:.4f} | AvgF1: {avg_f1_perc:.4f}")
            
                soft_prompt_path = f"{self.model_dir}/knit5.pth"
            
                if avg_em_perc > self.best_em:
                    if self.best_model_path:
                        if os.path.exists(self.best_model_path):
                            os.remove(self.best_model_path)
                            print(f"Removed previous best model at {self.best_model_path}")
                    self.best_em = avg_em_perc
                    self.early_stop_counter = 0
                    self.best_model_path = soft_prompt_path
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'epoch': epoch
                    }, self.best_model_path)
                else:
                    self.early_stop_counter += 1
                    print(f"Early stopping counter: {self.early_stop_counter} / {self.patience}")
                    if self.early_stop_counter >= self.patience:
                        print("Early stopping triggered. Stopping training")
                        return True
        else:
            log_tensorboard(self.writer, avg_loss, epoch, 'Validation', eval_metric='loss')
            log_tensorboard(self.writer, avg_em_perc, epoch, 'Validation', eval_metric='em')
            log_tensorboard(self.writer, avg_f1_perc, epoch, 'Validation', eval_metric='f1')
            print(f"Epoch {epoch} - Validation - AvgLoss: {avg_loss:.4f} | AvgEM: {avg_em_perc:.4f} | AvgF1: {avg_f1_perc:.4f}")
        
            soft_prompt_path = f"{self.model_dir}/knit5.pth"
        
            if avg_em_perc > self.best_em:
                if self.best_model_path and os.path.exists(self.best_model_path):
                    os.remove(self.best_model_path)
                    print(f"Removed previous best model at {self.best_model_path}")
                self.best_em = avg_em_perc
                self.early_stop_counter = 0
                self.best_model_path = soft_prompt_path
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch
                }, self.best_model_path)
            else:
                self.early_stop_counter += 1
                print(f"Early stopping counter: {self.early_stop_counter} / {self.patience}")
                if self.early_stop_counter >= self.patience:
                    print("Early stopping triggered. Stopping training")
                    return True
                
        # Optionally, synchronize the early_stop_counter and best_em across all processes
        if self.gpu_parallelization:
            # Broadcast best_em and early_stop_counter from rank 0 to all processes
            if self.rank == 0:
                best_em_tensor = torch.tensor([self.best_em], device=self.device)
                early_stop_tensor = torch.tensor([self.early_stop_counter], device=self.device)
            else:
                best_em_tensor = torch.tensor([0.0], device=self.device)
                early_stop_tensor = torch.tensor([0], device=self.device)
            
            dist.broadcast(best_em_tensor, src=0)
            dist.broadcast(early_stop_tensor, src=0)
            
            if self.rank != 0:
                self.best_em = best_em_tensor.item()
                self.early_stop_counter = early_stop_tensor.item()
        return False

   
    
            
            
            
            
            
    
    
    
    
    
