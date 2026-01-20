"""
Hyperbolic Looped T5 - TRAINABLE VERSION

This is different from looped_hyperbolic_t5.py which loops at inference time only.
This version is designed to be TRAINED with looping, following the key insight from
Yang et al. 2024 "Looped Transformers are Better at Learning Learning Algorithms":

    Looped transformers require training with the looping architecture,
    not just inference-time looping.

Architecture (Approach 1 - Loop inside hyperbolic space):
    Input → T5 Encoder → expmap → [Hyperbolic Loop Block × L] → logmap → Decoder

Key features:
1. Input injection at each loop iteration (prevents representation collapse)
2. Shared loop block weights (parameter efficient, like looped transformer paper)
3. Learnable curvature
4. Designed to load your existing KI checkpoint (identity layer)
"""

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import T5ForConditionalGeneration, T5Config
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from typing import Optional, Tuple, List

from .hyperbolic_model_utils import HyperbolicLayer
from .hyperbolic_nn_resnet.manifolds import PoincareBallCustomAutograd
from .hyperbolic_nn_resnet.modules import PoincareLinear


class HyperbolicLoopBlock(nn.Module):
    """
    A single hyperbolic loop block that operates in Poincaré ball space.
    Includes input injection for stable iteration.
    """
    def __init__(self, hidden_size: int, curvature: float, learnable_curvature: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.manifold = PoincareBallCustomAutograd(c=curvature, learnable=learnable_curvature)

        # Hyperbolic linear transformation
        self.hyp_linear = PoincareLinear(
            in_features=hidden_size,
            out_features=hidden_size,
            ball=self.manifold
        )

        # Layer norm in tangent space
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Residual weight (learnable) - key for input injection
        self.residual_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor, x_original: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Current hidden state in Poincaré ball
            x_original: Original input for residual/input injection
        Returns:
            Refined hidden state in Poincaré ball
        """
        # Apply hyperbolic transformation
        h = self.hyp_linear(x)

        # Input injection: weighted combination with original
        # This is KEY to preventing representation collapse
        if x_original is not None:
            alpha = torch.sigmoid(self.residual_weight)
            h = alpha * h + (1 - alpha) * x_original

        return h


class HyperbolicLoopedT5Trainable(nn.Module):
    """
    Hyperbolic Looped T5 designed for TRAINING with loops.

    Architecture:
        Input → T5 Encoder → expmap → [Hyperbolic Loop Block × L] → logmap → Decoder

    This is different from LoopedHyperbolicT5 which just loops the hyperbolic layer
    at inference time without training for it.
    """
    def __init__(
        self,
        model_name: str = 'google/t5-large-lm-adapt',
        num_loops: int = 4,
        curvature: float = 0.37,
        learnable_curvature: bool = True,
        input_injection: bool = True,
        soft_prompt_length: int = 100,
    ):
        super().__init__()

        self.num_loops = num_loops
        self.curvature = curvature
        self.input_injection = input_injection
        self.soft_prompt_length = soft_prompt_length
        self.model_name = model_name

        # Load base T5 model
        self.config = T5Config.from_pretrained(model_name)
        self.hidden_size = self.config.d_model

        # Initialize T5 components from pretrained
        print(f"Loading base T5 from {model_name}...")
        base_model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.shared = base_model.shared
        self.encoder = base_model.encoder
        self.decoder = base_model.decoder
        self.lm_head = base_model.lm_head
        del base_model

        # Poincaré ball manifold for expmap/logmap
        self.manifold = PoincareBallCustomAutograd(c=curvature, learnable=learnable_curvature)

        # Hyperbolic loop block (shared weights across loops)
        self.loop_block = HyperbolicLoopBlock(
            hidden_size=self.hidden_size,
            curvature=curvature,
            learnable_curvature=learnable_curvature
        )

        # Layer norm before and after hyperbolic processing
        self.pre_hyp_norm = nn.LayerNorm(self.hidden_size)
        self.post_hyp_norm = nn.LayerNorm(self.hidden_size)

        print(f"HyperbolicLoopedT5Trainable initialized:")
        print(f"  - Loops: {num_loops}")
        print(f"  - Curvature: {curvature} (learnable={learnable_curvature})")
        print(f"  - Input injection: {input_injection}")

    def load_ki_checkpoint(self, checkpoint_path: str, gpu_parallelization: bool = False):
        """
        Load Stage 1 Knowledge Integration checkpoint (trained with identity layer).

        This loads T5 encoder/decoder/embeddings. The hyperbolic loop components
        remain at their initialized values and will be trained in Stage 2.

        Args:
            checkpoint_path: Path to KI checkpoint (.pth file)
            gpu_parallelization: Whether checkpoint was saved with DataParallel

        Returns:
            self for chaining
        """
        print(f"Loading KI checkpoint from {checkpoint_path}")

        device = next(self.parameters()).device if len(list(self.parameters())) > 0 else 'cpu'
        checkpoint = torch.load(checkpoint_path, map_location=device)

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Handle DataParallel prefix
        if gpu_parallelization:
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        # Get our model's state dict for shape comparison
        model_state = self.state_dict()

        # Map checkpoint keys to our model structure
        mapped_state_dict = {}
        skipped_keys = []

        for key, value in state_dict.items():
            # Skip hyperbolic_layer from checkpoint (was identity, no meaningful params)
            if 'hyperbolic_layer' in key:
                skipped_keys.append(key)
                continue

            # Check if key exists in our model with matching shape
            if key in model_state and model_state[key].shape == value.shape:
                mapped_state_dict[key] = value
            else:
                skipped_keys.append(key)

        # Load the mapped weights
        missing, unexpected = self.load_state_dict(mapped_state_dict, strict=False)

        # Report
        print(f"  Loaded: {len(mapped_state_dict)} parameters (T5 components)")
        print(f"  Skipped from checkpoint: {len(skipped_keys)} keys")

        # Show what's randomly initialized
        hyp_missing = [k for k in missing if any(x in k for x in ['loop_block', 'manifold', 'hyp', '_norm'])]
        print(f"  Randomly initialized (hyperbolic loop): {len(hyp_missing)} keys")
        for key in hyp_missing[:5]:
            print(f"    - {key}")
        if len(hyp_missing) > 5:
            print(f"    ... and {len(hyp_missing) - 5} more")

        return self

    def forward_encoder_with_loops(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        return_loop_outputs: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass through encoder + hyperbolic looping.

        Returns:
            hidden_states: Final encoder hidden states
            loop_outputs: Optional list of outputs from each loop
        """
        # Standard encoder forward
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
        )
        hidden_states = encoder_outputs.last_hidden_state

        # Pre-normalization
        hidden_states = self.pre_hyp_norm(hidden_states)

        # Map to Poincaré ball
        h_poincare = self.manifold.expmap0(hidden_states)
        h_original = h_poincare if self.input_injection else None

        # Collect loop outputs for analysis
        loop_outputs = [] if return_loop_outputs else None

        # Iterative refinement in hyperbolic space
        for loop_idx in range(self.num_loops):
            h_poincare = self.loop_block(h_poincare, h_original)

            if return_loop_outputs:
                loop_outputs.append(self.manifold.logmap0(h_poincare))

        # Map back to Euclidean space
        hidden_states = self.manifold.logmap0(h_poincare)

        # Post-normalization
        hidden_states = self.post_hyp_norm(hidden_states)

        return hidden_states, loop_outputs

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # Encoder with hyperbolic looping
        hidden_states, _ = self.forward_encoder_with_loops(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )

        # Prepare decoder inputs
        if labels is not None and decoder_input_ids is None:
            decoder_input_ids = self._shift_right(labels)

        # Decoder forward
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Rescale for tied embeddings
        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.config.d_model ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        # Compute loss
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values if hasattr(decoder_outputs, 'past_key_values') else None,
            decoder_hidden_states=decoder_outputs.hidden_states if hasattr(decoder_outputs, 'hidden_states') else None,
            decoder_attentions=decoder_outputs.attentions if hasattr(decoder_outputs, 'attentions') else None,
            encoder_last_hidden_state=hidden_states,
        )

    def _shift_right(self, input_ids):
        """Shift input ids one token to the right for decoder input"""
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids

    def generate(self, input_ids=None, attention_mask=None, inputs_embeds=None, **kwargs):
        """Generation with hyperbolic looped encoding"""
        hidden_states, _ = self.forward_encoder_with_loops(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )

        # Simple greedy decoding
        batch_size = hidden_states.shape[0]
        device = hidden_states.device
        max_length = kwargs.get('max_length', 50)

        decoder_input_ids = torch.full(
            (batch_size, 1),
            self.config.decoder_start_token_id,
            dtype=torch.long,
            device=device
        )

        for _ in range(max_length - 1):
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=attention_mask,
                return_dict=True,
            )

            sequence_output = decoder_outputs.last_hidden_state
            if self.config.tie_word_embeddings:
                sequence_output = sequence_output * (self.config.d_model ** -0.5)

            logits = self.lm_head(sequence_output[:, -1:, :])
            next_token = logits.argmax(dim=-1)

            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=-1)

            if (next_token == self.config.eos_token_id).all():
                break

        return decoder_input_ids


class SoftPromptHyperbolicLoopedModel(nn.Module):
    """
    Wrapper that adds soft prompts to HyperbolicLoopedT5Trainable for Stage 2 training.
    """
    def __init__(
        self,
        hyperbolic_looped_model: HyperbolicLoopedT5Trainable,
        soft_prompt_length: int = 100,
        freeze_base: bool = True,
    ):
        super().__init__()

        self.model = hyperbolic_looped_model
        self.soft_prompt_length = soft_prompt_length
        self.hidden_size = hyperbolic_looped_model.hidden_size
        self.model_name = hyperbolic_looped_model.model_name

        # Initialize soft prompt
        self.soft_prompt = nn.Parameter(torch.randn(soft_prompt_length, self.hidden_size))
        nn.init.xavier_uniform_(self.soft_prompt)

        # Freeze base model if specified
        if freeze_base:
            for name, param in self.model.named_parameters():
                # Keep hyperbolic components trainable
                if 'loop_block' in name or 'hyp' in name or 'manifold' in name or '_norm' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        # Always keep soft prompt trainable
        self.soft_prompt.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"SoftPromptHyperbolicLoopedModel: {trainable:,} / {total:,} trainable params")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor, **kwargs):
        batch_size = input_ids.size(0)
        device = input_ids.device

        # Get input embeddings
        inputs_embeds = self.model.shared(input_ids)

        # Prepend soft prompt
        soft_prompt_batch = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        inputs_embeds = torch.cat([soft_prompt_batch, inputs_embeds], dim=1)

        # Update attention mask
        prompt_mask = torch.ones(batch_size, self.soft_prompt_length, dtype=attention_mask.dtype, device=device)
        attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        # Forward through model
        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
        batch_size = input_ids.size(0)
        device = input_ids.device

        inputs_embeds = self.model.shared(input_ids)

        soft_prompt_batch = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        inputs_embeds = torch.cat([soft_prompt_batch, inputs_embeds], dim=1)

        prompt_mask = torch.ones(batch_size, self.soft_prompt_length, dtype=attention_mask.dtype, device=device)
        attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        return self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs
        )

    def save_checkpoint(self, path: str):
        """Save soft prompt and hyperbolic loop block parameters"""
        torch.save({
            'soft_prompt': self.soft_prompt.data,
            'loop_block_state_dict': self.model.loop_block.state_dict(),
            'pre_hyp_norm': self.model.pre_hyp_norm.state_dict(),
            'post_hyp_norm': self.model.post_hyp_norm.state_dict(),
            'manifold_isp_c': self.model.manifold.isp_c.data if hasattr(self.model.manifold, 'isp_c') else None,
            'soft_prompt_length': self.soft_prompt_length,
            'num_loops': self.model.num_loops,
            'curvature': self.model.curvature,
        }, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load soft prompt and hyperbolic loop block parameters"""
        device = next(self.parameters()).device
        checkpoint = torch.load(path, map_location=device)

        self.soft_prompt.data = checkpoint['soft_prompt']
        self.model.loop_block.load_state_dict(checkpoint['loop_block_state_dict'])
        if 'pre_hyp_norm' in checkpoint:
            self.model.pre_hyp_norm.load_state_dict(checkpoint['pre_hyp_norm'])
        if 'post_hyp_norm' in checkpoint:
            self.model.post_hyp_norm.load_state_dict(checkpoint['post_hyp_norm'])
        if checkpoint.get('manifold_isp_c') is not None and hasattr(self.model.manifold, 'isp_c'):
            self.model.manifold.isp_c.data = checkpoint['manifold_isp_c']

        print(f"Loaded checkpoint from {path}")
