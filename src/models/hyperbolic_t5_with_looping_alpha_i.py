"""
T5 Model with Additional Bottleneck Layer (Integrated Looping Version)

This file maintains the full original T5 structure and argument handling, 
while integrating Shared-Weight Looping and Alpha_i Input Injection for both 
Hyperbolic and Euclidean (Linear) experiments.
"""

from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from transformers import T5ForConditionalGeneration, T5Config

from typing import Tuple, Optional, Any, Union
import torch.nn as nn
import torch
from torch.nn import CrossEntropyLoss

from .hyperbolic_model_utils import *

# --- Core Utility: Poincaré Ball Projection ---
def clamp_to_ball(x: torch.Tensor, max_norm: float = 0.9) -> torch.Tensor:
    """Project vectors to stay within the Poincaré ball boundary."""
    norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    clamped = torch.where(
        norm > max_norm,
        x * max_norm / norm,
        x
    )
    return clamped

class T5ModelWithLoopedHyperbolic(T5ForConditionalGeneration):
    def __init__(self,
                 layer_type : str,
                 model_name : str = 'google/t5-large-lm-adapt',
                 checkpoint_hyperbolic_knit5 : str = None,
                 with_model_state_dict = True,
                 curvature : Optional[float] = 0.3,
                 num_layers : int = 1,
                 gpu_parallelization = False, 
                 soft_prompt_length = 100,
                 only_map_soft_prompt = False,
                 # These parameters ensure compatibility with looping training scripts
                 num_loops: int = 1,
                 input_injection: bool = True,
                 max_norm: float = 0.9):
        
        config : T5Config = T5Config.from_pretrained(model_name)
        super(T5ModelWithLoopedHyperbolic, self).__init__(config=config)
        
        self.curvature = curvature
        self.soft_prompt_length = soft_prompt_length
        self.num_loops = num_loops
        self.input_injection = input_injection
        self.max_norm = max_norm
        self.only_map_soft_prompt = only_map_soft_prompt
        self.model_name = model_name
        self.additional_layer_type = layer_type
        
        in_features = config.d_model # 1024 for Large
        
        if self.only_map_soft_prompt:
            print(f"Passing only soft prompts through bottleneck")
        else:
            print(f"Passing everything through bottleneck layer")
            
        if layer_type not in ['linear', 'hyperbolic', 'identity']:
            raise ValueError(f"{layer_type} not supported. Use 'hyperbolic', 'linear', or 'identity'")

        # --- RECURRENT ARCHITECTURE SETUP ---
        # We use a single shared weight layer (1024 dimensions) to avoid
        # the 6144 dimension errors and to match your existing checkpoints.
        if layer_type == 'linear':
            self.hyperbolic_layer = nn.Linear(in_features=in_features, out_features=in_features)
            print(f"Using Euclidean Layer with {self.num_loops} shared loops")
        elif layer_type == 'hyperbolic':
            self.hyperbolic_layer = HyperbolicLayer(
                curvature=self.curvature, 
                type='poincare', 
                scaled=False, 
                learnable=True, 
                in_features=in_features, 
                out_features=in_features
            )
            print(f"Using Hyperbolic Layer with {self.num_loops} shared loops")
        else:
            self.hyperbolic_layer = nn.Identity()

        # Learnable Alpha for Input Injection (Anchoring the reasoning)
        self.alphas = nn.Parameter(torch.ones(num_loops) * 0.5)

        # --- CHECKPOINT LOADING ---
        if checkpoint_hyperbolic_knit5 is None:
            print("Initializing T5 Model from pretrained weights...")
            pretrained_model = T5ForConditionalGeneration.from_pretrained(model_name)
            missing, unexpected = self.load_state_dict(pretrained_model.state_dict(), strict = False)
            print(f"Missing: {len(missing)} keys, Unexpected: {len(unexpected)} keys")
            del pretrained_model
        else:
            print(f"Loading Checkpoint from {checkpoint_hyperbolic_knit5}")
            checkpoint = torch.load(checkpoint_hyperbolic_knit5, map_location='cpu')
            
            # Extract state_dict and handle module prefixing
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

        self.post_init()
        self.model_parallel = False
        self.device_map = None

    def _apply_recurrent_loop(self, h: torch.Tensor) -> torch.Tensor:
        """
        Processes hidden states through the shared weights iteratively
        using per-loop alphas (alpha_i).
        """
        h_original = h
        h_state = h

        for i in range(self.num_loops):
            # 1. Apply the shared layer f(h_{i-1})
            h_state = self.hyperbolic_layer(h_state)

            # 2. Euclidean Activation: Switch to Leaky ReLU for better gradient flow
            if self.additional_layer_type == 'linear':
                h_state = torch.nn.functional.leaky_relu(h_state, negative_slope=0.01)

            # 3. Hyperbolic Activation: Ball clamping for stability
            elif self.additional_layer_type == 'hyperbolic':
                h_state = clamp_to_ball(h_state, self.max_norm)

            # 4. Per-Loop Input Injection: h_i = alpha_i * f(h_{i-1}) + (1 - alpha_i) * h_enc
            if self.input_injection:
                # Sigmoid ensures the specific alpha_i stays between 0.0 and 1.0
                current_alpha = torch.sigmoid(self.alphas[i]) 
                h_state = current_alpha * h_state + (1 - current_alpha) * h_original
                
                # Final clamp for hyperbolic consistency if needed
                if self.additional_layer_type == 'hyperbolic':
                    h_state = clamp_to_ball(h_state, self.max_norm)
        
        return h_state

    def _forward_after_encoder(self,
                soft_prompt: Optional[torch.LongTensor] = None,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                decoder_input_ids: Optional[torch.LongTensor] = None,
                decoder_attention_mask: Optional[torch.BoolTensor] = None,
                head_mask: Optional[torch.FloatTensor] = None,
                decoder_head_mask: Optional[torch.FloatTensor] = None,
                cross_attn_head_mask: Optional[torch.Tensor] = None,
                encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                **kwargs):
        
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        #FUTURE-PROOFING: Head mask handling
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask

        # --- ENCODER LOGIC ---
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        # --- BOTTLENECK LOOP APPLICATION ---
        if self.only_map_soft_prompt:
            # Separate soft prompt and document text
            soft_prompt_hidden_state = hidden_states[:, :self.soft_prompt_length, :]
            soft_prompt_hidden_state = self._apply_recurrent_loop(soft_prompt_hidden_state)
            
            # Recombine
            hidden_states = torch.cat([
                soft_prompt_hidden_state, 
                hidden_states[:, self.soft_prompt_length:, :]
            ], dim=1)
        else:
            # Apply recurrent looping to the entire hidden state sequence
            hidden_states = self._apply_recurrent_loop(hidden_states)

        # Device mapping for parallelization
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        # Prepare Decoder Input IDs
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = self._shift_right(labels)

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # --- DECODER LOGIC ---
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        # Word embedding tie rescaling
        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
    
    def forward(self,
                soft_prompt: Optional[torch.LongTensor] = None,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                decoder_input_ids: Optional[torch.LongTensor] = None,
                decoder_attention_mask: Optional[torch.BoolTensor] = None,
                head_mask: Optional[torch.FloatTensor] = None,
                decoder_head_mask: Optional[torch.FloatTensor] = None,
                cross_attn_head_mask: Optional[torch.Tensor] = None,
                encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                **kwargs):
    
        return self._forward_after_encoder(
                             soft_prompt = soft_prompt,
                             input_ids = input_ids,
                             inputs_embeds=inputs_embeds,
                             attention_mask=attention_mask,
                             labels=labels,
                             decoder_input_ids=decoder_input_ids,
                             decoder_attention_mask=decoder_attention_mask,
                             head_mask=head_mask,
                             decoder_head_mask=decoder_head_mask,
                             cross_attn_head_mask=cross_attn_head_mask,
                             encoder_outputs=encoder_outputs,
                             past_key_values=past_key_values,
                             decoder_inputs_embeds=decoder_inputs_embeds,
                             use_cache=use_cache,
                             output_attentions=output_attentions,
                             output_hidden_states=output_hidden_states,
                             return_dict=return_dict,
                             **kwargs)
