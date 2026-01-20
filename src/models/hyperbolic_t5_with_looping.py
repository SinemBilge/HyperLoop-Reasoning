"""
T5 Model with Hyperbolic Layer + Looping

This extends the original T5ModelWithAdditionalLayer to add looping
of the hyperbolic layer during training, with input injection to prevent
representation collapse.

"""

from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from transformers import T5ForConditionalGeneration, T5Config

from typing import Tuple, Optional
import torch.nn as nn
import torch
from torch.nn import CrossEntropyLoss

from .hyperbolic_model_utils import HyperbolicLayer


def clamp_to_ball(x: torch.Tensor, max_norm: float = 0.9) -> torch.Tensor:
    """Clamp vectors to stay inside Poincaré ball"""
    norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    clamped = torch.where(
        norm > max_norm,
        x * max_norm / norm,
        x
    )
    return clamped


class T5ModelWithLoopedHyperbolic(T5ForConditionalGeneration):
    """
    T5 with looped hyperbolic layer.

    Same as T5ModelWithAdditionalLayer but loops the hyperbolic layer
    num_loops times with input injection.
    """

    def __init__(self,
                 layer_type: str,
                 model_name: str = 'google/t5-large-lm-adapt',
                 checkpoint_hyperbolic_knit5: str = None,
                 with_model_state_dict=True,
                 curvature: Optional[float] = 0.3,
                 gpu_parallelization=False,
                 soft_prompt_length=100,
                 num_loops: int = 4,
                 max_norm: float = 0.9,
                 input_injection: bool = True):
        config: T5Config = T5Config.from_pretrained(model_name)
        super(T5ModelWithLoopedHyperbolic, self).__init__(config=config)

        self.curvature = curvature
        self.soft_prompt_length = soft_prompt_length
        self.model_name = model_name
        self.num_loops = num_loops
        self.max_norm = max_norm
        self.input_injection = input_injection
        self.additional_layer_type = layer_type

        in_features = config.d_model

        print(f"Initializing T5ModelWithLoopedHyperbolic:")
        print(f"  - Layer type: {layer_type}")
        print(f"  - Num loops: {num_loops}")
        print(f"  - Max norm (clamp): {max_norm}")
        print(f"  - Input injection: {input_injection}")

        # Create the layer (same as original)
        if layer_type == 'linear':
            self.hyperbolic_layer = nn.Linear(in_features=in_features, out_features=in_features)
            print("Using Euclidean Additional Layer")
        elif layer_type == 'hyperbolic':
            self.hyperbolic_layer = HyperbolicLayer(
                curvature=self.curvature,
                type='poincare',
                scaled=False,
                learnable=True,
                in_features=in_features,
                out_features=in_features
            )
            print("Using Hyperbolic Additional Layer")
        elif layer_type == 'identity':
            self.hyperbolic_layer = nn.Identity()
            print("Using Identity (No Extra Layer)")
        else:
            raise ValueError(f"{layer_type} not supported. Only 'hyperbolic', 'linear', or 'identity'")

        # Learnable mixing weight for input injection
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # Load weights
        if checkpoint_hyperbolic_knit5 is None:
            print("Initializing T5 Model from pretrained...")
            pretrained_model = T5ForConditionalGeneration.from_pretrained(model_name)
            missing, unexpected = self.load_state_dict(pretrained_model.state_dict(), strict=False)
            print(f"Missing: {missing}")
            print(f"Unexpected: {unexpected}")
            del pretrained_model
        else:
            print(f"Loading Checkpoint from {checkpoint_hyperbolic_knit5}")
            checkpoint = torch.load(checkpoint_hyperbolic_knit5, map_location='cpu')

            if gpu_parallelization or True:  # Always try to strip module prefix
                if 'model_state_dict' in checkpoint:
                    checkpoint['model_state_dict'] = {
                        k.replace('module.', ''): v
                        for k, v in checkpoint['model_state_dict'].items()
                    }
                else:
                    checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}

            state_dict = checkpoint['model_state_dict'] if with_model_state_dict else checkpoint
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            print(f"Missing: {missing}")
            print(f"Unexpected: {unexpected}")

        self.post_init()
        self.model_parallel = False
        self.device_map = None

    def _apply_looped_hyperbolic(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply hyperbolic layer with looping and input injection.

        Args:
            hidden_states: Encoder output [batch, seq_len, hidden_dim]

        Returns:
            Processed hidden states
        """
        # Save original for input injection
        h_original = hidden_states
        h = hidden_states

        alpha = torch.sigmoid(self.alpha)

        for loop_idx in range(self.num_loops):
            # Apply hyperbolic layer
            h = self.hyperbolic_layer(h)

            # Clamp to stay inside ball (prevents gradient explosion)
            h = clamp_to_ball(h, self.max_norm)

            # Input injection: mix with original
            if self.input_injection:
                h = alpha * h + (1 - alpha) * h_original
                h = clamp_to_ball(h, self.max_norm)

        return h

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

        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask

        # Encode if needed
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        # Apply looped hyperbolic layer ===
        hidden_states = self._apply_looped_hyperbolic(hidden_states)

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

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

        # Decode
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
            **kwargs
        )

        sequence_output = decoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

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
                             soft_prompt=soft_prompt,
                             input_ids=input_ids,
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
                             return_dict=return_dict)
