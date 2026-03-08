from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from transformers import T5ForConditionalGeneration, T5Config

from typing import Tuple, Optional
import torch.nn as nn
import torch
from torch.nn import CrossEntropyLoss

from .hyperbolic_model_utils import *


class T5ModelWithAdditionalLayer(T5ForConditionalGeneration):
    def __init__(self,
                 layer_type: str,
                 model_name: str = "google/t5-large-lm-adapt",
                 checkpoint_hyperbolic_knit5: str = None,
                 with_model_state_dict: bool = True,
                 curvature: Optional[float] = 0.3,
                 num_layers: int = 1,
                 num_loop_iterations: int = 1):
        r"""
        params:
        model_name: Pretrained Model name.
        checkpoint_hyperbolic_knit5: If finetuned on knowledge integration give the checkpoint path
        curvature: curvature for exponential mapping
        """
        config: T5Config = T5Config.from_pretrained(model_name)
        super().__init__(config=config)

        self.curvature = curvature
        in_features = config.d_model
        self.model_name = model_name

        self.additional_layer_type = layer_type
        if layer_type not in ["euclidean", "hyperbolic", "identity"]:
            raise ValueError(
                f"{layer_type} not supported. Only 'hyperbolic' or 'euclidean' or 'identity'"
            )

        # ----- Build additional layer(s) -----
        if num_layers > 1:
            if layer_type in ["euclidean", "hyperbolic"]:
                layers = []
                for i in range(num_layers):
                    input_features = in_features if i == 0 else 1024 * 6
                    hidden_features = in_features if i == (num_layers - 1) else 1024 * 6
                    if layer_type == "euclidean":
                        layers.append(
                            nn.Linear(
                                in_features=input_features,
                                out_features=hidden_features,
                            )
                        )
                    else:
                        layers.append(
                            HyperbolicLayer(
                                curvature=self.curvature,
                                type="poincare",
                                scaled=False,
                                learnable=True,
                                in_features=input_features,
                                out_features=hidden_features,
                            )
                        )
                    layers.append(nn.ReLU())

                self.additional_layer = nn.Sequential(*layers)
                print(f"Using {layer_type.capitalize()} Additional Layer")
            elif layer_type == "identity":
                self.additional_layer = nn.Identity()
                print("Using Identity (No Extra Layer)")
        else:
            if layer_type == "euclidean":
                self.additional_layer = nn.Linear(
                    in_features=in_features, out_features=in_features
                )
                print("Using Euclidean Additional Layer")
            elif layer_type == "hyperbolic":
                self.additional_layer = HyperbolicLayer(
                    curvature=self.curvature,
                    type="poincare",
                    scaled=False,
                    learnable=True,
                    in_features=in_features,
                    out_features=in_features,
                )
                print("Using Hyperbolic Additional Layer")
            elif layer_type == "identity":
                self.additional_layer = nn.Identity()
                print("Using Identity (No Extra Layer)")

        # ----- Wrap in LoopedHyperbolicLayer if num_loop_iterations > 1 -----
        if num_loop_iterations > 1 and layer_type == "hyperbolic":
            self.additional_layer = LoopedHyperbolicLayer(
                self.additional_layer,
                num_iterations=num_loop_iterations,
                hidden_dim=in_features,
            )
            print(f"Wrapping hyperbolic layer with {num_loop_iterations} loop iterations")

        # ----- Load base T5 weights (pretrained or finetuned checkpoint) -----
        if checkpoint_hyperbolic_knit5 is None:
            print("Initializing T5 Model from pretrained weights...")
            pretrained_model = T5ForConditionalGeneration.from_pretrained(model_name)
            missing, unexpected = self.load_state_dict(
                pretrained_model.state_dict(), strict=False
            )
            print(f"Missing: {missing}")
            print(f"Unexpected: {unexpected}")
            del pretrained_model
        else:
            print(f"Loading Checkpoint from {checkpoint_hyperbolic_knit5}")
            checkpoint = torch.load(checkpoint_hyperbolic_knit5, map_location="cpu")

            # Extract model state dict
            if with_model_state_dict and isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint

            # Handle potential 'module.' prefix from DDP/DataParallel
            if any(k.startswith("module.") for k in state_dict.keys()):
                state_dict = {
                    k.replace("module.", "", 1): v for k, v in state_dict.items()
                }

            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            print(f"Missing: {missing}")
            print(f"Unexpected: {unexpected}")

        self.post_init()
        # Model parallel (unchanged from HF)
        self.model_parallel = False
        self.device_map = None

    # ------- your forward logic, unchanged except for device-agnosticism -------

    def _forward_after_encoder(
        self,
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
        **kwargs,
    ):

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
        hidden_states = self.additional_layer(hidden_states)

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

        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            labels = labels.to(lm_logits.device)
            loss = loss_fct(
                lm_logits.view(-1, lm_logits.size(-1)),
                labels.view(-1),
            )

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
    

    def forward(self, **kwargs):
        return self._forward_after_encoder(**kwargs)
