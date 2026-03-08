import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5Model
from src.utils.util import get_top_token_embeddings
from .hyperbolic_t5_additional_layer import T5ModelWithAdditionalLayer
from typing import Union


class SoftPromptModel(nn.Module):
    def __init__(self,
                 knit5: Union[T5Model, T5ModelWithAdditionalLayer],
                 model_name: str = "",
                 soft_prompt=None,
                 soft_prompt_length: int = 100,
                 use_soft_prompt: bool = True):
        super().__init__()
        self.knit5 = knit5
        self.model_name = model_name

        self.soft_prompt_length = soft_prompt_length
        self.use_soft_prompt = use_soft_prompt

        if soft_prompt is None:
            self.soft_prompt = self.init_soft_prompt()
        else:
            self.soft_prompt = soft_prompt

        # Freeze base model
        for param in self.knit5.parameters():
            param.requires_grad = False

        # Make hyperbolic additional layer trainable (if present)
        if hasattr(self.knit5, "additional_layer"):
            if self.knit5.additional_layer_type != "identity":
                print("Setting all parameters of additional_layer to requires_grad=True")
                for param in self.knit5.additional_layer.parameters():
                    param.requires_grad = True

        print(
            "Hyperbolic Layer learnable in soft prompt model class:",
            all(
                param.requires_grad
                for param in self.knit5.additional_layer.parameters()
            )
            if getattr(self.knit5, "additional_layer_type", "identity") != "identity"
            else False,
        )

        self.soft_prompt.requires_grad = True

    def init_soft_prompt(self):
        tokenizer = AutoTokenizer.from_pretrained(self.knit5.model_name)
        pp_length = self.soft_prompt_length
        pp_embedding_size = self.knit5.config.hidden_size

        pp_embeddings = torch.randn(pp_length, pp_embedding_size)

        # use top-k token embeddings instead of random for first part
        top_k_token_embeddings = get_top_token_embeddings(self.knit5, tokenizer, pp_length)
        pp_embeddings[: top_k_token_embeddings.size(0), :] = top_k_token_embeddings

        print(f"Initialized Soft Prompt Embeddings with shape: {pp_embeddings.shape}")
        print(f"Initializing Soft Prompt with top {pp_length} tokens from pretraining corpus")
        return nn.Parameter(pp_embeddings)

    def _forward_soft_prompt_only_enc(self, input_ids, attention_mask, labels, **forward_kwargs):
        """
        Only encoder side is prepended with soft-prompt tokens.
        NOTE: No hard-coded device; everything follows input_ids' device.
        """
        device = input_ids.device

        # (B, L_soft, H)
        soft_prompt_input = self.soft_prompt.unsqueeze(0).expand(
            input_ids.size(0), -1, -1
        ).to(device)

        # Convert input IDs to embeddings (knit5 shared embedding matrix)
        input_embeddings = self.knit5.shared(input_ids)

        # (B, L_soft + L_in, H)
        concatenated_embeddings = torch.cat(
            [soft_prompt_input, input_embeddings], dim=1
        )

        # Adjust attention mask (soft prompt tokens are all attended)
        soft_prompt_attention_mask = torch.ones(
            (attention_mask.size(0), soft_prompt_input.size(1)), device=device
        )
        concatenated_attention_mask = torch.cat(
            [soft_prompt_attention_mask, attention_mask], dim=1
        )

        outputs = self.knit5(
            inputs_embeds=concatenated_embeddings,
            attention_mask=concatenated_attention_mask,
            labels=labels,
            **forward_kwargs,
        )
        return outputs

    def forward(self, input_ids, attention_mask, labels, **forward_kwargs):
        return self._forward_soft_prompt_only_enc(
            input_ids, attention_mask, labels, **forward_kwargs
        )

    def generate(self, input_ids, attention_mask, **generate_kwargs):
        """
        Generation with soft prompts. Again, all devices are inferred from inputs.
        """
        device = input_ids.device

        soft_prompt_input = self.soft_prompt.unsqueeze(0).expand(
            input_ids.size(0), -1, -1
        ).to(device)

        input_embeddings = self.knit5.shared(input_ids)
        concatenated_embeddings = torch.cat(
            [soft_prompt_input, input_embeddings], dim=1
        )

        soft_prompt_attention_mask = torch.ones(
            (attention_mask.size(0), soft_prompt_input.size(1)), device=device
        )
        concatenated_attention_mask = torch.cat(
            [soft_prompt_attention_mask, attention_mask], dim=1
        )

        outputs = self.knit5.generate(
            inputs_embeds=concatenated_embeddings,
            attention_mask=concatenated_attention_mask,
            **generate_kwargs,
        )
        return outputs
