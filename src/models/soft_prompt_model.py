import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5Model
from src.utils.util import get_top_token_embeddings
# from ..config import Config
from .hyperbolic_t5_additional_layer import T5ModelWithAdditionalLayer
from typing import Union

class SoftPromptModel(nn.Module):
    def __init__(self,
                 knit5 : Union[T5Model, T5ModelWithAdditionalLayer],
                 model_name : str = '',
                 soft_prompt = None,
                 soft_prompt_length = 100,
                 use_soft_prompt = True):
        super(SoftPromptModel, self).__init__()
        self.knit5 = knit5
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#        if knit5_checkpoint_path is not None:
#           load_model_checkpoint(knit5, knit5_checkpoint_path, with_model_state_dict=with_model_state_dict, gpu_parallelization=gpu_parallelization)
        
        # self.config = Config()
        self.soft_prompt_length = soft_prompt_length
        self.use_soft_prompt = use_soft_prompt
        
        if soft_prompt is None:
            self.soft_prompt = self.init_soft_prompt()
        else:
            self.soft_prompt = soft_prompt

        for param in self.knit5.parameters():
            param.requires_grad = False
            

        # Remove if no learnable curvature
        if hasattr(self.knit5, 'hyperbolic_layer'):
            print(f"{self.knit5.additional_layer_type = }")
            if self.knit5.additional_layer_type != 'identity':
                print(f"Setting All parameters of hyperbolic layer to True")
                for param in self.knit5.hyperbolic_layer.parameters():
                    param.requires_grad = True

        print(f"Hyperbolic Layer learnable in soft prompt model class: {all(param.requires_grad for param in self.knit5.hyperbolic_layer.parameters()) if self.knit5.additional_layer_type != 'identity' else False}")
        
        self.soft_prompt.requires_grad = True

        
    def init_soft_prompt(self):
        tokenizer = AutoTokenizer.from_pretrained(self.knit5.model_name)
        #HP Soft Prompt will be tuned
        pp_length = self.soft_prompt_length
    
        pp_embedding_size = self.knit5.config.hidden_size
        pp_embeddings = torch.randn(pp_length, pp_embedding_size)
        
        #dont use random use top 100 most common tokens of tokenizer.getvocab
        top_k_token_embeddings = get_top_token_embeddings(self.knit5, tokenizer, pp_length)
        pp_embeddings[:top_k_token_embeddings.size(0), :] = top_k_token_embeddings

        #pp_embeddings = torch.cat([torch.randn(pp_length,1), pp_embeddings], dim = -1)
        print(f"Initialized Embeddings with Shape: {pp_embeddings.shape}")
        print(f"Initializing Soft Prompt with top {pp_length} tokens from pretraining corpus")
        return nn.Parameter(pp_embeddings)
    
    def _forward_soft_prompt_only_enc(self, input_ids, attention_mask, labels, **forward_kwargs):
        soft_prompt_input = self.soft_prompt.unsqueeze(0).expand(input_ids.size(0), -1, -1).to(self.device)

        input_embeddings = self.knit5.shared(input_ids)  # Convert input IDs to embeddings
        
        concatenated_embeddings = torch.cat([soft_prompt_input, input_embeddings], dim = 1)
        #Adjust attention mask (take all of the soft prompt tokens should be attented)
        soft_prompt_attention_mask = torch.ones((attention_mask.size(0), soft_prompt_input.size(1)), device=self.device)
        

        concatenated_attention_mask = torch.cat([soft_prompt_attention_mask, attention_mask], dim=1)
        outputs = self.knit5(inputs_embeds=concatenated_embeddings, attention_mask=concatenated_attention_mask, labels = labels, soft_prompt=soft_prompt_input, **forward_kwargs)
        return outputs

    def forward(self, input_ids, attention_mask, labels, **forward_kwargs):
        return self._forward_soft_prompt_only_enc(input_ids, attention_mask, labels, **forward_kwargs)
    
    
    def generate(self, input_ids, attention_mask, **generate_kwargs):
        soft_prompt_input = self.soft_prompt.unsqueeze(0).expand(input_ids.size(0), -1, -1).to(self.device)

        input_embeddings = self.knit5.shared(input_ids)  # Convert input IDs to embeddings
        
        concatenated_embeddings = torch.cat([soft_prompt_input, input_embeddings], dim = 1)
        #Adjust attention mask (take all of the soft prompt tokens should be attented)
        soft_prompt_attention_mask = torch.ones((attention_mask.size(0), soft_prompt_input.size(1)), device=self.device)
        

        concatenated_attention_mask = torch.cat([soft_prompt_attention_mask, attention_mask], dim=1)

        outputs = self.knit5.generate(inputs_embeds=concatenated_embeddings, attention_mask=concatenated_attention_mask, **generate_kwargs)
        return outputs