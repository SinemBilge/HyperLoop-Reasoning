from typing import Optional, Tuple
import torch

class ModelOutput():
    
    def __init__(self, last_hidden_state: torch.FloatTensor = None,
        hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None,
        attentions: Optional[Tuple[torch.FloatTensor, ...]] = None,       
        loss: Optional[torch.FloatTensor] = None,
        logits: torch.FloatTensor = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None,
        decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None,
        cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None,
        encoder_last_hidden_state: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None,
        encoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None):
            self.last_hidden_state: torch.FloatTensor = last_hidden_state
            self.hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = hidden_states
            self.attentions: Optional[Tuple[torch.FloatTensor, ...]] = attentions
            
            
            self.loss: Optional[torch.FloatTensor] = loss
            self.logits: torch.FloatTensor = logits
            self.past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = past_key_values
            self.decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = decoder_hidden_states
            self.decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = decoder_attentions
            self.cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = cross_attentions
            self.encoder_last_hidden_state: Optional[torch.FloatTensor] = encoder_last_hidden_state
            self.encoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = encoder_hidden_states
            self.encoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = encoder_attentions
        