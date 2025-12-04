from torch.utils.data import Dataset
import random
from tqdm import tqdm
import numpy as np
import sys
from src.config import Config
class C4Dataset(Dataset):
    """
    Gets the list of texts. 
    """
    def __init__(self, list_of_texts, tokenizer, corruption_rate=0.15, average_length_of_spans=3, objective = 'span_corruption'):
        if objective not in ['span_corruption', 'prefix_language_modeling']:
            raise ValueError(f'Unknown objective {objective}. Supported are [span_corruption, prefix_language_modeling]')
        
        
        self.config = Config()
        
        print(f'C4 Dataset with objective: {objective}')
        self.tokenizer = tokenizer
        self.corruption_rate = corruption_rate
        self.average_length_of_spans = average_length_of_spans
        self.objective = objective
        self.dataset = list_of_texts#self._cleanup_dataset(list_of_texts)

        
        
        
    def _cleanup_dataset(self, dataset): 
        cleaned_dataset = []     
        if self.objective == 'span_corruption':
            cleaned_dataset = [
                text for text in tqdm(dataset, desc=f'Cleanup Dataset: Remove texts with < {self.average_length_of_spans} corrupted tokens', file=sys.stdout, dynamic_ncols=True)
                if len(self.tokenizer.encode(text, add_special_tokens=False)) * self.corruption_rate >= self.average_length_of_spans
            ]
                
        else:
            cleaned_dataset = [
                text for text in tqdm(dataset, 
                                    desc=f"Cleanup Dataset: Remove texts with > {self.config.t5_model.tokenizer_max_length} tokens", 
                          file=sys.stdout) 
                    if len(self.tokenizer.encode(text, add_special_tokens=False)) <= self.config.t5_model.tokenizer_max_length
                ]
        count_removed = len(dataset) - len(cleaned_dataset)
        print(f'Cleaned {count_removed} Datapoints remaining {len(cleaned_dataset)} Datapoints')
        
        
        return cleaned_dataset
            
    def _span_corruption(self, text):
        input_ids = self.tokenizer(text, truncation=True, padding=True)['input_ids']
        
        total_tokens = len(input_ids)
        #print(f"Length of Total tokens: {total_tokens}")
        total_corrupted_tokens = int(total_tokens * self.corruption_rate)
        
        total_spans = total_corrupted_tokens // self.average_length_of_spans
        
        span_lengths = np.random.poisson(self.average_length_of_spans, total_spans)
        span_lengths = np.clip(span_lengths, 1, total_tokens // total_spans)
        
        total_corrupted_tokens = span_lengths.sum()
        if total_corrupted_tokens != int(total_tokens * self.corruption_rate):
            difference = int(total_tokens * self.corruption_rate) - total_corrupted_tokens
            if difference > 0:
                for i, current_length in enumerate(span_lengths):
                    if current_length < (total_tokens // total_spans):
                        span_lengths[i] += 1
                        total_corrupted_tokens += 1
                        if total_corrupted_tokens == int(total_tokens * self.corruption_rate):
                            break
            else:
                for i, current_length in enumerate(span_lengths):
                    if current_length > 1:
                        span_lengths[i] -= 1
                        total_corrupted_tokens -= 1
                        if total_corrupted_tokens == int(total_tokens * self.corruption_rate):
                            break          
                    
        #print(total_corrupted_tokens)
        #print(f" Span Lengths: {span_lengths}")
        
        span_starts = []
        current_position = 0
    
        for idx, length in enumerate(span_lengths):
            if current_position >= total_tokens:
                break
            # Ensure the span length does not exceed the remaining tokens
            length = min(length, total_tokens - current_position)
            start = np.random.randint(current_position, total_tokens - sum(span_lengths[idx:]) * 2 + 1)
            span_starts.append(start)
            current_position = start + length
        span_starts.sort()
        
        #print(f"Span Starts: {span_starts}")
        
        output_ids = input_ids.copy()
        corrupted_tokens = []
        sentinel_counter = 0
        sum_lengths = 0
        for start, length in zip(span_starts, span_lengths):
            #print(f'Input Sequence: {input_ids}')
            end = min(start + length, total_tokens)
            #print(f'(start, end): ({start}, {end})')
            sentinel_token = self.tokenizer.convert_tokens_to_ids([f'<extra_id_{sentinel_counter}>'])[0]
            #print(f'Sentinel Token ID: {sentinel_token}')
            corrupted_tokens.append(sentinel_token)
            corrupted_tokens.extend(input_ids[start:end])
            #print(f'Corrupted Tokens: {corrupted_tokens}')
            output_ids[start-sum_lengths:end-sum_lengths] = [sentinel_token]
            #print(f'Output IDs: {output_ids}')
            sentinel_counter += 1
            sum_lengths += length-1
        sentinel_token = self.tokenizer.convert_tokens_to_ids([f'<extra_id_{sentinel_counter}>'])[0]
        corrupted_tokens.append(sentinel_token)
        input_ids = output_ids
        target_ids = corrupted_tokens
        #print(f"Length of Target IDs: {len(target_ids)}")
        #print(f"Length of Input ids: {len(input_ids)}")
        
        input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        target_tokens = self.tokenizer.convert_ids_to_tokens(target_ids)
        
        input_sequence = self.tokenizer.convert_tokens_to_string(input_tokens)
        target_sequence = self.tokenizer.convert_tokens_to_string(target_tokens)
        
        return input_sequence, target_sequence, input_tokens, target_tokens
    def _prefix_language_modeling(self, text, seed=42):
        random.seed(seed)
        tokens = self.tokenizer.tokenize(text)
        split_point = random.randint(1, len(tokens)-1)
        input_tokens = tokens[:split_point]
        target_tokens = tokens[split_point:]
        
        input_sequence = self.tokenizer.convert_tokens_to_string(input_tokens)
        target_sequence = self.tokenizer.convert_tokens_to_string(target_tokens)
        return input_sequence, target_sequence
            
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        text = self.dataset[idx]
        if self.objective == 'span_corruption':
            input_sequence, target_sequence, _, _ = self._span_corruption(text)
            return input_sequence, target_sequence
        elif self.objective == 'prefix_language_modeling':
            input_sequence, decoder_sequence = self._prefix_language_modeling(text)
            return input_sequence, decoder_sequence