import torch
import torch.nn as nn
from typing import Optional, List, Dict
from src.models import T5ModelWithAdditionalLayer, SoftPromptModel


class LoopedHyperbolicReasoner(nn.Module):
    def __init__(
        self,
        parsing_model: SoftPromptModel,
        hopping_model: SoftPromptModel,
        num_loops: int = 2,
        aggregation_method: str = 'last',
        use_feedback: bool = True,
        device: str = 'cuda'
    ):
        super(LoopedHyperbolicReasoner, self).__init__()

        self.parsing_model = parsing_model
        self.hopping_model = hopping_model
        self.num_loops = num_loops
        self.aggregation_method = aggregation_method
        self.use_feedback = use_feedback
        self.device = device

        for param in self.parsing_model.parameters():
            param.requires_grad = False
        for param in self.hopping_model.parameters():
            param.requires_grad = False

        if aggregation_method == 'weighted_avg':
            self.loop_weights = nn.Parameter(torch.ones(num_loops) / num_loops)
        elif aggregation_method == 'attention':
            hidden_size = hopping_model.knit5.config.hidden_size
            self.attention_layer = nn.Linear(hidden_size, 1)

    def parse_question(self, input_ids, attention_mask, tokenizer):
        with torch.no_grad():
            outputs = self.parsing_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128,
                num_beams=5,
                early_stopping=True
            )

        parsed_paths = [tokenizer.decode(output, skip_special_tokens=True)
                       for output in outputs]

        return outputs, parsed_paths

    def single_hop(self, input_ids, attention_mask, loop_idx: int):
        outputs = self.hopping_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=256,
            num_beams=5,
            early_stopping=True,
            output_hidden_states=True,
            return_dict_in_generate=True
        )

        return outputs

    def aggregate_loop_outputs(self, loop_outputs: List[torch.Tensor]) -> torch.Tensor:
        if self.aggregation_method == 'last':
            return loop_outputs[-1]

        elif self.aggregation_method == 'weighted_avg':
            weights = torch.softmax(self.loop_weights, dim=0)
            stacked = torch.stack(loop_outputs, dim=0)
            aggregated = torch.sum(stacked * weights.view(-1, 1, 1), dim=0)
            return aggregated

        elif self.aggregation_method == 'attention':
            stacked = torch.stack(loop_outputs, dim=0)
            attention_scores = self.attention_layer(stacked)
            attention_weights = torch.softmax(attention_scores, dim=0)
            aggregated = torch.sum(stacked * attention_weights, dim=0)
            return aggregated

        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

    def forward(self, input_ids, attention_mask, tokenizer):
        parsed_output_ids, parsed_paths = self.parse_question(
            input_ids, attention_mask, tokenizer
        )

        loop_outputs = []
        current_input_ids = parsed_output_ids
        current_attention_mask = torch.ones_like(parsed_output_ids)

        intermediate_results = {
            'parsed_path': parsed_paths,
            'loop_outputs': []
        }

        for loop_idx in range(self.num_loops):
            hop_outputs = self.single_hop(
                input_ids=current_input_ids,
                attention_mask=current_attention_mask,
                loop_idx=loop_idx
            )

            loop_outputs.append(hop_outputs.sequences)

            decoded = [tokenizer.decode(seq, skip_special_tokens=True)
                      for seq in hop_outputs.sequences]
            intermediate_results['loop_outputs'].append({
                'loop_idx': loop_idx,
                'decoded_paths': decoded,
                'sequences': hop_outputs.sequences
            })

            if self.use_feedback and loop_idx < self.num_loops - 1:
                current_input_ids = hop_outputs.sequences
                current_attention_mask = torch.ones_like(hop_outputs.sequences)

        final_output = self.aggregate_loop_outputs(loop_outputs)

        return final_output, intermediate_results

    def evaluate(self, input_ids, attention_mask, tokenizer):
        with torch.no_grad():
            final_output, intermediate_results = self.forward(
                input_ids, attention_mask, tokenizer
            )

            final_decoded = [tokenizer.decode(seq, skip_special_tokens=True)
                           for seq in final_output]

            return final_decoded, intermediate_results


def create_looped_model(
    knit5_checkpoint_path: str,
    parsing_prompt_checkpoint_path: str,
    hopping_prompt_checkpoint_path: str,
    additional_layer_type: str = 'hyperbolic',
    curvature: float = 0.37,
    num_loops: int = 2,
    aggregation_method: str = 'last',
    device: str = 'cuda'
):
    from src.config import Config

    config = Config()
    GPU_PARALLELIZATION = True
    WITH_MODEL_STATE_DICT = GPU_PARALLELIZATION

    parsing_knit5 = T5ModelWithAdditionalLayer(
        layer_type=additional_layer_type,
        curvature=curvature,
        checkpoint_hyperbolic_knit5=knit5_checkpoint_path,
        with_model_state_dict=WITH_MODEL_STATE_DICT,
        gpu_parallelization=GPU_PARALLELIZATION,
        soft_prompt_length=config.random_walk_training.prompt_length
    )

    parsing_checkpoint = torch.load(parsing_prompt_checkpoint_path, map_location=device)
    parsing_prompt = nn.Parameter(parsing_checkpoint['soft_prompt_state_dict'], requires_grad=False)
    parsing_knit5.hyperbolic_layer.load_state_dict(parsing_checkpoint['additional_linear_layer'])

    parsing_model = SoftPromptModel(knit5=parsing_knit5, soft_prompt=parsing_prompt)

    hopping_knit5 = T5ModelWithAdditionalLayer(
        layer_type=additional_layer_type,
        curvature=curvature,
        checkpoint_hyperbolic_knit5=knit5_checkpoint_path,
        with_model_state_dict=WITH_MODEL_STATE_DICT,
        gpu_parallelization=GPU_PARALLELIZATION,
        soft_prompt_length=config.random_walk_training.prompt_length
    )

    hopping_checkpoint = torch.load(hopping_prompt_checkpoint_path, map_location=device)
    hopping_prompt = nn.Parameter(hopping_checkpoint['soft_prompt_state_dict'], requires_grad=False)
    hopping_knit5.hyperbolic_layer.load_state_dict(hopping_checkpoint['additional_linear_layer'])

    hopping_model = SoftPromptModel(knit5=hopping_knit5, soft_prompt=hopping_prompt)

    looped_model = LoopedHyperbolicReasoner(
        parsing_model=parsing_model,
        hopping_model=hopping_model,
        num_loops=num_loops,
        aggregation_method=aggregation_method,
        use_feedback=True,
        device=device
    )

    return looped_model


if __name__ == '__main__':
    from transformers import AutoTokenizer

    knit5_path = "/checkpoints/wikimultihop/knowledge_integration/knit5.pth"
    parsing_path = "/checkpoints/stage2_parsing/parsing_checkpoint.pth"
    hopping_path = "/checkpoints/stage2_parsing/hopping_checkpoint.pth"

    model = create_looped_model(
        knit5_checkpoint_path=knit5_path,
        parsing_prompt_checkpoint_path=parsing_path,
        hopping_prompt_checkpoint_path=hopping_path,
        num_loops=3,
        aggregation_method='last',
        device='cuda'
    )

    tokenizer = AutoTokenizer.from_pretrained('google/t5-large-lm-adapt')

    question = "Which country is the composer of the song Cloudburst from?"
    inputs = tokenizer(question, return_tensors='pt', padding=True)

    predictions, intermediate = model.evaluate(
        input_ids=inputs['input_ids'].to('cuda'),
        attention_mask=inputs['attention_mask'].to('cuda'),
        tokenizer=tokenizer
    )

    print("Parsed path:", intermediate['parsed_path'])
    for i, loop_result in enumerate(intermediate['loop_outputs']):
        print(f"Loop {i+1}:", loop_result['decoded_paths'])
    print("Final answer:", predictions)
