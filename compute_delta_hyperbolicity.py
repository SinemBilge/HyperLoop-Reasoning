from src.utils.util import load_dataset
from src.utils.trainer_utils import load_model_checkpoint
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.models import T5ModelWithAdditionalLayer
import torch
from src.config import Config
from torch.utils.data import DataLoader
from src.knowledge_graph import create_knowledge_graph_wikimultihop
import matplotlib.pyplot as plt
import sys
import argparse
def plot_layer_hyperbolicity(layer_hyperbolicity_dict, title, save_path = None,):
    layers = list(layer_hyperbolicity_dict.keys())
    means = [layer_hyperbolicity_dict[layer]['mean'] for layer in layers]
    std_devs = [layer_hyperbolicity_dict[layer]['std'] for layer in layers]
    
    # Create the error plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(layers, means, yerr=std_devs, fmt='o', capsize=5, label='Mean ± Std')
    
    # Adding labels and title
    plt.xlabel('Layer')
    plt.ylabel('Mean Hyperbolicity')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    
    # Show the plot
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
def _comput_delta_hyperbolicity(dataloader, model : T5ModelWithAdditionalLayer, tuned, parse, sentence=True):
    from tqdm import tqdm
    from src.utils.delta import batched_delta_hyp
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    model.to(device)
    layer_embedding_list = []

    with torch.no_grad():
        for incomplete_paths, complete_paths in tqdm(dataloader, file=sys.stdout):
            # Tokenize the batch with offsets for the input texts.
            inputs = tokenizer(
                incomplete_paths,
                padding=True,
                truncation=True,
                return_tensors="pt",
                return_offsets_mapping=True
            )
            offset_mappings = inputs.offset_mapping  # Shape: (batch_size, seq_len, 2)
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)
            
            # Tokenize the complete paths (labels) as needed.
            labels = tokenizer(
                complete_paths,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            label_input_ids = labels.input_ids.to(device)
            
            # Forward pass through the model (request hidden states)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                labels=label_input_ids
            )
            # Extract token embeddings from the last hidden state.
            # Shape: (batch_size, seq_len, hidden_size)
            token_embeddings = outputs.encoder_hidden_states[-1]
            # mask = attention_mask.unsqueeze(-1).float()  # shape: (batch_size, seq_len, 1)

            # Multiply token embeddings by the mask so that padding tokens (mask=0) are zeroed out
            # masked_embeddings = token_embeddings * mask

            # # Sum the embeddings along the sequence dimension and divide by the sum of the mask (number of tokens)
            # sentence_embeddings = masked_embeddings.sum(dim=1) / mask.sum(dim=1)
            # layer_embedding_list.append(sentence_embeddings.cpu().detach())
            # sentence_embeddings now has shape: (batch_size, hidden_size)
            batch_size = input_ids.shape[0]
            for i in range(batch_size):
                if parse and not sentence:
                    # print(f"Question: {incomplete_paths[i]}")
                    tokens = tokenizer.convert_ids_to_tokens(input_ids[i])
                    embeddings = token_embeddings[i]  # (seq_len, hidden_size)
                    mask = attention_mask[i].bool()
                    current_word_embeds = []  # Holds token embeddings for the current word.
                    current_word = ""         # Accumulates the actual word string.
                    
                    for j, token in enumerate(tokens):
                        if not mask[j]:
                            continue  # Skip padded tokens.
                        if token == tokenizer.pad_token:
                            continue  # Explicitly skip pad tokens.
                        
                        # Process each token inside the loop.
                        if token.startswith("▁"):
                            # If there is an existing word, finalize it.
                            if current_word_embeds:
                                aggregated = torch.stack(current_word_embeds, dim=0).mean(dim=0)
                                # print("Word:", current_word, "Shape:", aggregated.shape)
                                layer_embedding_list.append(aggregated.cpu().detach())
                                current_word_embeds = []  # Reset for the new word.
                            # Start a new word (remove the T5 boundary marker).
                            current_word = token.lstrip("▁")
                            current_word_embeds.append(embeddings[j])
                        else:
                            # Continuation of the current word.
                            current_word += token  # Optionally add a space if needed.
                            current_word_embeds.append(embeddings[j])
                    
                    # Finalize the last word in this sample if any tokens remain.
                    if current_word_embeds:
                        aggregated = torch.stack(current_word_embeds, dim=0).mean(dim=0)
                        # print("Word:", current_word, "Shape:", aggregated.shape)
                        layer_embedding_list.append(aggregated.cpu().detach())
                elif not parse and not sentence:
                    text = incomplete_paths[i]
                   # Split the text into segments based on the delimiter
                    segments = [seg.strip() for seg in text.split(";")]
                    offset_mapping_i = offset_mappings[i].tolist()  # List of (start, end) for sample i
                    token_embeddings_i = token_embeddings[i]  # Shape: (seq_len, hidden_size)
                   
                    component_embeddings = []
                    for seg in segments:
                        print("Processing segment:", seg)
                        start_idx = text.find(seg)
                        if start_idx == -1:
                           print(f"Segment not found: {seg} in sample index {i}")
                           continue
                        end_idx = start_idx + len(seg)
                        
                        #Use overlapping condition:
                        token_indices = [
                           j for j, (s, e) in enumerate(offset_mapping_i)
                           if (e > start_idx and s < end_idx)
                        ]
                        
                        if token_indices:
                            seg_embedding = token_embeddings_i[token_indices].mean(dim=0)
                            print("Segment:", seg, "has token indices:", token_indices, "and embedding shape:", seg_embedding.shape)
                            component_embeddings.append(seg_embedding)
                        else:
                           print(f"No tokens found for segment: '{seg}' in sample index {i}")
                
                    layer_embedding_list.extend([component.cpu().detach() for component in component_embeddings])
                elif sentence:
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

                    # Sum the embeddings along the sequence dimension while considering the mask
                    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)

                    # Sum the mask to get the count of non-padding tokens for each input
                    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

                    # Compute the mean to obtain pooled embeddings with shape: (batch_size, hidden_size)
                    pooled_embeddings = sum_embeddings / sum_mask

                    # If you need to store each pooled embedding in a list and then stack them:
                    layer_embedding_list = []
                    for i in range(pooled_embeddings.size(0)):
                        layer_embedding_list.append(pooled_embeddings[i].cpu().detach())
                    


        
    output_embeddings = torch.stack(layer_embedding_list, dim=0)
    mean, std, c = batched_delta_hyp(output_embeddings, batch_size=600, n_tries=15)
    dataloader_string = dataloader_string = "Parse" if parse else "Random Walk" if random_walk else "Knowledge Integration" if knowledge_integration else ""
    tuned_string = "Knit5" if tuned else "T5"
    print(f"Done!")
    print(f"Delta Hyperbolicity on {dataloader_string} Dataset of {tuned_string} Model")
    print(f"{mean = }")
    print(f"{std = }")
    print(f"{c = }")


    
    #plot_layer_hyperbolicity(layer_hyperbolicity_dict, save_path = f"delta_hyperbolicity_{dataloader_string}_{tuned_string}.png", title=f'Delta Hyperbolicity on {dataloader_string} Dataset of {tuned_string} Model')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Random Walk Training Training')
    parser.add_argument('--dataset_type', type=str, choices=['knowledge_integration', 'random_walk', 'parse'], required=True)
    parser.add_argument('--tuned', action='store_true', help='Takes Knit5')
    parser.add_argument('--dataset', type=str, help='choose dataset')
    parser.add_argument('--model_checkpoint', type=str, help='model checkpoint path')

    args = parser.parse_args()
    dataset_type = args.dataset_type
    tuned = args.tuned
    random_walk = dataset_type == 'random_walk'
    parse = dataset_type == 'parse'
    knowledge_integration = dataset_type == 'knowledge_integration'
    dataset = args.dataset
    model_checkpoint_path = args.model_checkpoint
    dataset_string = "Parse" if parse else "Random Walk"
    tuned_string = "Tuned" if tuned else "Not Tuned"
    config = Config()
    print(f"Start {tuned_string} {dataset_string} on {dataset}")

    #model_checkpoint_path = 'checkpoints/metaqa/knowledge_integration/Jan04_23-55-58_AdaFactor_0.001_-0.8362570675638017_knowledge_integration_bsize64_lr0.001_max_answers_1/knit5.pth'    
    #model_checkpoint_path = 'checkpoints/metaqa/knowledge_integration/Jan04_23-55-58_AdaFactor_0.001_-0.8362570675638017_knowledge_integration_bsize64_lr0.001_max_answers_1/knit5.pth'
    #model_checkpoint_path = 'checkpoints/mlpq/knowledge_integration/Jan05_10-42-23_AdaFactor_0.001_knowledge_integration_bsize64_lr0.001/knit5.pth'
    model_name = config.t5_model.model_name
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Loading Model...")
    from src.models import T5ModelWithAdditionalLayer
    knit5_model = T5ModelWithAdditionalLayer(layer_type='identity')
    import torch.nn as nn
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    knit5_model.to(device)
    GPU_PARALLELIZATION = False if dataset in ['2wikimultihop', 'wikimultihop', '2wikihop', 'wikihop'] else True
    WITH_MODEL_STATE_DICT = GPU_PARALLELIZATION
    if tuned:
        load_model_checkpoint(knit5_model, model_checkpoint_path, with_model_state_dict=WITH_MODEL_STATE_DICT, gpu_parallelization=GPU_PARALLELIZATION)

    from src.knowledge_graph import create_knowledge_graph_metaqa, create_knowledge_graph_wikimultihop, create_knowledge_graph_pql
    from src.utils.util import load_train_test_pql_dataset

   
    from src.datasets import RandomWalkMetaQADataset, ParseMetaQADataset, RandomWalkWikiHopDataset, RandomWalkPQLDataset
    from src.datasets import RandomWalkMLPQDataset, ParseMLPQDataset, ParseWikHopDataset
    from src.knowledge_graph import create_knowledge_graph_mlpq
    from torch.utils.data import ConcatDataset
    if random_walk:
        if dataset in ['2wikimultihop', 'wikimultihop', '2wikihop', 'wikihop']:
            train_dataset, dev_dataset, test_dataset, kg_train, kg_dev, kg_test = load_dataset('dataset/2wikimultihop', do_correct_wrong_evidences=True)

            all_data = pd.concat([train_dataset, dev_dataset, test_dataset])
            all_kg = create_knowledge_graph_wikimultihop(all_data)

            print(f"Nodes in Data: {len(list(all_kg.nodes()))}")

            print(f"Lenght Train Data: {len(train_dataset)}")
            print(f"Lenght Dev Data: {len(dev_dataset)}")
            print(f"Lenght Test Data: {len(test_dataset)}")

            random_walk_train = RandomWalkWikiHopDataset(all_kg, dev_dataset, test_dataset, steps=3, type='train')
            random_walk_dev = RandomWalkWikiHopDataset(all_kg, dev_dataset, test_dataset, steps=3, type='dev')
            random_walk_test = RandomWalkWikiHopDataset(all_kg, dev_dataset, test_dataset, steps=3, type='test')


        elif dataset in ['metaqa']:
            # df_kg = pd.read_csv("dataset/metaqa/kb.txt", sep="|")
            # kg = create_knowledge_graph_metaqa(df_kg, from_kb=True)

            df_dev = pd.read_json("dataset/metaqa/2hops/qa_dev_evidences.json")
            df_train = pd.read_json("dataset/metaqa/2hops/qa_train_evidences.json")
            df_test = pd.read_json("dataset/metaqa/2hops/qa_test_evidences.json")
            MAX_ANSWER = 1
            df_kg = pd.concat([df_dev, df_train, df_test])
            kg = create_knowledge_graph_metaqa(df_kg, from_kb=False, max_answers=MAX_ANSWER)
            random_walk_train = RandomWalkMetaQADataset(kg, df_dev, df_test, steps=3, type='train')
            random_walk_dev = RandomWalkMetaQADataset(kg, df_dev, df_test, steps=3, type='dev')
            random_walk_test = RandomWalkMetaQADataset(kg, df_dev, df_test, steps=3, type='test')
        elif dataset in ['mlpq']:
            #txt_file_paths = ['dataset/mlpq/Triples_in_questions/EN_KG', 'dataset/mlpq/Triples_in_questions/FR_KG']
            train_dataframe = pd.read_json('dataset/mlpq/Questions/fr-en/2-hop/2hop_train_question_evidences.json', lines=True)
            validation_dataframe = pd.read_json('dataset/mlpq/Questions/fr-en/2-hop/2hop_dev_question_evidences.json', lines=True)
            test_dataframe = pd.read_json('dataset/mlpq/Questions/fr-en/2-hop/2hop_test_question_evidences.json', lines=True)

            df_kg = pd.concat([train_dataframe, validation_dataframe, test_dataframe])
            kg = create_knowledge_graph_mlpq(df_kg, from_kb = False)

            random_walk_train = RandomWalkMLPQDataset(kg, validation_dataframe, test_dataframe, steps=3, type='train')
            random_walk_dev = RandomWalkMLPQDataset(kg, validation_dataframe, test_dataframe, steps=3, type='dev')
            random_walk_test = RandomWalkMLPQDataset(kg, validation_dataframe, test_dataframe, steps=3, type='test')
        elif dataset in ['pql']:
            file_path = "dataset/pathquestion/PQ-2H.txt"
            train, val, test = load_train_test_pql_dataset(file_path, random_state = 789)
            file_path_kb = "dataset/pathquestion/2H-kb.txt"
            kg = create_knowledge_graph_pql(file_path, from_kb=False)

            random_walk_train = RandomWalkPQLDataset(kg, val, test, steps=3, type='train')
            random_walk_dev = RandomWalkPQLDataset(kg, val, test, steps=3, type='dev')
            random_walk_test = RandomWalkPQLDataset(kg, val, test, steps=3, type='test')
        elif dataset in ['pq-3hop']:
            file_path = "dataset/pathquestion/PQ-3H.txt"
            train, val, test = load_train_test_pql_dataset(file_path, random_state = 789)
            # file_path_kb = "dataset/pathquestion/2H-kb.txt"
            kg = create_knowledge_graph_pql(file_path, from_kb=False, hops=3)

            random_walk_train = RandomWalkPQLDataset(kg, val, test, steps=4, type='train')
            #random_walk_test = RandomWalkPQLDataset(kg, val, test, steps=3, type='test')
            random_walk_dev = RandomWalkPQLDataset(kg, val, test, steps=4, type='dev')
            
            #from torch.utils.data import ConcatDataset
            
            #random_walk_dev = ConcatDataset([random_walk_dataloader_dev, random_walk_test])
        else:
            raise ValueError(f"Unknown Dataset")
        
        combined_dataset = ConcatDataset([random_walk_train])
    elif parse:
        if dataset in ['2wikimultihop', 'wikimultihop', '2wikihop', 'wikihop']:
            train_dataset, dev_dataset, test_dataset, kg_train, kg_dev, kg_test = load_dataset('dataset/2wikimultihop', do_correct_wrong_evidences=True)

            all_data = pd.concat([train_dataset, dev_dataset, test_dataset])
            all_kg = create_knowledge_graph_wikimultihop(all_data)

            print(f"Nodes in Data: {len(list(all_kg.nodes()))}")

            print(f"Lenght Train Data: {len(train_dataset)}")
            print(f"Lenght Dev Data: {len(dev_dataset)}")
            print(f"Lenght Test Data: {len(test_dataset)}")

            parse_train = ParseWikHopDataset(train_dataset)
            parse_dev = ParseWikHopDataset(dev_dataset)
        elif dataset in ['metaqa']:
            df_dev = pd.read_json("dataset/metaqa/2hops/qa_dev_evidences.json")
            df_train = pd.read_json("dataset/metaqa/2hops/qa_train_evidences.json")
            #df_test = pd.read_json("dataset/metaqa/2hops/qa_test_evidences.json")
            MAX_ANSWER = 1
            parse_train = ParseMetaQADataset(df_train, max_answers=MAX_ANSWER)
            parse_dev = ParseMetaQADataset(df_dev, max_answers=MAX_ANSWER)
        elif dataset in ['mlpq']:
            
            validation_dataframe = pd.read_json('dataset/mlpq/Questions/fr-en/2-hop/2hop_dev_question_evidences.json', lines=True)
            train_dataframe = pd.read_json('dataset/mlpq/Questions/fr-en/2-hop/2hop_train_question_evidences.json', lines=True)

            parse_train = ParseMLPQDataset(train_dataframe)
            parse_dev = ParseMLPQDataset(validation_dataframe)
        elif dataset in ['pql']:
            from src.datasets import ParsePQLDataset
            file_path = "dataset/pathquestion/PQ-2H.txt"
            train, val, test = load_train_test_pql_dataset(file_path, random_state = 789)
            file_path_kb = "dataset/pathquestion/2H-kb.txt"
            kg = create_knowledge_graph_pql(file_path, from_kb=False)

            parse_train = ParsePQLDataset(train)
            parse_dev = ParsePQLDataset(val)
        else:
            raise ValueError(f"Unknown Dataset")
        combined_dataset = ConcatDataset([parse_train])
    elif knowledge_integration:
        from src.datasets import KnowledgeIntegrationWikiHopDataset, KnowledgeIntegrationMetaQADataset, KnowledgeIntegrationMLPQDataset, KnowledgeIntegrationPQLDataset
        if dataset in ['2wikimultihop', 'wikimultihop', '2wikihop', 'wikihop']: 
            print("Training on 2WikiMultiHopQA")
            train_dataset, dev_dataset, test_dataset, kg_train, kg_dev, kg_test = load_dataset("dataset/2wikimultihop", do_correct_wrong_evidences=True)


            print("Creating Single Hop Datasets...")
            dataset_with_all_entries = pd.concat([train_dataset, dev_dataset, test_dataset])

            ki_dataset = KnowledgeIntegrationWikiHopDataset(dataset_with_all_entries)

        elif dataset in ['metaqa']:
            print("Training on MetaQA")
            #metaqa_kg_data = pd.read_csv('dataset/metaqa/kb.txt', sep="|")

            #Comment Out if we want to use the whole kb. This used only the kb for questions with a single answer
            df_dev = pd.read_json("dataset/metaqa/2hops/qa_dev_evidences.json")
            df_train = pd.read_json("dataset/metaqa/2hops/qa_train_evidences.json")
            df_test = pd.read_json("dataset/metaqa/2hops/qa_test_evidences.json")
            max_answers = 1
            df_kg = pd.concat([df_dev, df_train, df_test])
            kg = create_knowledge_graph_metaqa(df_kg, from_kb=False, max_answers=max_answers)

            ki_dataset = KnowledgeIntegrationMetaQADataset(kg)
        elif dataset in ['mlpq']:
            #txt_file_paths = ['dataset/mlpq/Triples_in_questions/EN_KG', 'dataset/mlpq/Triples_in_questions/FR_KG']
            train_dataframe = pd.read_json('dataset/mlpq/Questions/fr-en/2-hop/2hop_train_question_evidences.json', lines=True)
            validation_dataframe = pd.read_json('dataset/mlpq/Questions/fr-en/2-hop/2hop_dev_question_evidences.json', lines=True)
            test_dataframe = pd.read_json('dataset/mlpq/Questions/fr-en/2-hop/2hop_test_question_evidences.json', lines=True)

            df_kg = pd.concat([train_dataframe, validation_dataframe, test_dataframe])
            kg = create_knowledge_graph_mlpq(df_kg, from_kb = False, hops=2)
            ki_dataset = KnowledgeIntegrationMLPQDataset(kg)
        elif dataset in ['mlpq-3hop']:
            train_dataframe = pd.read_json('dataset/mlpq/Questions/fr-en/3-hop/3hop_train_question_evidences.json', lines=True)
            validation_dataframe = pd.read_json('dataset/mlpq/Questions/fr-en/3-hop/3hop_dev_question_evidences.json', lines=True)
            test_dataframe = pd.read_json('dataset/mlpq/Questions/fr-en/3-hop/3hop_test_question_evidences.json', lines=True)

            df_kg = pd.concat([train_dataframe, validation_dataframe, test_dataframe])
            kg = create_knowledge_graph_mlpq(df_kg, from_kb = False, hops=3)
            ki_dataset = KnowledgeIntegrationMLPQDataset(kg)
        elif dataset in ['pql']:
            # from src.utils.util import load_train_test_pql_dataset
            file_path_kb = "dataset/pathquestion/PQL-KB.txt"
            file_path_paths = "dataset/pathquestion/PQ-2H.txt"
            kg = create_knowledge_graph_pql(file_path_paths, from_kb=False)
            ki_dataset = KnowledgeIntegrationPQLDataset(kg)
        elif dataset in ['pql-3hop']:
            # from src.utils.util import load_train_test_pql_dataset
            # file_path_kb = "dataset/pathquestion/PQL-KB.txt"
            file_path_paths = "dataset/pathquestion/PQ-3H.txt"
            kg = create_knowledge_graph_pql(file_path_paths, from_kb=False)
            ki_dataset = KnowledgeIntegrationPQLDataset(kg)
        else:
            raise ValueError("Unknown Dataset")  
        combined_dataset = ConcatDataset([ki_dataset])

    dataloader = DataLoader(combined_dataset, shuffle=False, batch_size=256, num_workers=1)

    


    _comput_delta_hyperbolicity(dataloader, knit5_model, tuned, parse)