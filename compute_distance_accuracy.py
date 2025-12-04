import torch
import numpy as np
from tqdm import tqdm  # for progress display
from src.utils.util import load_dataset, get_top_token_embeddings, load_train_test_pql_dataset
import pandas as pd
from src.train.soft_prompt_trainer import SoftPromptTrainer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from src.config import Config
from torch.utils.data import DataLoader
from src.knowledge_graph import create_knowledge_graph_wikimultihop, create_knowledge_graph_metaqa, create_knowledge_graph_mlpq, create_knowledge_graph_pql
from src.models import SoftPromptModel, T5ModelWithAdditionalLayer
from src.datasets import RandomWalkMetaQADataset, RandomWalkMLPQDataset, RandomWalkWikiHopDataset, RandomWalkPQLDataset
import argparse
import os
from math import exp, log
from src.models.hyperbolic_model_utils import PoincareBallCustomAutograd
import torch.nn as nn


from src.knowledge_graph import create_knowledge_graph_metaqa
from src.datasets import RandomWalkMetaQADataset
def cosine_similarity(a, b):
    """Compute the cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
def euclidean_distance(a, b):
    """Compute the Euclidean distance between two vectors."""
    return np.linalg.norm(a - b)
import numpy as np

def hyperbolic_distance(u, v, c=1.0):
    """
    Calculate the hyperbolic distance between two points in the Poincaré disk model
    with curvature c.

    Parameters:
    -----------
    u, v : array_like
        Coordinates of the two points. They must lie within the disk 
        { x : c * ||x||^2 < 1 } (i.e. within a radius 1/sqrt(c)).
    c : float
        The curvature (a positive scalar). For c=1, this reduces to the standard unit disk.
    
    Returns:
    --------
    d : float
        The hyperbolic distance between u and v.
    """
    # Convert points to numpy arrays
    u = np.array(u)
    v = np.array(v)
    

    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    condition_u = norm_u < 1 / np.sqrt(c)
    condtion_v = norm_v < 1 / np.sqrt(c)

    if not condition_u:
        print(f"{norm_u = }")
    if not condtion_v:
        print(f"{norm_v = }")

    diff_norm = np.linalg.norm(u - v)
    
    # Compute the intermediate term using the curvature c
    intermediate = 2 * c * (diff_norm**2) / ((1 - c * norm_u**2) * (1 - c * norm_v**2))
    arg = 1 + intermediate
    
    # For numerical safety, ensure arg is at least 1 (arccosh is defined for x >= 1)
    arg = max(arg, 1.0)
    
    # Multiply by 1/sqrt(c) to scale the distance according to the curvature
    return (1 / np.sqrt(c)) * np.arccosh(arg)

# =============================================================================
# Helper: Check if distances from the source entity are strictly increasing.
# =============================================================================
def check_entity_ordering_for_sample(embeddings, tokens, euclidean = True, curvature = 0.0):
    """
    Given:
      - embeddings: a list of pooled embeddings (one per unit) in the order they appear.
      - tokens: a list of corresponding unit strings.
    This function selects only the entities (assumed to be in even positions: indices 0,2,4,...)
    and computes the Euclidean distances from the first (source) entity to every other entity.
    Returns:
      - is_correct: True if the distances (from source to entity_1, entity_2, …) are strictly increasing.
      - dists: the list of computed distances (for debugging).
      - entity_tokens: the list of tokens corresponding to the entities.
    """
    # Select only even-indexed elements (0, 2, 4, …)
    #entity_embeddings = [embeddings[i] for i in range(len(embeddings)) if i % 2 == 0]
    #entity_tokens = [tokens[i] for i in range(len(tokens)) if i % 2 == 0]
    
    if len(embeddings) < 3:
        return False, [], tokens
    
    source = embeddings[0]
    print(f"Calculating Distance for {tokens[0]} and  {tokens[1:]}")
    dists = [euclidean_distance(source, emb) if euclidean else hyperbolic_distance(source, emb, c=curvature) for emb in embeddings[1:]]
    is_correct = all(dists[i] > dists[i+1] for i in range(len(dists) - 1))
    return is_correct, dists, tokens

# =============================================================================
# Helper: Check if distances from the source entity are strictly increasing.
# =============================================================================
def check_entity_ordering_for_sample_decoder(embeddings, tokens):
    """
    Given:
      - embeddings: a list of pooled embeddings (one per unit) in the order they appear.
      - tokens: a list of corresponding unit strings.
    This function selects only the entities (assumed to be in even positions: indices 0,2,4,...)
    and computes the Euclidean distances from the first (source) entity to every other entity.
    Returns:
      - is_correct: True if the distances (from source to entity_1, entity_2, …) are strictly increasing.
      - dists: the list of computed distances (for debugging).
      - entity_tokens: the list of tokens corresponding to the entities.
    """
    # Select only even-indexed elements (0, 2, 4, …)
    entity_embeddings = [embeddings[i] for i in range(len(embeddings)) if i % 2 == 0]
    entity_tokens = [tokens[i] for i in range(len(tokens)) if i % 2 == 0]
    
    if len(entity_embeddings) < 2:
        return False, [], entity_tokens
    
    source = entity_embeddings[0]
    print(f"Calculating Distance for {entity_tokens[0]} and  {entity_tokens[1:]}")
    dists = [cosine_similarity(source, emb) for emb in entity_embeddings[1:]]
    is_correct = all(dists[i] > dists[i+1] for i in range(len(dists) - 1))
    return is_correct, dists, entity_tokens


# =============================================================================
# (Re)Use your mean pooling function (assumes text is semicolon-separated).
# =============================================================================
def mean_pool_semicolon_text(text, final_layer_tensor, tokenizer):
    """
    Splits `text` by ';' to obtain N "units" (entities/relations) and
    mean-pools the corresponding subword embeddings from final_layer_tensor.
    Returns:
      - pooled_embeddings: list of (hidden_dim,) numpy arrays (one per unit)
      - actual_units: list of the textual segments (str)
    """
    units = [u.strip() for u in text.split(';')]
    num_units = len(units)
    final_np = final_layer_tensor[0].cpu().numpy()  # shape: [seq_len, hidden_dim]
    enc_inputs = tokenizer(text, return_tensors='pt')
    input_ids_ = enc_inputs["input_ids"][0]
    tokens_ = tokenizer.convert_ids_to_tokens(input_ids_)
    unit_embeddings = [[] for _ in range(num_units)]
    current_unit_idx = 0
    for i, tok in enumerate(tokens_):
        tok_str = tok.replace("▁", "").strip()
        if tok_str == ";":
            if current_unit_idx < num_units - 1:
                current_unit_idx += 1
            continue
        unit_embeddings[current_unit_idx].append(final_np[i])
    pooled_embeddings = []
    actual_units = []
    for u_idx in range(num_units):
        sublist = unit_embeddings[u_idx]
        if not sublist:
            continue
        mean_emb = np.mean(sublist, axis=0)
        pooled_embeddings.append(mean_emb)
        actual_units.append(units[u_idx])
    print(f"Pooled Embeddings for {actual_units}")
    return pooled_embeddings, actual_units






def compute_distance_acc(dataset, prompt_checkpoint_euclidean, prompt_checkpoint_hyperbolic, model_checkpoint_path, dataset_type):
    MAX_ANSWER = None
    GPU_PARALLELIZATION = False if dataset in ['2wikimultihop', 'wikimultihop', '2wikihop', 'wikihop'] else True
    WITH_MODEL_STATE_DICT = GPU_PARALLELIZATION
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
    elif dataset in ['mlpq-3hop']:
        train_dataframe = pd.read_json('dataset/mlpq/Questions/fr-en/3-hop/3hop_train_question_evidences.json', lines=True)
        validation_dataframe = pd.read_json('dataset/mlpq/Questions/fr-en/3-hop/3hop_dev_question_evidences.json', lines=True)
        test_dataframe = pd.read_json('dataset/mlpq/Questions/fr-en/3-hop/3hop_test_question_evidences.json', lines=True)

        df_kg = pd.concat([train_dataframe, validation_dataframe, test_dataframe])
        kg = create_knowledge_graph_mlpq(df_kg, from_kb = False, hops=3)

        random_walk_train = RandomWalkMLPQDataset(kg, validation_dataframe, test_dataframe, steps=4, type='train')
        random_walk_dev = RandomWalkMLPQDataset(kg, validation_dataframe, test_dataframe, steps=4, type='dev')
    elif dataset in ['pql']:
        file_path = "dataset/pathquestion/PQ-2H.txt"
        train, val, test = load_train_test_pql_dataset(file_path, random_state = 789)
        file_path_kb = "dataset/pathquestion/2H-kb.txt"
        kg = create_knowledge_graph_pql(file_path, from_kb=False)

        random_walk_train = RandomWalkPQLDataset(kg, val, test, steps=3, type='train')
        random_walk_dev = RandomWalkPQLDataset(kg, val, test, steps=3, type='dev')
        random_walk_test = RandomWalkPQLDataset(kg, val, test, steps=3, type='test')
    else:
        raise ValueError(f"Unknown Dataset")
    # print(f"Number of Random Walks Train: {len(random_walk_train)}")
    print(f"Number of Random Walk Dev: {len(random_walk_dev)}")


    #Specify Hyperparameters via config file
    config = Config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Training on device: {device}')
    if dataset_type == 'train':
        dataset = random_walk_train
    elif dataset_type == 'dev':
        dataset = random_walk_dev
    elif dataset_type == 'test':
        dataset = random_walk_test
    elif dataset_type == 'combined':
        from torch.utils.data import ConcatDataset
        dataset = ConcatDataset([random_walk_dev, random_walk_test])
    else:
        raise ValueError('Only for dataset_type: train, dev, test')
    
    
    random_walk_dev_dataloader = DataLoader(dataset, batch_size=64, num_workers=1)



    #google/t5-large-lm-adapt
    model_name = config.t5_model.model_name
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Loading Model...")
    # '../checkpoints/metaqa/random_walk_training/hyperbolic/Jan20_04-36-49_AdaFactor_0.3_-0.9519287202435303_hyperbolic_after_encoder_bsize60_prompt_lenght100_lr0.3_curvature-0.9519287202435303_additional_layer_lr0.001_max_answer_1/soft_prompt_epoch_74_val_loss_0.3420_em_0.283266.pth'
    soft_prompt_checkpoint = torch.load(prompt_checkpoint_hyperbolic)
    soft_prompt = nn.Parameter(soft_prompt_checkpoint['soft_prompt_state_dict'])
    additional_layer = soft_prompt_checkpoint['additional_linear_layer']
    hyperbolic_knit5 = T5ModelWithAdditionalLayer(layer_type='hyperbolic', checkpoint_hyperbolic_knit5=model_checkpoint_path, with_model_state_dict=WITH_MODEL_STATE_DICT, gpu_parallelization=GPU_PARALLELIZATION)
    hyperbolic_knit5.hyperbolic_layer.load_state_dict(additional_layer)
    model_hyperbolic = SoftPromptModel(hyperbolic_knit5, None, soft_prompt=soft_prompt)
    print(f"Loaded Hyperbolic checkpoint from {prompt_checkpoint_hyperbolic}")
    # '../checkpoints/metaqa/random_walk_training/euclidean/Jan19_12-49-03_AdaFactor_0.3_0.541324854612918_linear_after_encoder_bsize60_prompt_lenght100_lr0.3_curvature0.541324854612918_additional_layer_lr0.001_max_answer_1/soft_prompt_epoch_66_val_loss_0.2485_em_0.229167.pth'
    soft_prompt_checkpoint = torch.load(prompt_checkpoint_euclidean)
    soft_prompt = nn.Parameter(soft_prompt_checkpoint['soft_prompt_state_dict'])
    additional_layer = soft_prompt_checkpoint['additional_linear_layer']
    euclidean_knit5 = T5ModelWithAdditionalLayer(layer_type='linear', checkpoint_hyperbolic_knit5=model_checkpoint_path, with_model_state_dict=WITH_MODEL_STATE_DICT, gpu_parallelization=GPU_PARALLELIZATION)
    euclidean_knit5.hyperbolic_layer.load_state_dict(additional_layer)
    model_euclidean = SoftPromptModel(euclidean_knit5, None, soft_prompt=soft_prompt)
    print(f"Loaded Euclidean checkpoint from {prompt_checkpoint_euclidean}")
    model_euclidean.to(device)
    model_hyperbolic.to(device)
    c = model_hyperbolic.knit5.hyperbolic_layer.manifold.c.item()
    print(f"{c = }")

    model_euclidean.eval()
    model_hyperbolic.eval()

    # correct_by_hyperbolic_only = []  # Store (input, label, hyperbolic_pred, euclidean_pred)

    # for input_text, label_text in tqdm(random_walk_dev_dataloader, desc=f"Hyperbolic Correct & Euclidean Incorrect {len(correct_by_hyperbolic_only)}"):
    #     # Tokenize input and label
    #     inputs = tokenizer(input_text, padding=True, truncation=True, return_tensors='pt')
    #     label_ids = tokenizer(label_text, padding=True, truncation=True, return_tensors='pt').input_ids.to(device)
    #     input_ids = inputs.input_ids.to(device)
    #     attention_mask = inputs.attention_mask.to(device)

    #     with torch.no_grad():
    #         # Get predictions from model_euclidean
    #         outputs_euclidean = model_euclidean.generate(input_ids=input_ids,
    #                                             attention_mask=attention_mask,
    #                                                 max_length=50,
    #                                                 num_beams = 5,
    #                                                 early_stopping=True)
    #         pred_text_euclidean = tokenizer.batch_decode(outputs_euclidean, skip_special_tokens=True)

    #         # Get predictions from model_hyperbolic
    #         outputs_hyperbolic = model_hyperbolic.generate(input_ids=input_ids,
    #                                             attention_mask=attention_mask,
    #                                                 max_length=50,
    #                                                 num_beams = 5,
    #                                                 early_stopping=True)
    #         pred_text_hyperbolic = tokenizer.batch_decode(outputs_hyperbolic, skip_special_tokens=True)

    #     for euclidean_pred, hyperbolic_pred, label, input in zip(pred_text_euclidean, pred_text_hyperbolic, label_text, input_text):
    #         # Check correctness
    #         correct_hyperbolic = (euclidean_pred.strip().lower() == label.strip().lower())
    #         correct_euclidean = (hyperbolic_pred.strip().lower() == label.strip().lower())
    #         if correct_hyperbolic:
    #             print(f"{euclidean_pred = }")
    #             print(f"{hyperbolic_pred = }")
    #         # Store cases where hyperbolic got it right, but euclidean did not
    #         if correct_hyperbolic and not correct_euclidean:
    #             correct_by_hyperbolic_only.append((input, label, hyperbolic_pred, euclidean_pred))
    #         # print(f"{euclidean_pred = }")
    #         # print(f"{hyperbolic_pred = }")
    #     # Display results
    # print(f"Number of cases where model_hyperbolic was correct but model_euclidean was not: {len(correct_by_hyperbolic_only)}")

    # # Print some examples
    # for i, (inp, lbl, hyp_pred, euc_pred) in enumerate(correct_by_hyperbolic_only[:10]):  # Show first 10 cases
    #     print(f"\nExample {i+1}:")
    #     print(f"Input: {inp}")
    #     print(f"True Label: {lbl}")
    #     print(f"Hyperbolic Prediction: {hyp_pred}")
    #     print(f"Euclidean Prediction: {euc_pred}")



    # =============================================================================
    # For each sample, get the ordering correctness for decoder and encoder
    # for both model_euclidean and model_hyperbolic.
    # =============================================================================

    # Optionally, work on a subset for speed
    # random_walk_dev_subset = []
    # for idx, (inp, lbl) in enumerate(random_walk_dev):
    #     if idx > 2000:
    #         break
    #     random_walk_dev_subset.append((inp, lbl))

    model_euclidean.eval()
    model_hyperbolic.eval()
    # Counters for comparing second hops
    correct_second_hop = 0  # Count of samples where hyperbolic's second hop is closer
    valid_samples = 0       # Count of samples that have at least a second hop

    correct_first_hop = 0
    correct_both_hops = 0
    for input_text, label_text in tqdm(dataset, desc="Samples"):
        with torch.no_grad():
            # Tokenize inputs and labels
            inputs = tokenizer(input_text, padding=True, truncation=True, return_tensors='pt')
            labels = tokenizer(label_text, padding=True, truncation=True, return_tensors='pt')
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)
            label_ids = labels.input_ids.to(device)
            
            # Get decoder outputs for Euclidean model
            outputs_euclid = model_euclidean(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=label_ids,
                output_hidden_states=True,
                return_dict=True
            )
            final_dec_euclid = outputs_euclid.encoder_hidden_states[-1]
            final_dec_euclid = model_euclidean.knit5.hyperbolic_layer(final_dec_euclid)
            
            # Get decoder outputs for Hyperbolic model
            outputs_hyp = model_hyperbolic(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=label_ids,
                output_hidden_states=True,
                return_dict=True
            )
            final_dec_hyp = outputs_hyp.encoder_hidden_states[-1]
            final_dec_hyp = model_hyperbolic.knit5.hyperbolic_layer(final_dec_hyp)
        # Assume this function returns a list of pooled embeddings and corresponding tokens
        pooled_dec_euclid, tokens_dec_euclid = mean_pool_semicolon_text(input_text, final_dec_euclid, tokenizer)
        pooled_dec_hyp, tokens_dec_hyp = mean_pool_semicolon_text(input_text, final_dec_hyp, tokenizer)
        
        # Ensure we have at least three entities (source + first hop + second hop)
        # The helper function `check_entity_ordering_for_sample` works on even indices:
        # Index 0 = source, Index 1 = first hop (from original index 2), Index 2 = second hop (from original index 4)
        _, dists_euclid_entities, _ = check_entity_ordering_for_sample(pooled_dec_euclid, tokens_dec_euclid, euclidean=True)
        _, dists_hyp_entities, _ = check_entity_ordering_for_sample(pooled_dec_hyp, tokens_dec_hyp, euclidean=True, curvature=c)
        
        # We need at least two cosine similarity values (first hop: dists[0], second hop: dists[1])
        if len(dists_euclid_entities) > 1 and len(dists_hyp_entities) > 1:
            valid_samples += 1
            # dists[0] corresponds to the first hop; dists[1] is the second hop.
            euclid_second = dists_euclid_entities[1]
            hyp_second = dists_hyp_entities[1]

            euclid_first = dists_euclid_entities[0]
            hyp_first = dists_hyp_entities[0]

            print(f"Euclidean Distance. Second Hop: {euclid_second}")
            print(f"Hyperbolic Distance. Second Hop: {hyp_second}")
            print(f"Euclidean Distance. First Hop: {euclid_first}")
            print(f"Hyperbolic Distance. First Hop: {hyp_first}")
            if hyp_first > euclid_first:
                correct_first_hop += 1
            if hyp_second > euclid_second:
                correct_second_hop +=1

            # Assuming that a higher cosine similarity means "closer"
            # if hyp_second < hyp_first:
            #     correct_second_hop += 1
            # if euclid_second < euclid_first:
            #     correct_second_hop_euclid+=1
            # correct_second_cond = hyp_second > euclid_second
            # correct_first_cond = hyp_first > euclid_first
            # if correct_first_cond and correct_second_cond:
            #     correct_both_hops+=1
            #     print(f"Euclidean Sim. Second Hop: {euclid_second}")
            #     print(f"Hyperbolic Sim. Second Hop: {hyp_second}")
            #     print(f"Euclidean Sim. First Hop: {euclid_first}")
            #     print(f"Hyperbolic Sim. First Hop: {hyp_first}")
            # if correct_second_cond:
            #     correct_second_hop+=1
            #     # print(f"{label_text = }")
            #     print(f"Euclidean Sim. Second Hop: {euclid_second}")
            #     print(f"Hyperbolic Sim. Second Hop: {hyp_second}")
            # if correct_first_cond:
            #     correct_first_hop+=1
            #     print(f"Euclidean Sim. First Hop: {euclid_first}")
            #     print(f"Hyperbolic Sim. First Hop: {hyp_first}")

    # Compute and print the accuracy metric
    if valid_samples > 0:
        accuracy = 100 * correct_second_hop / valid_samples
        print(f"Hyperbolic First > Euclid first {correct_first_hop * 100 / valid_samples}")
        print(f"Hyperbolic Second > Euclid Second {correct_second_hop * 100 / valid_samples}")
        # print("Hyperbolic second hop is more similar than euclidean in {:.2f}% of valid samples ({} out of {} samples).".format(
        #     accuracy, correct_second_hop, valid_samples))
        # print(f"Hyperbolic First Hop > Euclidean First Hop {correct_first_hop * 100/ valid_samples} of valid samples {correct_first_hop} / {valid_samples}")
        # print(f"Hyperbolic First Hop > Euclidean First Hop and Hyperbolic Second Hop > Euclidean First Hop {correct_both_hops * 100/ valid_samples} of valid samples {correct_both_hops} / {valid_samples}")
    else:
        print("No valid samples with at least two hops were found.")


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Random Walk Training Training')
    parser.add_argument('--dataset', type=str, nargs='?', default=None, help='Specify the dataset (e.g., metaqa, 2wikimultihop)')
    parser.add_argument(
        '--prompt_checkpoint_euclidean',
        type=str,
        default=None,
        help='Specify Checkpoint Path for Prompt'
    )
    parser.add_argument(
        '--prompt_checkpoint_hyperbolic',
        type=str,
        default=None,
        help='Specify Checkpoint Path for Prompt'
    )
    parser.add_argument(
        '--model_checkpoint_path',
        type=str,
        default=None,
        help='Specify Checkpoint Path for KNit5'
    )
    parser.add_argument(
        '--dataset_type',
        type=str,
        default='train',
        help='Specify Checkpoint Path for KNit5'
    )
    # New argument: --additional_layer
    args = parser.parse_args()
    dataset = args.dataset
    prompt_checkpoint_hyperbolic = args.prompt_checkpoint_hyperbolic
    prompt_checkpoint_euclidean = args.prompt_checkpoint_euclidean
    model_checkpoint_path = args.model_checkpoint_path
    dataset_type = args.dataset_type

    compute_distance_acc(dataset, prompt_checkpoint_euclidean, prompt_checkpoint_hyperbolic, model_checkpoint_path, dataset_type)
