from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import networkx as nx
import torch
import torch.nn as nn
import pandas as pd
import sys
import os
from tqdm import tqdm
# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory to the system path
sys.path.append(current_dir)


from src.knowledge_graph import create_knowledge_graph_wikimultihop
import numpy as np
from sklearn.model_selection import train_test_split


def get_n_hops_from_node(knowledge_graph: nx.MultiDiGraph, start_node, n_hops: int):
    paths = [(start_node, "")]

    for _ in range(n_hops):
        new_paths = []
        for current_node, path_str in paths:
            # Get all successors of the current node
            for successor in knowledge_graph.successors(current_node):
                # Get all edges (relations) between current_node and successor
                for key, edge_data in knowledge_graph[current_node][successor].items():
                    relation = edge_data.get('relation', 'related_to')
                    # Create new path string
                    new_path_str = f"{path_str} ; {relation} ; {successor}" if path_str else f"{current_node} ; {relation} ; {successor}"
                    # Append the new path to the new_paths list
                    new_paths.append((successor, new_path_str))
        paths = new_paths
    
    # Return only the path strings
    return [path_str for _, path_str in paths]

#We fix it
def correct_wrong_evidences(df : pd.DataFrame):
    wrong = {}
    for _, entry in df.iterrows():
        index = entry['_id']
        evidences = entry['evidences']
        
        # Ensure there are at least two evidence triples
        if len(evidences) >= 2:
            triple1 = evidences[0]
            triple2 = evidences[1]
        else:
            print(f'Evidences {evidences}')
        if triple1[2] != triple2[0]:
            wrong[index]=(triple1[2], triple2[0])
    #print(len(wrong))
    
    def update_evidences(evidences):
        if len(evidences) >= 2:
            evidences[0][-1] = evidences[1][0]
        return evidences
    df.loc[df['_id'].isin(list(wrong.keys())), 'evidences'] = df.loc[df['_id'].isin(list(wrong.keys())), 'evidences'].apply(update_evidences)
def check_for_wrong_2_hops(df : pd.DataFrame):
    def check_e2_not_equal_e3(evidences):
        return evidences[0][2] != evidences[1][0]
    df['wrong_evidences'] = df['evidences'].apply(check_e2_not_equal_e3)
    rows_with_wrong_evidences = df[df['wrong_evidences']]
    
    return rows_with_wrong_evidences
def krackhardt_hierarchy_score(knowledge_graph):
    """
    Calculate the Krackhardt hierarchy score for a directed graph.
    """
    adj_matrix = nx.to_numpy_array(knowledge_graph, dtype=np.int8)
    R = np.array(adj_matrix, dtype=np.int8)
    
    numerator = np.sum(R * (1-R.T))
    denominator = np.sum(R)
    
    if denominator == 0:
        return 0.0
    else:
        khs = numerator / denominator
        return khs        
def print_datapoint(dataset, idx):
    print(f"id: {dataset['_id'][idx]}")
    print(f"type: {dataset['type'][idx]}") # Type of Question: ['bridge_comparison' 'compositional' 'inference' 'comparison'] / 
                                                # Bridge_comparison: Involves comparing attributes of entities through a connecting intermediate entity.
                                                # Compositional: Requires combining multiple pieces of information to derive the answer.
                                                #   Inference: Needs logical reasoning to deduce the answer from given facts.
                                                #   Comparison: Directly compares attributes or facts of two or more entities.
    print(f"question: {dataset['question'][idx]}") # Original Question
    print(f"context: {dataset['context'][idx]}") # Context can be used to concatonate with question to form input sequence
    print(f"evidences: {dataset['evidences'][idx]}") # e1, r1, e2 For knowledge Graph
    print(f"answer: {dataset['answer'][idx]}") # Label / Answer
import pickle
def load_train_test_pql_dataset(file_path : str, test_ratio = 0.1, random_state = 42):
    data_list = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            splitted_line = line.split('\t')
            question = splitted_line[0]
            answer = splitted_line[1].split('(')[0]
            path = splitted_line[2].split('#')
            if "<end>" in path:
                end_index = path.index("<end>")
                path = path[:end_index]
            data_list.append({
                'question':question,
                'answer': answer,
                'evidences':path
            })
    data = pd.DataFrame(data_list)
    train_dataset, test_dataset = train_test_split(data, test_size=test_ratio, random_state=random_state)
    train_dataset, val_dataset = train_test_split(data, test_size=0.11, random_state=0)
    return train_dataset, val_dataset, test_dataset

        
            
            

def load_dataset(path: str,
                 do_correct_wrong_evidences : bool = False):
    print(f"Loading Datasets...")
    train_dataset = pd.read_json(f"{path}/train.json").dropna(how='all')
    test_dataset = pd.read_json(f"{path}/dev.json").dropna(how='all')
    
    def contains_has_part(evidences):
        return any(r == 'has part' for e1, r, r2 in evidences)
    
    train_dataset = train_dataset[~train_dataset['evidences'].apply(contains_has_part)]
    test_dataset = test_dataset[~test_dataset['evidences'].apply(contains_has_part)]
    
    
    validation_ratio = 0.1
    train_dataset = train_dataset[(train_dataset['type'] == 'compositional') | (train_dataset['type'] == 'inference')]
    train_dataset, dev_dataset = train_test_split(train_dataset, test_size=validation_ratio, random_state=120)
    test_dataset = test_dataset[(test_dataset['type'] == 'compositional') | (test_dataset['type'] == 'inference')]
    
    if do_correct_wrong_evidences:
        print(f'Correct Wrong 2 Hops...')
        correct_wrong_evidences(train_dataset)
        correct_wrong_evidences(dev_dataset)
        correct_wrong_evidences(test_dataset)

    print("Creating Knowledge Graphs...")
    kg_train = create_knowledge_graph_wikimultihop(train_dataset)
    kg_dev = create_knowledge_graph_wikimultihop(dev_dataset)
    kg_test = create_knowledge_graph_wikimultihop(test_dataset)
    
    return train_dataset, dev_dataset, test_dataset, kg_train, kg_dev, kg_test


def load_c4_dataset(base_path: str,
                    number_of_files : int,
                    chunk_size : int = 100_000):
    list_of_texts = []
    for i in tqdm(range(number_of_files), desc='Loading Dataframe with c4 Data...', file = sys.stdout, dynamic_ncols=True):
        for chunk in pd.read_json(base_path.format(i), lines=True, chunksize=chunk_size):
            list_of_texts.extend(chunk['text'])
    return list_of_texts
    
    
def get_top_token_embeddings(model : AutoModelForSeq2SeqLM, tokenizer : AutoTokenizer, k : int):
    vocab = tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda item: item[1])
    
    top_tokens = [token for token, idx in sorted_vocab[:k]]
    
    top_tokens_ids = tokenizer.convert_tokens_to_ids(top_tokens)
    
    token_embeddings = model.shared.weight.data
    
    top_embeddings = token_embeddings[top_tokens_ids]
    return top_embeddings   

def count_learnable_parameters(model: nn.Module):
    """
    This function calculates the number of learnable parameters in a PyTorch model.
    
    Args:
    model (nn.Module): The PyTorch model.
    
    Returns:
    int: The total number of learnable parameters in the model.
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params
import numpy as np
import random
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Ensures deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
##---------------------hyperbolic operations ----------------------- CODE TAKEN FROM https://github.com/HazyResearch/KGEmb/blob/master/utils/hyperbolic.py
import torch
import torch.nn.functional as F
# ################# MATH FUNCTIONS ########################

class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-5, 1 - 1e-5)
        ctx.save_for_backward(x)
        dtype = x.dtype
        x = x.double()
        return (torch.log_(1 + x).sub_(torch.log_(1 - x))).mul_(0.5).to(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 - input ** 2)


def artanh(x):
    return Artanh.apply(x)


def tanh(x):
    return x.clamp(-15, 15).tanh()


# ################# HYP OPS ########################

MIN_NORM = 1e-15
BALL_EPS = {torch.float32: 1e-5, torch.float64: 1e-7}

def expmap0(u : torch.Tensor, c : float):
    """Exponential map taken at the origin of the Poincare ball with curvature c."""
    sqrt_c = c ** 0.5
    u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
    return project(gamma_1, c)

def logmap0(y : torch.Tensor, c : float):
    """Logarithmic map taken at the origin of the Poincare ball with curvature c."""
    sqrt_c = c ** 0.5
    y_norm = y.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    return y / y_norm / sqrt_c * torch.atanh(sqrt_c * y_norm)

def project(x : torch.Tensor, c : float):
    """Project points to Poincare ball with curvature c."""
    norm = x.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    eps = BALL_EPS[x.dtype]
    maxnorm = (1 - eps) / (c ** 0.5)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)

def mobius_add(x, y, *, c=1.0):
    r"""
    Mobius addition is a special operation in a hyperbolic space.
    .. math::
        x \oplus_c y = \frac{
            (1 + 2 c \langle x, y\rangle + c \|y\|^2_2) x + (1 - c \|x\|_2^2) y
            }{
            1 + 2 c \langle x, y\rangle + c^2 \|x\|^2_2 \|y\|^2_2
        }
    In general this operation is not commutative:
    .. math::
        x \oplus_c y \ne y \oplus_c x
    But in some cases this property holds:
    * zero vector case
    .. math::
        \mathbf{0} \oplus_c x = x \oplus_c \mathbf{0}
    * zero negative curvature case that is same as Euclidean addition
    .. math::
        x \oplus_0 y = y \oplus_0 x
    Another usefull property is so called left-cancellation law:
    .. math::
        (-x) \oplus_c (x \oplus_c y) = y
    Parameters
    ----------
    x : tensor
        point on the Poincare ball
    y : tensor
        point on the Poincare ball
    c : float|tensor
        ball negative curvature
    Returns
    -------
    tensor
        the result of mobius addition
    """
    c = torch.as_tensor(c).type_as(x)
    return _mobius_add(x, y, c)


def _mobius_add(x, y, c):
    x2 = x.pow(2).sum(dim=-1, keepdim=True)
    y2 = y.pow(2).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
    return num / (denom + 1e-5)

#----------------------------------------------------------
