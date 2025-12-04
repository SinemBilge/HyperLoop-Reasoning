import torch
from torch.utils.data import Dataset
import random
from tqdm import tqdm
import sys
import networkx
from collections import defaultdict
import numpy as np

class RandomWalkMetaQADataset(Dataset):
    """
    Generates random walks from a Knowledge Graph, organizing them into incomplete paths with multiple completions.

    Each sample consists of:
    - Incomplete Path: "entity1 ; relation1 ; relation2"
    - Complete Paths: "entity1 ; relation1 ; entity2 ; relation2 ; entity3_1 | entity1 ; relation1 ; entity2 ; relation2 ; entity3_2 | ..."
    
    Ensures that for all complete paths, entity1 != entity3.
    """
    def __init__(self, all_kg, validation_dataframe, test_dataframe, steps, type='train', max_answers : int = 1):
        self.kg: networkx.MultiDiGraph = all_kg
        self.steps = steps  # Not directly used, but retained for compatibility

        if type not in ['train', 'dev', 'test']:
            raise ValueError(f"Unknown type: {type}. Supported Types are ['train', 'dev', 'test']") 

        # Extract validation and test walks
        walks_val = self._get_walks(validation_dataframe)
        walks_test = self._get_walks(test_dataframe)
        
        if type == 'train':
            # Generate random walks
            all_walks = list(self._generate_random_walks())
            print(f"Number of Random Walks without removing test & val: {len(all_walks)}")
            
            # Remove validation and test walks
            walks_val_test = set(walks_val + walks_test)
            filtered_walks = [walk for walk in all_walks if walk not in walks_val_test]
            print(f"Number of Random Walks removing test & val: {len(filtered_walks)}")
            walks = filtered_walks
        elif type == 'test':
            walks = walks_test
        elif type == 'dev':
            walks = walks_val

        # Organize walks into incomplete and complete paths
        self.data = self._organize_walks(walks)
        cleaned_data = []
        for incomplete, complete in self.data:
            splitted_complete = complete.split('|')
            if len(splitted_complete) > max_answers:
                continue
            cleaned_data.append((incomplete, complete))
        print(f"Left with {len(cleaned_data)} / {len(self.data)} Walks of lengths {max_answers}")

        self.data = cleaned_data


        
    def _get_walks(self, dataset):
        """
        Extract walks from the dataset's evidences.
        
        Parameters:
        - dataset (DataFrame): DataFrame containing a column 'evidences' where each entry is a list of evidence tuples.

        Returns:
        - List of walks as tuples.
        """
        walks = set() 
        for entry in dataset['evidences']:
            for evidence in entry:
                tuple_evidence = tuple(evidence)
                walks.add(tuple_evidence)
        return list(walks)
    
    def _random_walk(self, start_node, hops=2):
        """
        Perform a random walk starting from start_node with a given number of hops.

        Parameters:
        - start_node: The node from which to start the walk.
        - hops (int): Number of hops (relations) to traverse.

        Returns:
        - List representing the walk: [entity1, relation1, entity2, relation2, entity3]
        """
        path = [start_node]
        current_node = start_node
        for _ in range(hops):
            neighbors = list(self.kg.successors(current_node))
            if not neighbors:
                break  # Dead-end
            next_node = random.choice(neighbors)
            # Select a random relation if multiple edges exist
            edge_data = self.kg.get_edge_data(current_node, next_node)
            if not edge_data:
                break  # No edge data found
            relations = [edata['relation'] for edata in edge_data.values()]
            if not relations:
                break  # No relation found
            relation = random.choice(relations)
            path.append(relation)
            path.append(next_node)
            current_node = next_node
        return path
    
    def _generate_random_walks(self, num_walks_per_node=20, walk_length=3, num_iterations=5, base_seed = 42):
        """
        Generate a set of unique random walks.

        Parameters:
        - num_walks_per_node (int): Number of walks to generate per node per iteration.
        - walk_length (int): Number of entities in each walk.
        - num_iterations (int): Number of iterations to perform.

        Returns:
        - Set of walks as tuples.
        """
        all_paths = set()
        nodes = list(self.kg.nodes())
        for iter_num in tqdm(range(num_iterations), desc=f"Generating walks", file=sys.stdout, dynamic_ncols=True):
            iteration_seed = base_seed+iter_num
            print(f"Seed: {iteration_seed}")
            random.seed(iteration_seed)  # Change seed for each iteration
            np.random.seed(iteration_seed)
            for node in nodes:
                for _ in range(num_walks_per_node):
                    path = self._random_walk(node, hops=walk_length - 1)
                    if path and len(path) == 2 * (walk_length - 1) + 1:  # Exact length
                        if path[0] != path[2*walk_length - 2]:
                            all_paths.add(tuple(path))
        return all_paths
    
    def _organize_walks(self, walks):
        """
        Organize walks into a dictionary where each key is an incomplete path
        and the value is a list of possible complete paths.

        Incomplete Path: "entity1 ; relation1 ; relation2"
        Complete Paths: "entity1 ; relation1 ; entity2 ; relation2 ; entity3"

        Ensures that entity1 != entity3.

        Parameters:
        - walks (List[Tuple]): List of walks as tuples.

        Returns:
        - List of tuples: [(incomplete_path, completions), ...]
        """
        incomplete_to_completions = defaultdict(set)
        for walk in walks:
            if len(walk) != 5:
                continue  # Ensure the walk is exactly 2 hops: e1 ; r1 ; e2 ; r2 ; e3
            e1, r1, e2, r2, e3 = walk
            if e1 == e3:
                continue  # Exclude walks where e1 == e3

            # Define the incomplete path as "e1 ; r1 ; r2"
            incomplete = f"{e1} ; {r1} ; {r2}"
            # Define the complete path as "e1 ; r1 ; e2 ; r2 ; e3"
            complete = f"{e1} ; {r1} ; {e2} ; {r2} ; {e3}"
            incomplete_to_completions[incomplete].add(complete)
        
        # Convert to list of tuples (incomplete, completions)
        data = []
        for incomplete, completions in incomplete_to_completions.items():
            # Sort completions for consistency
            sorted_completions = sorted(completions)
            # Join completions with '|'
            completions_joined = " | ".join(sorted_completions)
            data.append((incomplete, completions_joined))
        
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        incomplete_path, completions = self.data[idx]
        return incomplete_path, completions
