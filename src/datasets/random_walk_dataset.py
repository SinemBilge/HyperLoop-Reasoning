from torch.utils.data import Dataset
import random
from tqdm import tqdm
import sys
import numpy as np
class RandomWalkDataset(Dataset):
    """Gets the Knowledge Graph as an input and should create Random Walks. For each node sample up to 20 random walks of length 3, do this 5 times with different seeds.
    #Hold out the random walks which are triples in the validation and test set.

    Returns: Random Walk dataset should output an incomplete sequence like "e1 ; r1 ; r2 ; ... ; rn-1;" and a complete sequence "e1 ; r1 ; e2 ; ... ; rn-1 ; en"
    """
    def __init__(self, all_kg, validation_dataset, test_dataset, steps, type = 'train'):
        self.kg = all_kg
        self.steps = steps
        
        if type not in ['train', 'test', 'dev']:
            raise ValueError(f"Unknown type: {type}. Supported Types are ['train', 'test', 'dev']") 
        
        walks_val = self._get_walks(validation_dataset)
        walks_test = self._get_walks(test_dataset)
        if type == 'train':
            self.data = list(self._generate_random_walks(walk_length=steps))
            
            print(f"Number of Random Walks without removing test & val: {len(self.data)}")
            
            walks_val_test = set(walks_val + walks_test)
        
            #Hold out the ones from the validaiton and test set
            self.data = [walk for walk in self.data if walk not in walks_val_test]
            print(f"Number of Random Walks with removing test & val: {len(self.data)}")
        elif type == 'test':
            self.data = walks_test
        elif type == 'dev':
            self.data = walks_val
           
    def _get_walks(self, dataset):
        raise NotImplementedError("Subclasses must implement the `_get_walks` method.")
    
        
    def _random_walk(self, start_node, hops=2):
        path = [start_node]
        current_node = start_node
        for _ in range(hops):
            neighbors = list(self.kg.successors(current_node))
            if not neighbors:
                break  # Restart the walk if dead-end is reached
            next_node = random.choice(neighbors)
            # Randomly select one of the edges (relations) if multiple exist between the nodes
            edge_data = self.kg.get_edge_data(current_node, next_node)
            relation = random.choice(list(edge_data.values()))['relation']
            path.append(relation)
            path.append(next_node)
            current_node = next_node
        return path
    
    def _generate_random_walks(self, num_walks_per_node=20, walk_length=3, num_iterations=5, base_seed = 42):
        all_paths = set()
        nodes = list(self.kg.nodes())
        for iter_num in tqdm(range(num_iterations), desc=f"Generating walks", file=sys.stdout, dynamic_ncols=True):
            iteration_seed = base_seed+iter_num
            print(f"Seed: {iteration_seed}")
            random.seed(iteration_seed)  # Change seed for each iteration
            np.random.seed(iteration_seed)
            for idx, node in enumerate(nodes):
                walks = set()
                for _ in range(num_walks_per_node):
                    path = self._random_walk(node, walk_length-1)
                    if path and len(path) == 2*walk_length - 1:  # Ensure path is exactly of the required length
                        walks.add(tuple(path))
                all_paths.update(walks)
        return all_paths
    
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        random_walk = self.data[idx]
        
        complete_path = f"{random_walk[0]}"
        incomplete_path = f"{random_walk[0]}"
        for i in range(1, len(random_walk)):
            complete_path += f" ; {random_walk[i]}"
            if i % 2 != 0:
                incomplete_path += f" ; {random_walk[i]}"

        return incomplete_path, complete_path