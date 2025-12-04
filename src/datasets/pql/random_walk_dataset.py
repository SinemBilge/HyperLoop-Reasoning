from ..random_walk_dataset import RandomWalkDataset
import pandas as pd
class RandomWalkPQLDataset(RandomWalkDataset):
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
        """
        Extract walks from the dataset's evidences.
        
        Parameters:
        - dataset (DataFrame): DataFrame containing a column 'evidences' where each entry is a list of evidence tuples.

        Returns:
        - List of walks as tuples.
        """
        walks = set() 
        for entry in dataset['evidences']:
            tuple_evidence = tuple(entry)
            walks.add(tuple_evidence)
        return list(walks)
    