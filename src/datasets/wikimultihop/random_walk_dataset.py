from ..random_walk_dataset import RandomWalkDataset
class RandomWalkWikiHopDataset(RandomWalkDataset):
    """Gets the Knowledge Graph as an input and should create Random Walks. For each node sample up to 20 random walks of length 3, do this 5 times with different seeds.
    #Hold out the random walks which are triples in the validation and test set.

    Returns: Random Walk dataset should output an incomplete sequence like "e1 ; r1 ; r2 ; ... ; rn-1;" and a complete sequence "e1 ; r1 ; e2 ; ... ; rn-1 ; en"
    """   
    def _get_walks(self, dataset):
        walks = set() 
        for entry in dataset['evidences']:
            triple1 = entry[0]
            triple2 = entry[1]
            walk = (triple1[0], triple1[1], triple1[2], triple2[1], triple2[2])
            walks.add(walk)
        return list(walks)
    
