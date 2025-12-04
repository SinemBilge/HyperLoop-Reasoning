from ..knowledge_integration_dataset import KnowledgeIntegrationDataset
class KnowledgeIntegrationWikiHopDataset(KnowledgeIntegrationDataset):
    """
    Needs a dataset that has all entries so train + dev + test. 
    """
    def __init__(self, dataset_with_all_entries):
        self.dataset = dataset_with_all_entries
        unique_triples = set()
        for entry in dataset_with_all_entries['evidences']:
            for triple in entry:
                unique_triples.add(tuple(triple))
        
        self.data = list(unique_triples)
    def __len__(self):
        return len(self.data)
    
    def __getitem__ (self, idx):
        e1, relation, e2 = self.data[idx]
        input_str = f"{e1} ; {relation}"
        label = f"{e2}"
        
        return input_str, label