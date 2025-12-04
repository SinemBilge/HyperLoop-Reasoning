from torch.utils.data import Dataset
import networkx as nx

class KnowledgeIntegrationDataset(Dataset):
    """
    Takes in a Graph with all entities and relations
    """
    def __init__(self, kg : nx.MultiGraph):
        self.data = []

        for u, v, data in kg.edges(data = True):
            rel = data.get('relation', 'unknown_relation')
            self.data.append((f"{u} ; {rel}", f"{v}"))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__ (self, idx):
        input_str, label = self.data[idx]
        return input_str, label