from torch.utils.data import Dataset
from .single_hop_dataset import SingleHopDataset
class OneWikiHopDataset(Dataset):
    """

    """
    def __init__(self, train_dataset, dev_dataset, test_dataset, type='train'):
        if type not in ['train', 'test', 'dev']:
            raise ValueError(f"Unknown type: {type}. Supported Types are ['train', 'test', 'dev']")
        self.type = type
        triples_not_to_contain = set()
        if type == 'train':
            self.dataset = SingleHopDataset(train_dataset, triples_not_to_contain)
        elif type == 'dev':
            single_hop_dataset_train = SingleHopDataset(train_dataset, triples_not_to_contain)
            for triple in single_hop_dataset_train.data['evidences']:
                triples_not_to_contain.add(triple)
            self.dataset = SingleHopDataset(dev_dataset, triples_not_to_contain)
        elif type == 'test':
            single_hop_dataset_train = SingleHopDataset(train_dataset, triples_not_to_contain)
            for triple in single_hop_dataset_train.data['evidences']:
                triples_not_to_contain.add(triple)
            single_hop_dataset_dev = SingleHopDataset(dev_dataset, triples_not_to_contain)
            for triple in single_hop_dataset_dev.data['evidences']:
                triples_not_to_contain.add(triple)
            self.dataset = SingleHopDataset(test_dataset, triples_not_to_contain)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        question = self.dataset[idx]['question']
        context = self.dataset[idx]['context']
        evidences = self.dataset[idx]['evidences']
        answer = self.dataset[idx]['answer']
        
        return question, answer