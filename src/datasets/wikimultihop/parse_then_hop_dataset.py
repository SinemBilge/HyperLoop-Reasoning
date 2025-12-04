from ..parse_then_hop_dataset import ParseDataset, ParseThenHopDataset
class ParseWikHopDataset(ParseDataset):
    def __getitem__ (self, idx):
        question = self.dataset.iloc[idx]['question']
        evidences = self.dataset.iloc[idx]['evidences']
        
        evidence_list = [evidences[0][0], evidences[0][1], evidences[0][2], evidences[1][1], evidences[1][2]]
        incomplete_path = f"{evidence_list[0]}"
        for i in range(1, len(evidence_list)):
            if i % 2 != 0:
                incomplete_path += f" ; {evidence_list[i]}"
            
        return question, incomplete_path
    
class ParseThenHopWikiHopDataset(ParseThenHopDataset):
    def __getitem__ (self, idx):
        question = self.dataset.iloc[idx]['question']
        evidences = self.dataset.iloc[idx]['evidences']
        
        evidence_list = [evidences[0][0], evidences[0][1], evidences[0][2], evidences[1][1], evidences[1][2]]
        complete_path = f"{evidence_list[0]}"
        for i in range(1, len(evidence_list)):
            complete_path += f" ; {evidence_list[i]}"

            
        return question, complete_path