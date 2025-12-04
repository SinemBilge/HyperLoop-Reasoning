import pandas as pd
from ..parse_then_hop_dataset import ParseDataset, ParseThenHopDataset
class ParseMetaQADataset(ParseDataset):
    def __init__(self, dataframe: pd.DataFrame, max_answers : int = 1):
        """
        Initializes the dataset with the provided DataFrame.

        Args:
            dataframe (pd.DataFrame): The input DataFrame containing 'question' and 'evidences' columns.
        """
        super().__init__(dataframe)
        self.dataset = self.dataset[self.dataset['evidences'].apply(lambda x : len(x) <= max_answers)]

        for evidence_list in self.dataset['evidences']:
            if len(evidence_list) > max_answers:
                print(f"Found List with more than {max_answers} evidences: {evidence_list}")
        print(f"No Evidence List found with more than {max_answers}")



    @staticmethod
    def construct_incomplete_paths(evidences):   
        evidence_list = evidences[0]
        # Construct incomplete path for the current evidence list
        incomplete_path = f"{evidence_list[0]}"  # Start with the first entity
        for i in range(1, len(evidence_list)):
            if i % 2 != 0:  # Add only relations (odd indices)
                incomplete_path += f" ; {evidence_list[i]}"
            
        
        return incomplete_path

    
class ParseThenHopMetaQADataset(ParseThenHopDataset):
    # def __init__(self, dataframe : pd.DataFrame, max_answers : int = 3):
    #     super().__init__(dataframe)
    #     self.dataset = self.dataset[self.dataset['evidences'].apply(lambda x : len(x) <= max_answers)]
        
    
    def __getitem__ (self, idx):
        question = self.dataset.iloc[idx]['question']
        evidences = self.dataset.iloc[idx]['evidences']
        
        complete_path = self._process_evidences(evidences)

            
        return question, complete_path
    
    def _process_evidences(evidences):
        """
        Processes the 'evidences' to create incomplete sequences.

        Args:
            evidences (list): A list of complete sequences, where each complete sequence is a list in the format 
                              [e1, r1, e2, r2, ..., rn-1, en].

        Returns:
            str: A string representing the incomplete sequences, formatted as 
                 "e1 ; r1 ; e2 ; r2 ... ; rn-1 | e1 ; r1 ; ... | ...".
        """
        if not isinstance(evidences, list):
            raise TypeError("Each 'evidences' entry must be a list of sequences.")
        
        complete_sequences = []
        
        for seq_idx, sequence in enumerate(evidences):
            if not isinstance(sequence, list):
                raise TypeError(f"Sequence at index {seq_idx} is not a list.")
            
            if len(sequence) < 2:
                # A valid complete sequence should have at least one entity and one relation
                raise ValueError(f"Sequence at index {seq_idx} is too short to process.")
            
            # Remove the last entity to create an incomplete sequence
            complete = sequence
            
            # Join the elements with ' ; '
            complete_str = ' ; '.join(map(str, complete))
            
            complete_sequences.append(complete_str)
        
        # Join all incomplete sequences with ' | '
        final_incomplete_sequence = ' | '.join(complete_sequences)
        
        return final_incomplete_sequence