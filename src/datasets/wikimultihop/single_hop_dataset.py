from torch.utils.data import Dataset
import random
import pandas as pd
class SingleHopDataset(Dataset):
    """Single Hop Dataset gets a dataset

    
    """
    def __init__(self, dataset, triples_not_to_contain = set()):
        self.dataset = dataset
        self.triples_not_to_contain = triples_not_to_contain
        self.relation_question_mapping = {
            'director': ['Who is the director of X?', 'Who directed the film X?'],
            'date of birth': ['What is the date of birth of X?', 'When is X’s birthday?', 'When was X born?'],
            'date of death': ['When did X die?', 'What is the date of death of X?'],
            'country': ['What country is X from?', 'What is thenationality of X?'],
            'country of citizenship': ['What country is X from?', 'What is the nationality of X?'],
            'award received': ['What is the award that X received?', 'Which award did X receive?'],
            'cause of death': ['Why did X die?', 'What was the causeof X’s death?'],
            'composer': ['Who is the composer of X?', 'Who composed X?'],
            'creator': ['Who is the creator of X?', 'Who created X?'],
            'child': ['Who is the child of X?'],
            'doctoral advisor': ['Who is the doctoral advisor of X?'],
            'editor': ['Who is the editor of X?', 'Who edited X?'],
            'educated at': ['Where did X graduate from?', 'What is the alma mater of X?', 'Where did X study?'],
            'employer': ['Who is the employer of X?', 'Where does X work?'],
            'father': ['Who is the father of X?', 'Who is X’s father?'],
            'mother': ['Who is the mother of X?', 'Who is X’s mother?'],
            'founded by': ['Who is the founder of X?', 'Who founded X?'],
            'inception': ['When was X founded?'],
            'manufacturer': ['Who manufactures X?'],
            'performer': ['Who is the performer of the song X?', 'Who performed the song X?'],
            'place of birth': ['Where was X born?', 'What is the place of birth of X?'],
            'place of burial': ['Where was X buried?', 'Where is the place of burial of X?'],
            'place of death': ['Where did X die?', 'Where is the place of death of X?'],
            'place of detention': ['Where did X go to prison?', 'Where was X detained?'],
            'presenter': ['Who is the presenter of X?', 'Who presented X?'],
            'publisher': ['Who published X?', 'What company published X?'],
            'sibling': ['Who is the sibling of X?', 'Who is X’s sibling?'],
            'spouse': ['Who is the spouse of X?', 'Who is X’s spouse?'],
            'student of': ['Who was the teacher of X?', 'Who was X’s teacher?'],
        }
        
        
        
        self.data = self._create_single_hop_dataset(dataset)
    
    def _create_single_hop_dataset(self, dataset):
        unique_triples = set()
        for entry in dataset['evidences']:
            for triples in entry:
                unique_triples.add(tuple(triples))
        
        
        rows = []
        unique_id = 0 
        for triple in unique_triples:
            if triple in self.triples_not_to_contain:
                continue
            e1, relation, e2 = triple[0], triple[1], triple[2]
            possible_questions = self.relation_question_mapping[relation]
            random_question = random.choice(possible_questions)
            random_question = random_question.replace('X', e1) # Replace the X with the entity

            rows.append({
                '_id': unique_id, #Just a Counter to give unique ids
                'type': 'single_hop_question',
                'question': random_question, # Random Question from the Dictionary
                'answer': e2, # Label
                'context': 0, # TODO Need to add the context. Take the relevant context parts of the Multi Hop Question.
                'evidences': triple # Single Hop has only one evidence
            })
            
            unique_id += 1
        single_hop_dataset = pd.DataFrame(rows,columns=['_id', 'type', 'question', 'context', 'evidences', 'answer'])
        return single_hop_dataset
                
                   
                
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data.iloc[idx]