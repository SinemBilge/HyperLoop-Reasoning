from .metaqa import *
from .wikimultihop import *
from .mlpq import *
from .pql import  *
import pandas as pd
from ..utils.util import load_dataset, load_train_test_pql_dataset
from ..knowledge_graph import create_knowledge_graph_metaqa, create_knowledge_graph_mlpq, create_knowledge_graph_pql, create_knowledge_graph_wikimultihop

def get_random_walk_dataset(dataset):
    if dataset in ['2wikimultihop', 'wikimultihop', '2wikihop', 'wikihop']:
        train_dataset, dev_dataset, test_dataset, kg_train, kg_dev, kg_test = load_dataset('dataset/2wikimultihop', do_correct_wrong_evidences=True)

        all_data = pd.concat([train_dataset, dev_dataset, test_dataset])
        all_kg = create_knowledge_graph_wikimultihop(all_data)

        print(f"Nodes in Data: {len(list(all_kg.nodes()))}")

        print(f"Lenght Train Data: {len(train_dataset)}")
        print(f"Lenght Dev Data: {len(dev_dataset)}")
        print(f"Lenght Test Data: {len(test_dataset)}")

        random_walk_train = RandomWalkWikiHopDataset(all_kg, dev_dataset, test_dataset, steps=3, type='train')
        random_walk_dev = RandomWalkWikiHopDataset(all_kg, dev_dataset, test_dataset, steps=3, type='dev')
        random_walk_test = RandomWalkWikiHopDataset(all_kg, dev_dataset, test_dataset, steps=3, type='test')


    elif dataset in ['metaqa']:
        # df_kg = pd.read_csv("dataset/metaqa/kb.txt", sep="|")
        # kg = create_knowledge_graph_metaqa(df_kg, from_kb=True)

        df_dev = pd.read_json("dataset/metaqa/2hops/qa_dev_evidences.json")
        df_train = pd.read_json("dataset/metaqa/2hops/qa_train_evidences.json")
        df_test = pd.read_json("dataset/metaqa/2hops/qa_test_evidences.json")
        MAX_ANSWER = 1
        df_kg = pd.concat([df_dev, df_train, df_test])
        kg = create_knowledge_graph_metaqa(df_kg, from_kb=False, max_answers=MAX_ANSWER)
        random_walk_train = RandomWalkMetaQADataset(kg, df_dev, df_test, steps=3, type='train')
        random_walk_dev = RandomWalkMetaQADataset(kg, df_dev, df_test, steps=3, type='dev')
        random_walk_test = RandomWalkMetaQADataset(kg, df_dev, df_test, steps=3, type='test')
    elif dataset in ['mlpq']:
        #txt_file_paths = ['dataset/mlpq/Triples_in_questions/EN_KG', 'dataset/mlpq/Triples_in_questions/FR_KG']
        train_dataframe = pd.read_json('dataset/mlpq/Questions/fr-en/2-hop/2hop_train_question_evidences.json', lines=True)
        validation_dataframe = pd.read_json('dataset/mlpq/Questions/fr-en/2-hop/2hop_dev_question_evidences.json', lines=True)
        test_dataframe = pd.read_json('dataset/mlpq/Questions/fr-en/2-hop/2hop_test_question_evidences.json', lines=True)

        df_kg = pd.concat([train_dataframe, validation_dataframe, test_dataframe])
        kg = create_knowledge_graph_mlpq(df_kg, from_kb = False)

        random_walk_train = RandomWalkMLPQDataset(kg, validation_dataframe, test_dataframe, steps=3, type='train')
        random_walk_dev = RandomWalkMLPQDataset(kg, validation_dataframe, test_dataframe, steps=3, type='dev')
        random_walk_test = RandomWalkMLPQDataset(kg, validation_dataframe, test_dataframe, steps=3, type='test')
    elif dataset in ['mlpq-3hop']:
        train_dataframe = pd.read_json('dataset/mlpq/Questions/fr-en/3-hop/3hop_train_question_evidences.json', lines=True)
        validation_dataframe = pd.read_json('dataset/mlpq/Questions/fr-en/3-hop/3hop_dev_question_evidences.json', lines=True)
        test_dataframe = pd.read_json('dataset/mlpq/Questions/fr-en/3-hop/3hop_test_question_evidences.json', lines=True)

        df_kg = pd.concat([train_dataframe, validation_dataframe, test_dataframe])
        kg = create_knowledge_graph_mlpq(df_kg, from_kb = False, hops=3)

        random_walk_train = RandomWalkMLPQDataset(kg, validation_dataframe, test_dataframe, steps=4, type='train')
        random_walk_dev = RandomWalkMLPQDataset(kg, validation_dataframe, test_dataframe, steps=4, type='dev')
        random_walk_test = RandomWalkMLPQDataset(kg, validation_dataframe, test_dataframe, steps=4, type='test')
    elif dataset in ['pql']:
        file_path = "dataset/pathquestion/PQ-2H.txt"
        train, val, test = load_train_test_pql_dataset(file_path, random_state = 789)
        file_path_kb = "dataset/pathquestion/2H-kb.txt"
        kg = create_knowledge_graph_pql(file_path, from_kb=False)

        random_walk_train = RandomWalkPQLDataset(kg, val, test, steps=3, type='train')
        random_walk_test = RandomWalkPQLDataset(kg, val, test, steps=3, type='test')
        random_walk_dev = RandomWalkPQLDataset(kg, val, test, steps=3, type='dev')
        
        #from torch.utils.data import ConcatDataset
        
        #random_walk_dev = ConcatDataset([random_walk_dataloader_dev, random_walk_test])
    elif dataset in ['pq-3hop']:
        file_path = "dataset/pathquestion/PQ-3H.txt"
        train, val, test = load_train_test_pql_dataset(file_path, random_state = 789)
        # file_path_kb = "dataset/pathquestion/2H-kb.txt"
        kg = create_knowledge_graph_pql(file_path, from_kb=False, hops=3)

        random_walk_train = RandomWalkPQLDataset(kg, val, test, steps=4, type='train')
        #random_walk_test = RandomWalkPQLDataset(kg, val, test, steps=3, type='test')
        random_walk_dev = RandomWalkPQLDataset(kg, val, test, steps=4, type='dev')
        random_walk_test = RandomWalkPQLDataset(kg, val, test, steps=4, type='test')
        
        #from torch.utils.data import ConcatDataset
        
        #random_walk_dev = ConcatDataset([random_walk_dataloader_dev, random_walk_test])
    else:
        raise ValueError(f"Unknown Dataset")
    print(f"#Train Random Walks: {random_walk_train}")
    print(f"#Dev Random Walks: {random_walk_dev}")
    print(f"#Test Random Walks: {random_walk_test}")
    return random_walk_train, random_walk_dev, random_walk_test
def get_parse_dataset(dataset):
    if dataset in ['2wikimultihop', 'wikimultihop', '2wikihop', 'wikihop']:
        train_dataset, dev_dataset, test_dataset, kg_train, kg_dev, kg_test = load_dataset('dataset/2wikimultihop', do_correct_wrong_evidences=True)

        all_data = pd.concat([train_dataset, dev_dataset, test_dataset])
        all_kg = create_knowledge_graph_wikimultihop(all_data)

        print(f"Nodes in Data: {len(list(all_kg.nodes()))}")

        print(f"Lenght Train Data: {len(train_dataset)}")
        print(f"Lenght Dev Data: {len(dev_dataset)}")
        print(f"Lenght Test Data: {len(test_dataset)}")

        parse_train = ParseWikHopDataset(train_dataset)
        parse_dev = ParseWikHopDataset(dev_dataset)
        parse_test = ParseWikHopDataset(test_dataset)
    elif dataset in ['metaqa']:
        df_dev = pd.read_json("dataset/metaqa/2hops/qa_dev_evidences.json")
        df_train = pd.read_json("dataset/metaqa/2hops/qa_train_evidences.json")
        df_test = pd.read_json("dataset/metaqa/2hops/qa_test_evidences.json")
        MAX_ANSWER = 1
        parse_train = ParseMetaQADataset(df_train, max_answers=MAX_ANSWER)
        parse_dev = ParseMetaQADataset(df_dev, max_answers=MAX_ANSWER)
        parse_test = ParseMetaQADataset(df_test, max_answers=MAX_ANSWER)
    elif dataset in ['mlpq']:
        validation_dataframe = pd.read_json('dataset/mlpq/Questions/fr-en/2-hop/2hop_dev_question_evidences.json', lines=True)
        train_dataframe = pd.read_json('dataset/mlpq/Questions/fr-en/2-hop/2hop_train_question_evidences.json', lines=True)
        test_dataframe = pd.read_json('dataset/mlpq/Questions/fr-en/2-hop/2hop_test_question_evidences.json', lines=True)

        parse_train = ParseMLPQDataset(train_dataframe)
        parse_dev = ParseMLPQDataset(validation_dataframe)
        parse_test = ParseMLPQDataset(test_dataframe)
    elif dataset in ['mlpq-3hop']:
        validation_dataframe = pd.read_json('dataset/mlpq/Questions/fr-en/3-hop/3hop_dev_question_evidences.json', lines=True)
        train_dataframe = pd.read_json('dataset/mlpq/Questions/fr-en/3-hop/3hop_train_question_evidences.json', lines=True)
        test_dataframe = pd.read_json('dataset/mlpq/Questions/fr-en/3-hop/3hop_test_question_evidences.json', lines=True)

        parse_train = ParseMLPQDataset(train_dataframe)
        parse_dev = ParseMLPQDataset(validation_dataframe)
        parse_test = ParseMLPQDataset(test_dataframe)
    elif dataset in ['pql']:
        file_path = "dataset/pathquestion/PQ-2H.txt"
        train, val, test = load_train_test_pql_dataset(file_path, random_state = 789)

        parse_train = ParsePQLDataset(train)
        parse_dev = ParsePQLDataset(val)
        parse_test = ParsePQLDataset(test)
    elif dataset in ['pq-3hop']:
        file_path = "dataset/pathquestion/PQ-3H.txt"
        train, val, test = load_train_test_pql_dataset(file_path, random_state = 789)
        # file_path_kb = "dataset/pathquestion/2H-kb.txt"

        parse_train = ParsePQLDataset(train)
        parse_dev = ParsePQLDataset(val)
        parse_test = ParsePQLDataset(test)
        
        #from torch.utils.data import ConcatDataset
        
        #random_walk_dev = ConcatDataset([random_walk_dataloader_dev, random_walk_test])
    else:
        raise ValueError(f"Unknown Dataset")
    return parse_train, parse_dev, parse_test
def get_parse_then_hop_test_dataset(dataset):
    if dataset in ['2wikimultihop', 'wikimultihop', '2wikihop', 'wikihop']:
        train_dataset, dev_dataset, test_dataset, kg_train, kg_dev, kg_test = load_dataset('dataset/2wikimultihop', do_correct_wrong_evidences=True)

        all_data = pd.concat([train_dataset, dev_dataset, test_dataset])
        all_kg = create_knowledge_graph_wikimultihop(all_data)

        print(f"Nodes in Data: {len(list(all_kg.nodes()))}")

        print(f"Lenght Train Data: {len(train_dataset)}")
        print(f"Lenght Dev Data: {len(dev_dataset)}")
        print(f"Lenght Test Data: {len(test_dataset)}")
        parse_then_hop_test = ParseThenHopWikiHopDataset(test_dataset)


    elif dataset in ['metaqa']:
        # df_kg = pd.read_csv("dataset/metaqa/kb.txt", sep="|")
        # kg = create_knowledge_graph_metaqa(df_kg, from_kb=True)

        df_dev = pd.read_json("dataset/metaqa/2hops/qa_dev_evidences.json")
        df_train = pd.read_json("dataset/metaqa/2hops/qa_train_evidences.json")
        df_test = pd.read_json("dataset/metaqa/2hops/qa_test_evidences.json")
        MAX_ANSWER = 1
        #df_kg = pd.concat([df_dev, df_train, df_test])
        #kg = create_knowledge_graph_metaqa(df_kg, from_kb=False, max_answers=MAX_ANSWER)
        parse_then_hop_test = ParseThenHopMetaQADataset(df_test, max_answers=MAX_ANSWER)
    elif dataset in ['mlpq']:
        #txt_file_paths = ['dataset/mlpq/Triples_in_questions/EN_KG', 'dataset/mlpq/Triples_in_questions/FR_KG']
        train_dataframe = pd.read_json('dataset/mlpq/Questions/fr-en/2-hop/2hop_train_question_evidences.json', lines=True)
        validation_dataframe = pd.read_json('dataset/mlpq/Questions/fr-en/2-hop/2hop_dev_question_evidences.json', lines=True)
        test_dataframe = pd.read_json('dataset/mlpq/Questions/fr-en/2-hop/2hop_test_question_evidences.json', lines=True)

        #df_kg = pd.concat([train_dataframe, validation_dataframe, test_dataframe])
        #kg = create_knowledge_graph_mlpq(df_kg, from_kb = False)

        parse_then_hop_test = ParseThenHopMLPQDataset(test_dataframe)
    elif dataset in ['pql']:
        file_path = "dataset/pathquestion/PQ-2H.txt"
        train, val, test = load_train_test_pql_dataset(file_path, random_state = 789)

        parse_then_hop_test = ParseThenHopPQLDataset(test)
    elif dataset in ['pq-3hop']:
        file_path = "dataset/pathquestion/PQ-3H.txt"
        train, val, test = load_train_test_pql_dataset(file_path, random_state = 789)

        parse_then_hop_test = ParseThenHopPQLDataset(test)
    else:
        raise ValueError(f"Unknown Dataset")
    return parse_then_hop_test
def get_knowledge_integration_dataset(dataset):
    pass