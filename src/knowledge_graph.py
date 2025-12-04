import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import pickle

def create_knowledge_graph_pql(file_path : str, from_kb = False, hops= 2):
    G = nx.MultiDiGraph()
    if from_kb:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                # Remove whitespace at the start and end
                line = line.strip()
                
                # Split the line by tabs
                parts = line.split('\t')
                
                # Check we have at least three parts
                if len(parts) == 3:
                    head = parts[0]
                    relation = parts[1]
                    tail = parts[2]
                    G.add_edge(head, tail, key=relation, relation=relation)
                else:
                    print(f"Length of line != 3: {line}")
    else:
        with open(file_path, "r", encoding="utf-8") as f:  # "rb" mode means "read binary"
            for line in f:
                line = line.strip()
                splitted_line = line.split('\t')
                path = splitted_line[2]
                path = path.split('#')
                for i in range(hops):
                    head = path[2*i]
                    relation = path[2*i+1]
                    tail = path[2*i+2]

                    #i = 0 --> 0,1,2
                    #i= 1 --> 2,3,4
                    #i=2--> 4,5,6
                    G.add_edge(head, tail, key=relation, relation=relation) 
    return G
def create_knowledge_graph_mlpq(txt_file_paths : list, from_kb = True, hops = 2):
    G = nx.MultiDiGraph()
    if from_kb:
        for txt_file_path in txt_file_paths:
            with open(txt_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    # Remove whitespace at the start and end
                    line = line.strip()
                    
                    # Split the line by tabs
                    parts = line.split('@@@')
                    
                    # Check we have at least three parts
                    if len(parts) == 3:
                        head = parts[0].lower()
                        relation = parts[1].lower()
                        tail = parts[2].lower()
                        G.add_edge(head, tail, key=relation, relation=relation)
                    else:
                        print(f"Length of line != 3: {line}")
    else:
        for evidence in txt_file_paths['evidences']:
            for i in range(hops):
                head = evidence[2*i]
                relation = evidence[2*i+1]
                tail = evidence[2*i+2]

                #i = 0 --> 0,1,2
                #i= 1 --> 2,3,4
                #i=2--> 4,5,6

                G.add_edge(head, tail, key=relation, relation=relation) 
            

    return G

#Need to use a undirected Graph here otherwise we will not learn the paths neccessarry for the questions
# Example we have a question: which person wrote the films directed by [Yuriy Norshteyn]	Sergei Kozlov

#But in the Graph it is:
# Film --> directed by --> Yuriy Norshteyn
# Film --> written_by --> Sergei Kozlov
def create_knowledge_graph_metaqa(data : pd.DataFrame, iterations = 0, from_kb = True, max_answers = 1):
    index = 0
    G = nx.MultiDiGraph()
    if from_kb:
        for idx, row in tqdm(data.iterrows(), desc="Creating Knowledge Graph...", total=len(data) if iterations == 0 else iterations, dynamic_ncols=True):
            if index > iterations:
                return G
            head = row['entity1'].strip().lower()
            relation = row['relation'].strip().lower()
            tail = row['entity2'].strip().lower()
            if relation == "has_tags":
                continue
            G.add_edge(head, tail, key=relation, relation=relation)
            if head != tail:
                G.add_edge(tail, head, key=f"{relation}_reversed", relation=f"{relation}_reversed")
            if iterations > 0:
                index += 1
    else:
        for evidences_list in tqdm(data['evidences']):
            if max_answers is None or len(evidences_list) <= max_answers:
                for evidence in evidences_list:
                    entity1, relation1, entity2, relation2, entity3 = evidence
                    G.add_edge(entity1, entity2, key=relation1, relation=relation1)
                    G.add_edge(entity2, entity3, key=relation2, relation=relation2)
        
    # print(max_count)    
    return G
def create_knowledge_graph_wikimultihop(data, iterations = 0):
    index = 0
    G = nx.MultiDiGraph()
    for entry in data['evidences']:
        if index > iterations:
            return G
        for triples in entry:
            head = triples[0].strip()
            relation = triples[1].strip()
            tail = triples[2].strip()
            G.add_edge(head, tail, key=relation, relation=relation)
        if iterations > 0:
            index += 1
            
    return G
from collections import defaultdict

def visualize_knowledge_graph(kg):
    # Generate positions for all nodes
    pos = nx.spring_layout(kg)

    # Draw the nodes and edges
    nx.draw(
        kg, pos,
        with_labels=True,
        node_size=1500,
        node_color="lightblue",
        font_size=7,
        font_weight="bold",
        edge_color="gray"
    )

    # For a MultiGraph/MultiDiGraph, multiple edges may exist between the same nodes.
    # We need to combine their 'relation' attributes into a single label.
    edge_relations = defaultdict(list)
    for u, v, data in kg.edges(data=True):
        # Extract the relation attribute (if it exists)
        relation = data.get('relation', '')
        edge_relations[(u, v)].append(relation)

    # Combine multiple relations into a single label per node pair
    # You could separate them by commas, newlines, or any other delimiter
    edge_labels = {edge: "\n".join(rels) for edge, rels in edge_relations.items()}

    # Draw the combined edge labels
    nx.draw_networkx_edge_labels(kg, pos, edge_labels=edge_labels, font_color='red', font_size=7)

    # Show the plot
    plt.show()

    
def print_graph(kg):
    print("Nodes:", kg.nodes())
    print("Edges:", kg.edges(data=True))
    print(f"#Nodes: {len(kg.nodes())}")
    print(f"#Edges: {len(kg.edges(data=True))}")

#################################################
#METAQA HOW TO GET THE EVIDENCES
#We use the qtype txt files which kind of gives us the path to the answer
import networkx as nx
import re
import pandas as pd
from collections import Counter
import itertools
from tqdm import tqdm
import ast
import os

# Mapping from (source_type, target_type) to knowledge graph relations
# '_reversed' indicates traversal in the reverse direction (from target to source)
import networkx as nx
import re
import pandas as pd
from collections import Counter
import itertools
from tqdm import tqdm
import ast
import os

# Mapping from (source_type, target_type) to knowledge graph relations
# '_reversed' indicates traversal in the reverse direction (from target to source)
TRANSITION_TO_RELATION = {
    ('movie', 'language'): 'in_language',
    ('movie', 'year'): 'release_year',
    ('movie', 'writer'): 'written_by',
    ('movie', 'director'): 'directed_by',
    ('movie', 'genre'): 'has_genre',
    ('movie', 'actor'): 'starred_actors',

    ('language', 'movie'): 'in_language_reversed',
    ('year', 'movie'): 'release_year_reversed',
    ('writer', 'movie'): 'written_by_reversed',
    ('director', 'movie'): 'directed_by_reversed',
    ('genre', 'movie'): 'has_genre_reversed',
    ('actor', 'movie'): 'starred_actors_reversed',
}

def extract_starting_entity(question):
    """
    Extracts the starting entity from the question.
    Assumes the entity is enclosed within square brackets [ ].
    """
    match = re.search(r'\[(.*?)\]', question)
    if match:
        return match.group(1).strip().lower()
    else:
        return None

def split_answers(answer_str):
    """
    Splits the answer string by '|' and trims whitespace.
    """
    return [ans.strip().lower() for ans in answer_str.split('|')]

def validate_triplet(G, entity1, relation, entity2, debug=False):
    """
    Validates whether a triplet (entity1, relation, entity2) exists in the knowledge graph.

    Args:
        G (networkx.Graph): The knowledge graph.
        entity1 (str): Source entity.
        relation (str): Relation name.
        entity2 (str): Target entity.
        debug (bool): If True, print debug statements.

    Returns:
        bool: True if the triplet exists, False otherwise.
    """
    if debug:
        print(f"Validating triplet: ({entity1}, {relation}, {entity2})")
    if G.has_edge(entity1, entity2):
        edge_attrs = G.get_edge_data(entity1, entity2)
        if G.is_multigraph():
            # For MultiGraphs, check if any edge has the specified relation
            for key, attrs in edge_attrs.items():
                if attrs.get('relation', '').strip().lower() == relation.lower():
                    if debug:
                        print(f"  Triplet exists via edge {key}: ({entity1}, {relation}, {entity2})")
                    return True
        else:
            # For simple graphs, check the relation attribute
            rel_attr = edge_attrs.get('relation', '').strip().lower()
            if rel_attr == relation.lower():
                if debug:
                    print(f"  Triplet exists: ({entity1}, {relation}, {entity2})")
                return True
    if debug:
        print(f"  Triplet does NOT exist: ({entity1}, {relation}, {entity2})")
    return False

def find_paths_with_relations(G, start, end, relation_path, debug=False):
    """
    Given a relation path (e.g., 'writer_to_movie_to_genre'), maps it to knowledge graph relations,
    traverses the KG starting from 'start', follows the relations in sequence, and finds paths to 'end'.

    Args:
        G (networkx.Graph): The knowledge graph.
        start (str): The starting entity.
        end (str): The target entity.
        relation_path (str): The relation path from the txt file (e.g., 'writer_to_movie_to_genre').
        debug (bool): If True, print debug statements.

    Returns:
        List of tuples: Each tuple contains (path, relation_sequence)
    """
    # Split the relation path into transitions
    parts = relation_path.split('_to_')  # ['writer', 'movie', 'genre']
    transitions = list(zip(parts[:-1], parts[1:]))  # [('writer', 'movie'), ('movie', 'genre')]

    # Map transitions to relations using the provided mapping
    relations = []
    for src, tgt in transitions:
        relation = TRANSITION_TO_RELATION.get((src.lower(), tgt.lower()))
        if not relation:
            if debug:
                print(f"  Warning: No relation mapping found for transition ({src}, {tgt}). Skipping path.")
            return []  # Invalid path due to missing mapping
        relations.append(relation)

    # Define the relation sequence
    relation_sequence = relations  # e.g., ['written_by_reversed', 'has_genre']

    # Initialize list to collect paths
    paths_rel = []

    # Initialize stack for DFS: (current_node, current_path, relations_left)
    stack = [(start, [start], relation_sequence.copy())]

    while stack:
        current_node, current_path, relations_left = stack.pop()
        if debug:
            print(f"Traversing from '{current_node}' with relations left: {relations_left}")

        if not relations_left:
            if current_node == end:
                # Collect the relation sequence
                relation_seq = tuple(current_path[i] for i in range(1, len(current_path), 2))
                paths_rel.append((current_path.copy(), relation_seq))
                if debug:
                    print(f"  Found valid path: {current_path}")
            continue

        next_relation = relations_left[0]
        if debug:
            print(f"  Next relation to traverse: '{next_relation}'")


        actual_relation = next_relation

        # Traverse based on direction
        neighbors = G.successors(current_node)

        for neighbor in neighbors:
            if G.has_edge(current_node, neighbor):
                edge_attrs = G.get_edge_data(current_node, neighbor)
                
                # If current_node is the same as neighbor, add reversed edge attributes
                if current_node == neighbor:
                    reversed_edge_attrs = {}
                    for key, attrs in edge_attrs.items():
                        # Create a new key with '_reversed' appended
                        new_key = f"{key}_reversed"
                        
                        # Create a new attributes dictionary with 'relation' modified
                        new_attrs = {}
                        for attr_key, attr_value in attrs.items():
                            if attr_key == 'relation' and isinstance(attr_value, str):
                                new_attrs[attr_key] = f"{attr_value}_reversed"
                            else:
                                new_attrs[attr_key] = attr_value  # Keep other attributes unchanged
                        
                        # Add the modified key and attributes to the reversed_edge_attrs dictionary
                        reversed_edge_attrs[new_key] = new_attrs
                    
                    # Merge the reversed_edge_attrs into the original edge_attrs
                    edge_attrs.update(reversed_edge_attrs)
                
                # Debugging output
                if debug:
                    print(f"edge_attrs = {edge_attrs}")
                    print(f"actual_relation = {actual_relation}")
            else:
                continue  # No such edge

            if G.is_multigraph():
                # Check if any edge has the required relation
                if any(attrs.get('relation', '').strip().lower() == actual_relation.lower() for attrs in edge_attrs.values()):
                    # Find the specific edge
                    for key, attrs in edge_attrs.items():
                        if attrs.get('relation', '').strip().lower() == actual_relation.lower():
                            if debug:
                                print(f"  Current Node: '{current_node}', Neighbor: '{neighbor}', Relation: '{next_relation}'")
                                
                            # Handle self-loop: if current_node == neighbor and relation is reversed, use forward relation
                            if current_node == neighbor and next_relation.endswith('_reversed'):
                                adjusted_relation = actual_relation
                                #print(f"  Adjusting relation for self-loop from '{next_relation}' to '{adjusted_relation}'")
                            else:
                                adjusted_relation = actual_relation
                                
                            # Append relation and neighbor
                            current_path_extended = current_path.copy()
                            current_path_extended.append(adjusted_relation)
                            current_path_extended.append(neighbor)
                            if debug:
                                print(f"  Extending path: {current_path_extended}")
                                
                            # For incoming, triplet is (neighbor, relation, current_node)
                            if validate_triplet(G, current_node, adjusted_relation, neighbor, debug=debug):
                                stack.append((neighbor, current_path_extended, relations_left[1:].copy()))

                            break  # Only need to find one matching edge

    return paths_rel

def process_questions(question_answer_file, relation_path_file, G, n):
    """
    Processes each question-answer pair, reads the corresponding relation path from the txt file,
    maps the relations, validates triplets in the knowledge graph, and constructs evidences.

    Args:
        question_answer_file (str): Path to the TSV file containing questions and answers.
        relation_path_file (str): Path to the TXT file containing relation paths per question.
        G (networkx.Graph): The knowledge graph.
        n (int): The exact number of relations (hops) to consider.

    Returns:
        pandas.DataFrame: DataFrame containing the question, answers, and evidences.
    """
    # Read the relation path file
    with open(relation_path_file, 'r') as f:
        relation_paths = f.read().splitlines()

    # Read the question-answer file
    df = pd.read_csv(question_answer_file, sep='\t', header=None, names=['question', 'answer'])

    # Check if the number of questions matches the number of relation paths
    if len(df) != len(relation_paths):
        raise ValueError("The number of questions and relation paths do not match.")

    # Lists to store DataFrame columns
    questions = []
    answers_list = []
    evidences_list = []

    # Iterate over each row in the DataFrame along with the corresponding relation path
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Questions"):
        question = row['question']
        answer_str = row['answer'].lower()
        relation_path = relation_paths[index].strip()  # e.g., 'writer_to_movie_to_genre'

        # Extract the starting entity from the question
        start_entity = extract_starting_entity(question)
        if not start_entity:
            print(f"  No starting entity found in question: '{question}'. Skipping.\n")
            continue  # Skip to the next row

        # Split the answers by '|'
        answers = split_answers(answer_str)
        if not answers:
            print(f"  No answers found for question: '{question}'. Skipping.\n")
            continue  # Skip if no answers

        # Determine if debugging should be enabled
        debug = False#(start_entity == 'malcolm x')

        # For each answer, find the path based on the relation path
        evidences = []
        all_valid = True  # Flag to ensure all answers have valid paths

        for answer in answers:
            # Find the path using the relation path
            paths_rel = find_paths_with_relations(G, start_entity, answer, relation_path, debug=debug)
            if not paths_rel:
                print(f"    No valid path found for answer '{answer}' with relation path '{relation_path}'.")
                all_valid = False
                break  # Skip this question if any answer lacks a valid path
            else:
                # Since the relation path is predefined, we expect only one path
                path, relation_seq = paths_rel[0]
                evidences.append(path)  # Append the full path including entities and relations

        if not all_valid:
            print(f"  Skipping question due to missing paths for some answers: '{question}'.\n")
            continue  # Skip adding this question

        # Append data to lists
        questions.append(question)
        answers_list.append(answers)
        evidences_list.append(evidences)

    # Create the DataFrame
    results_df = pd.DataFrame({
        'question': questions,
        'answers': answers_list,
        'evidences': evidences_list
    })

    return results_df

def display_outgoing_relations(G, entity):
    """
    Displays all outgoing relations from a specified entity.

    Args:
        G (networkx.Graph): The knowledge graph.
        entity (str): The entity to inspect.
    """
    if G.has_node(entity):
        print(f"Outgoing relations from '{entity}':")
        for neighbor in G.neighbors(entity):
            edge_attrs = G.get_edge_data(entity, neighbor)
            if G.is_multigraph():
                for key, attrs in edge_attrs.items():
                    print(f"  --[{attrs.get('relation', 'no_relation')}]--> {neighbor}")
            else:
                print(f"  --[{edge_attrs.get('relation', 'no_relation')}]--> {neighbor}")
    else:
        print(f"Entity '{entity}' does not exist in the knowledge graph.")
