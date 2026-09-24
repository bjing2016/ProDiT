import numpy as np
#import obonet
import networkx as nx

def get_go_graph():
    # download the GO ontology 
    url = 'http://purl.obolibrary.org/obo/go/go.obo'
    graph = obonet.read_obo(url) 
    return graph

def get_go_type(graph, go_id):
    """ 
    Returns the type of the go_id (e.g. molecular_function, cellular_component, biological_process)
    """
    return graph.nodes[go_id]['namespace']

def get_go_term_paths(graph, go_id, go_type=[]):

    if go_id not in graph: # go_id not in obo graph or go_id not in go_type
        return None  
    
    # find all paths from the specified GO term to the root nodes
    paths = []
    for root in (n for n, d in graph.out_degree() if d == 0):  
        if root not in graph:
            continue  # skip 
        for path in nx.all_simple_paths(graph, source=go_id, target=root):
            paths.append(path)  # source is now go_id

    paths_to_keep = []
    if len(go_type) > 0: # if we want to only consider paths with certain go_types (e.g. molecular_function)
        for path in paths:
            in_type = True
            for i in range(len(path)): # id in path:
                id = path[i]
                if get_go_type(graph, id) not in go_type:
                    in_type = False
            if in_type:
                paths_to_keep.append(path) # keep only paths where all go_ids are of the specified go_type(s)
        return paths_to_keep
    return paths    

def get_relationship_between_go_terms(graph, go_id1, go_id2):
    """ 
    Returns relationship go_id1 --> go_id2 (order matters)
    """
    # check if the terms exist in the graph
    if go_id1 not in graph or go_id2 not in graph:
        return None
    
    # iterate over all relationships in term2
    for parent, relationship_type in graph[go_id1].items():
        if parent == go_id2:
            for relation in relationship_type: # {'is_a': {}}
                return relation
    
    # if no relationship is found, return None
    return None

def filter_path_by_relationship_type(graph, path, relationship_types=[]):
    """ 
    Filters path (list of GO terms) by relationship type (e.g. is_a, part_of); path goes from 
    child to root

    Will return path up until the first relationship that is not in relationship_types
    """
    filtered_path = []
    for i in range(len(path)):
        if i < len(path) - 1:
            relationship = get_relationship_between_go_terms(graph, path[i], path[i+1])
            if relationship is not None and relationship in relationship_types:
                filtered_path.append(path[i])
            else: 
                break
    return filtered_path

def simplify_paths(paths):
    """
    Given paths of go terms, returns the set of unique go terms from those paths (unordered)
    """
    unique_terms = set(go_id for subpath in paths for go_id in subpath)
    return list(unique_terms)

def get_associated_go_terms(graph, go_ids, go_type=[], relationship_types=[]):
    """
    Function that takes the list of go terms associated with a protein, gets their paths (just for specific
    relationship_type and go category) and gets the unique list of terms for its paths
    """
    all_paths = [] # all paths for all go terms
    for go_id in go_ids: # for each go_id in the list of go terms associated with this protein
        paths = get_go_term_paths(graph, go_id, go_type) # get all the paths (with some GO type)
        if paths is not None:
            if len(relationship_types) > 0: # if we are filtering by graph relationship types
                new_paths = []
                for path in paths:
                    new_paths.append(filter_path_by_relationship_type(graph, path, relationship_types))
                paths = new_paths
            all_paths.extend(paths)

    # simplify final paths 
    unique_terms = simplify_paths(all_paths)

    if len(unique_terms) == 0:
        return None
    return unique_terms

def parse_res_go_data_(data_str):
    entries = {}
    current_entry_id = None
    lines = data_str.strip().split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith(">"):
            current_entry_id = line[1:] 
            entries[current_entry_id] = []
            i += 2 # skip the sequence line
        elif line.startswith("GO:"):
            parts = line.split()
            if len(parts) == 3:
                go_id = parts[0]
                start = int(parts[1])
                end = int(parts[2])
                entries[current_entry_id].append({
                    "go_id": go_id,
                    "start": start,
                    "end": end
                })
            i += 1
        else:
            i += 1 # skip lines that are not entry IDs or GO terms

    return entries
