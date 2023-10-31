
import networkx as nx
import itertools
import numpy as np
import pickle
import hanlp
import os
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


terms = set()
cache_paths = {}
test_steps_in_kg = set()

def lemmatize(word, tag):
    ''' Change a word to its general form; note that this process is not required in Chinese.

    :param word: the input word
    :param tag: lexical property
    :return: the general form
    '''
    lemmatizer = WordNetLemmatizer()
    if tag.startswith('NN'):
        return lemmatizer.lemmatize(word, 'n')
    elif tag.startswith('VB'):
        return lemmatizer.lemmatize(word, 'v')
    elif tag.startswith('JJ'):
        return lemmatizer.lemmatize(word, 'a')
    elif tag.startswith('R'):
        return lemmatizer.lemmatize(word, 'r')
    else:
        return word

def get_terms():
    if len(terms) > 0:
        return
    with open('DomainConcept.txt', encoding="utf-8") as file:
        line = file.readline()
        while(line):
            if line.strip() != '':
                tmps = line.strip().split(" ")
                if len(tmps) > 1:
                    terms.add(tuple(tmps))
                else:
                    terms.add(line.strip())
            line = file.readline()

# init NLP tool
get_terms()
HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH)
tok = HanLP['tok/fine']
tok.dict_combine = terms

function_list = []


defined_abbr = {
    'query': 'q',
    'domain': 'd',
    'test function': 'tf',
    'test step': 'ts',
    'test case': 'tc',
    'function param': 'fp'
}


defined_relation_weight = {
    'synonym': 1,
    'hyponymy': 1/2,
    'association': 1/3,
    'related_to_dc': 1,
    'containment': 1,
    'implementation': 1
}

defined_meta_paths = {
    'q-d-tf': ['query', 'domain', 'test function'],
    'q-d-d-tf': ['query', 'domain', 'domain', 'test function'],
    'q-d-fp-tf': ['query', 'domain', 'function param', 'test function'],
    'q-d-d-fp-tf': ['query', 'domain', 'domain', 'function param', 'test function'],
    'q-d-ts-tf': ['query', 'domain', 'test step', 'test function'],
    'q-d-d-ts-tf': ['query', 'domain', 'domain', 'test step','test function']
}



def calc_path_score(graph, path, defined_weight):
    ''' Calculate the score of a path instance based on the defined relationship weights.

    :param graph: the test-oriented knowledge graph
    :param path: the path instance matched the meta-path
    :param defined_weight: the defined relationship weights
    :return: the score of the path instance
    '''
    paths = [path[i:i+2] for i in range(len(path)-1)]
    score = 1
    for item in paths:
        ptype = graph[item[0]][item[1]]['type']
        score *= defined_weight[ptype]
    return score

def get_meta_path_results(graph, source, target, meta_paths):
    ''' Based on the source node, target node and the defined meta-paths, find the corresponding path instances and calculate the score.

    :param graph: the test-oriented knowledge graph
    :param source: the source node in test-oriented knowledge graph
    :param target: the target node in test-oriented knowledge graph
    :param meta_paths: the defined meta_paths
    :return: the score between source and target
    '''
    rtn = {}
    source_used_neighbors = set()
    source_neighbors = set(graph.neighbors(source))
    target_used_neighbors = set()
    target_neighbors = set()
    for k,v in meta_paths.items():
        rtn[k] = []
    all_paths = []
    for nei in source_neighbors:
        if nei in cache_paths:
            if target in cache_paths[nei]:
                lst = cache_paths[nei][target]
                for l in lst:
                    tmp = [source]
                    tmp.extend(l)
                    all_paths.append(tmp)

    for path in all_paths:
        tmps = []
        for n in path:
            tmp = defined_abbr[graph.nodes[n]['type']]
            tmps.append(tmp)
        pattern = '-'.join(tmps)
        if pattern in meta_paths:
            score = calc_path_score(graph, path, defined_relation_weight)
            rtn[pattern].append((path, score))
            source_used_neighbors.add(path[1])
            last_domain_index = len(path)-2
            if tmps[last_domain_index] == 'd':
                target_used_neighbors.add(path[last_domain_index])

    for item in graph.neighbors(target):
        if graph.nodes[item]['type'] == 'domain':
            target_neighbors.add(item)
    score = 0
    path_count = 0
    for k,paths in rtn.items():
        for p in paths:
            score += p[1]
            path_count += 1
    if path_count > 0:
        score = score / path_count
    if len(source_neighbors) > 0:
        score = score * (len(source_used_neighbors) * 1.0 / len(source_neighbors))
    if len(target_neighbors) > 0:
        score = score * (len(target_used_neighbors) * 1.0 / len(target_neighbors))
    return score

    
def rec_test_functions(graph, query, functions, topK=10):
    ''' recommend test functions for the query using a test-oriented knowledge graph.

    :param graph: the test-oriented knowledge graph
    :param query: the query
    :param functions: the test functions to be recommended
    :param topK: return topK recommendation results
    :return: recommendation results
    '''

    if len(test_steps_in_kg) == 0:
        for node in graph.nodes():
            if graph.nodes[node]['type'] == 'test step':
                test_steps_in_kg.add(node)

    graph.add_node(query)
    graph.nodes[query]['type'] = 'query'

    r = HanLP(query, tasks='tok')
    words = nltk.pos_tag(r['tok/fine'])
    for word in words:
        word = lemmatize(word[0], word[1])
        word = word.strip()
        if graph.has_node(word):
            graph.add_edge(query, word)
            graph[query][word]['type'] = 'related_to_dc'
    
    score_list = []
    for func in functions:
        score = get_meta_path_results(graph, query, func, defined_meta_paths)
        score_list.append(score)
    topk_f_idx = np.argsort(-np.array(score_list))
    ranked_candidate_functions = []
    for idx in topk_f_idx:
        ranked_candidate_functions.append(functions[idx])
    
    return ranked_candidate_functions[0:topK]




    

def build_cache(graph, defined_meta_paths, filename = 'path_cache'):
    ''' build cache for path instances using meta-paths

    :param graph: the test-oriented knowledge graph
    :param defined_meta_paths: defined meta-paths
    :param filename: cached filename
    :return:
    '''
    if os.path.exists(filename):
        return
    cache = {}
    cut = 0
    reverse_patterns = {}
    for metaname,metapath in defined_meta_paths.items():
        if cut < len(metapath) - 2:
            cut = len(metapath) - 2
        reverse_metapath = metapath[::-1]
        tmps = []
        for n in reverse_metapath:
            tmp = defined_abbr[n]
            tmps.append(tmp)
        tmps.pop()
        reverse_pattern = '-'.join(tmps)
        reverse_patterns[reverse_pattern] = metaname

    for node in graph.nodes():
        if graph.nodes[node]['type'] == 'test function':
            function_name = node
            for t in graph.nodes():
                r = nx.all_simple_paths(graph, source=function_name, target=t, cutoff=cut)
                for v in r:
                    k = v[-1]
                    tmps2 = []
                    for n2 in v:
                        tmp2 = defined_abbr[graph.nodes[n2]['type']]
                        tmps2.append(tmp2)
                    pattern = '-'.join(tmps2)
                    if pattern in reverse_patterns:
                        if k not in cache:
                            cache[k] = {}
                        if function_name not in cache[k]:
                            cache[k][function_name] = []
                        cache[k][function_name].append(v[::-1])



    with open(filename, 'wb') as f:
        pickle.dump(cache, f, pickle.HIGHEST_PROTOCOL)
    print("cache builded")


if __name__ == '__main__':

    # kg.gpickle is the test-oriented knowledge graph with pickle format
    with open("kg.gpickle", 'rb') as f:
        G = pickle.load(f)
    # function_list is the list of test functions
    function_list = ['create_sdh', 'create_eth', 'create_eth_path', 'query_sdh']
    # create a path cache file before recommending test functions
    build_cache(G, defined_meta_paths)
    # load the path cache
    with open("path_cache", 'rb') as f:
        cache_paths = pickle.load(f)
    # recommend test functions
    queries = [
        'create an EPL service', 
        'create an SDH service', 
        'confirm whether the service was created successfully',
        'precompute an Ethernet service'
    ]
    for query in queries:
        print('query: ' + query)
        print(rec_test_functions(G, query, function_list, 10))
        print('------------')
        

   