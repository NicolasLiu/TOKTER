
import json
import OpenHowNet
import hanlp
import networkx as nx
import pickle
import numpy as np


hownet_dict = OpenHowNet.HowNetDict(init_babel=True)

terms = set()
syn_set = set()
syn_set_tmp = set()
hyponym_words = {}   
association_relations = {}
association_set_tmp = set()



def get_terms():
    if len(terms) > 0:
        return
    with open('DomainConcept.txt', encoding="utf-8") as file:
        line = file.readline()
        while(line):
            if line.strip() != '':
                terms.add(line.strip())
            line = file.readline()
    

def get_synonym():
    for t in terms:
        results = hownet_dict.get_synset(t)
        for r in results:
            for s in r.en_synonyms:
                if s == t:
                    continue
                if s in terms:
                    tmp1 = s + " " + t
                    tmp2 = t + " " + s
                    if tmp1 not in syn_set_tmp and tmp2 not in syn_set_tmp:
                        syn_set.add(tmp1)
                        syn_set_tmp.add(tmp1)
                        syn_set_tmp.add(tmp2)
            for s in r.zh_synonyms:
                if s == t:
                    continue
                if s in terms:
                    tmp1 = s + " " + t
                    tmp2 = t + " " + s
                    if tmp1 not in syn_set_tmp and tmp2 not in syn_set_tmp:
                        syn_set.add(tmp1)
                        syn_set_tmp.add(tmp1)
                        syn_set_tmp.add(tmp2)

    print("Get Synonym: %d" % len(syn_set))

def save_synonym():
    with open('Synonym.txt', "w", encoding="utf-8") as file:
        for t in syn_set:
            file.write(t+"\n")
    print("Write Synonym.txt %d" % len(syn_set))

def get_nn_amod():
    corpus = []
    tosave = []
    with open('corpus.txt', encoding="utf-8") as file:
        line = file.readline()
        while(line):
            if line.strip() != '':
                corpus.append(line.strip())
            line = file.readline()
    HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH)
    tok = HanLP['tok/fine']
    tok.dict_force = terms
    for text in corpus:
        r = HanLP(text, tasks='dep')
        for i in range(len(r['dep'])):
            if r['dep'][i][1] == 'nn' or r['dep'][i][1] == 'amod':
                a_i = r['dep'][i][0] - 1
                b_i = i
                a = r['tok/fine'][a_i]
                b = r['tok/fine'][b_i]
                a_in = 0
                ab_in = 0
                if a in terms:
                    a_in = 1
                if b+a in terms:
                    ab_in = 1
                if b_i + 1 == a_i and a_in == 1 and ab_in == 1:
                    tosave.append(b+a + " " + r['dep'][i][1] + " " + b + " " + a)
    with open('nn_amod_out.txt', "w", encoding="utf-8") as file:
        for line in tosave:
            file.write(line + "\n")


def get_hyponymy():
    # nn amod
    with open('nn_amod_out.txt', encoding="utf-8") as file:
        line = file.readline()
        while(line):
            if line.strip() != '':
                tmps = line.strip().split()
                if tmps[0] not in hyponym_words:
                    hyponym_words[tmps[0]] = [0, tmps[3]]
                hyponym_words[tmps[0]][0] += 1
            line = file.readline()
    print("hyponym_words size: %d" % len(hyponym_words))

def save_hyponymy():
    with open('Hyponymy.txt', "w", encoding="utf-8") as file:
        for k,v in hyponym_words.items():
            line = k + " " + v[1] + "\n"
            file.write(line)
    print("Write Hyponymy.txt %d" % len(hyponym_words))


def get_dobj_nsubj():
    corpus = []
    tosave = []
    with open('corpus.txt', encoding="utf-8") as file:
        line = file.readline()
        while(line):
            if line.strip() != '':
                corpus.append(line.strip())
            line = file.readline()
    HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH)
    tok = HanLP['tok/fine']
    tok.dict_force = terms
    for text in corpus:
        r = HanLP(text, tasks='dep')
        for i in range(len(r['dep'])):
            if r['dep'][i][1] == 'dobj':
                a_i = r['dep'][i][0] - 1
                b_i = i
                a = r['tok/fine'][a_i]
                b = r['tok/fine'][b_i]
                a_in = 0
                b_in = 0
                if a in terms:
                    a_in = 1
                if b in terms:
                    b_in = 1
                if a_in == 1 and b_in == 1:
                    print(a,r['dep'][i][1],b,flush=True)
            if r['dep'][i][1] == 'nsubj':
                b_i = r['dep'][i][0] - 1
                a_i = i
                a = r['tok/fine'][a_i]
                b = r['tok/fine'][b_i]
                a_in = 0
                b_in = 0
                if a in terms:
                    a_in = 1
                if b in terms:
                    b_in = 1
                if a_in == 1 and b_in == 1:
                    tosave.append(a + " " + r['dep'][i][1] + " " + b)
    with open('dobj_nsubj_out.txt', "w", encoding="utf-8") as file:
        for line in tosave:
            file.write(line + "\n")

def get_association():
    with open('dobj_nsubj_out.txt', encoding="utf-8") as file:
        line = file.readline()
        while(line):
            if line.strip() != '':
                tmps = line.strip().split()
                a = tmps[0].strip().lower()
                rl = tmps[1].strip().lower()
                b = tmps[2].strip().lower()
                tmp1 = a + " " + b
                tmp2 = b + " " + a
                if tmp1 not in association_set_tmp and tmp2 not in association_set_tmp:
                    association_relations[tmp1] = 1
                    association_set_tmp.add(tmp1)
                    association_set_tmp.add(tmp2)
                else:
                    if tmp1 in association_relations:
                        association_relations[tmp1] += 1
                    if tmp2 in association_relations:
                        association_relations[tmp2] += 1
                
            line = file.readline()

    print("association_relations: %d" % len(association_relations))


def save_association():
    with open('Association.txt', "w", encoding="utf-8") as file:
        for k,v in association_relations.items():
            line = k + " " + str(v) + "\n"
            file.write(line)
    print("Write Association.txt %d" % len(association_relations))


def build_kg():
    get_terms()
    
    get_synonym()
    save_synonym()

    get_nn_amod()
    get_hyponymy()
    save_hyponymy()

    get_dobj_nsubj()
    get_association()
    save_association()

    G = nx.Graph()

    # Load domain concept entity
    for t in terms:
        G.add_node(t)
        G.nodes[t]['type'] = 'domain'
    domain_count = G.number_of_nodes()
    print("Load domain: %d" % domain_count)

    # Load Synonym Relation
    with open('Synonym.txt', "r", encoding="utf-8") as file:
        line = file.readline()
        count = 0
        while line:
            tmps = line.split(' ')
            a = tmps[0].lower().strip()
            b = tmps[1].lower().strip()
            if not G.has_node(a) or not G.has_node(b):
                line = file.readline()
                continue
            G.add_edge(a, b)
            G[a][b]['type'] = 'synonym'
            count += 1
            line = file.readline()
    synonym_count = count
    print("Load synonym: %d" % synonym_count)

    # Load Hyponymy Relation
    with open('Hyponymy.txt', "r", encoding="utf-8") as file:
        line = file.readline()
        count = 0
        while line:
            tmps = line.split(' ')
            a = tmps[0].lower().strip()
            b = tmps[1].lower().strip()
            if not G.has_node(a) or not G.has_node(b):
                line = file.readline()
                continue
            if G.has_edge(a,b):
                print(line)
            G.add_edge(a, b)
            G[a][b]['type'] = 'hyponymy'
            G[a][b]['high'] = b
            G[a][b]['low'] = a
            count += 1
            line = file.readline()
    hyponymy_count = count
    print("Load hyponymy: %d" % hyponymy_count)

    # Load Association Relation
    with open('Association.txt', "r", encoding="utf-8") as file:
        line = file.readline()
        count = 0
        while line:
            tmps = line.split(' ')
            a = tmps[0].lower().strip()
            b = tmps[1].lower().strip()
            w = int(tmps[2].lower().strip())
            if not G.has_node(a) or not G.has_node(b):
                line = file.readline()
                continue
            if not G.has_edge(a,b):
                G.add_edge(a, b)
                G[a][b]['type'] = 'association'
                G[a][b]['weight'] = w
                count += 1
            line = file.readline()
    association_count = count
    print("Load association: %d" % association_count)

    domain_edge_count = G.number_of_edges()
    print("Load all Domain Knowledge, node %d edge %d" % (domain_count, domain_edge_count))

    related_to_count = 0
    contain_count = 0
    implement_count = 0

    # Load test function entity
    print("Loading test function")
    function_list = np.load("function_list.npy", allow_pickle=True).tolist()

    for func in function_list:
        fname = func['name'].strip()
        fdesc = func['desc']
        G.add_node(fname)
        G.nodes[fname]['type'] = 'test function'
        G.nodes[fname]['description'] = fdesc
        r = HanLP(fdesc, tasks='tok')
        for word in r['tok/fine']:
            word = word.strip()
            if G.has_node(word):
                G.add_edge(fname, word)
                G[fname][word]['type'] = 'related_to_dc'
                related_to_count += 1
 
    # Load function parameter entity
    print("Loading function param")
    with open("function_param.json", "r", encoding="utf-8") as f:
        function_param = json.load(f)
        for k,v in tqdm(function_param.items()):
            fname = k
            params = v['param']
            for k1,v1 in params.items():
                pname = fname + " " + k1
                pdesc = v1['description'].strip().lower()
                pvalues = v1['values']
                G.add_node(pname)
                G.nodes[pname]['type'] = 'function param'
                G.nodes[pname]['description'] = pdesc
                G.add_edge(fname, pname)
                G[fname][pname]['type'] = 'containment'
                for pv in pvalues:
                    pv = pv.strip().lower()
                    if G.has_node(pv):
                        G.add_edge(pname, pv)
                        G[pname][pv]['type'] = 'related_to_dc'
                        related_to_count += 1
                tmps = pdesc.split(" ")
                if G.has_node(k1):
                    G.add_edge(pname, k1)
                    G[pname][k1]['type'] = 'related_to_dc'
                    related_to_count += 1
                for tmp in tmps:
                    tmp = tmp.strip().lower()
                    if G.has_node(tmp):
                        G.add_edge(pname, tmp)
                        G[pname][tmp]['type'] = 'related_to_dc'
                        related_to_count += 1

    # Load test step entity
    print("Loading test step")
    with open("train.data", "rb") as f:
        train_data = pickle.load(f)
    for item in train_data:
        step = item['operate']
        if not G.has_node(step):
            G.add_node(step)
            G.nodes[step]['type'] = 'test step'
            r = HanLP(step, tasks='tok')
            for word in r['tok/fine']:
                word = word.strip()
                if G.has_node(word):
                    G.add_edge(step, word)
                    G[step][word]['type'] = 'related_to_dc'
                    related_to_count += 1
            for f in item['function']:
                if not G.has_node(f):
                    print("error")
                    print(item)
                    print(f)
                G.add_edge(step, f)
                G[step][f]['type'] = 'implementation'
                implement_count += 1
    print("Loaded test step")




    # save the knowledge graph 
    with open("kg.gpickle", 'wb') as f:
        pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
    




        

if __name__ == '__main__':
    
    build_kg()






