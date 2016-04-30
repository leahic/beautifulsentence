# -*- coding: utf-8 -*-
# Licensed under the GNU GPLv2 - http://www.gnu.org/licenses/gpl-2.0.html

import numpy as np
import networkx as nx

def get_similarity(s1, s2):

    co_occur_num = float( len( set(s1) & set(s2) ) )

    if abs(co_occur_num) <= 1e-12:
        return 0.
    
    denominator = np.log(float(len(s1))) + np.log(float(len(s2)))
    
    if abs(denominator) <= 1e-12:
        return 0.
    
    return co_occur_num / denominator

def solve(sentences, sim_func = get_similarity, pagerank_config = {'alpha': 0.85,}):
    sentences_num = len(sentences)        
    graph = np.zeros((sentences_num, sentences_num))
    
    for x in range(sentences_num):
        for y in range(x, sentences_num):
            similarity = sim_func( sentences[x], sentences[y] )
            graph[x, y] = similarity
            graph[y, x] = similarity
            
    nx_graph = nx.from_numpy_matrix(graph)
    scores = nx.pagerank(nx_graph, **pagerank_config)
    result = sorted( scores.items() , key = lambda x : x[0] )
    return [ value  for index , value in result ]