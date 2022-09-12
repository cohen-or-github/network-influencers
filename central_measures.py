import networkx as nx
from simulator import simulator_per_group


def max_degree(degree_iter):
    """
    :param degree_iter: iterator of nodes degree
    :return: the node with the maximum degree
    """
    max_degree = 0
    node_id = None
    for pair in degree_iter:
        if pair[1] > max_degree:
            max_degree = pair[1]
            node_id = pair[0]
    return node_id, max_degree


def clustring_co(G, node):
    """
    :param G: graph
    :param node: a node
    :return: the node's CC
    """
    all_neighbours_counter = 0
    overlap_counter = 0
    for node1 in list(G[node].keys()):
        all_neighbours_counter += 1
        for node2 in list(G[node].keys()):
            if node1 in list(G[node2].keys()):
                overlap_counter += 1
    overlap_counter /= 2
    return overlap_counter / all_neighbours_counter


def closeness(G, index, distance_mat):
    """
    :param G: graph
    :param index: the current node index in the graph's nodes list
    :param distance_mat: a distance matrix
    :return: the CC of the node
    """
    n = len(G.nodes)
    distances_sum = 0
    for i in range(n):
        distances_sum += distance_mat[index][i]
    return (n-1) / distances_sum


def harmonic(G, index, distance_mat):
    """
    :param G: graph
    :param index: the current node index in the graph's nodes list
    :param distance_mat: a distance matrix
    :return: the HC of the node
    """
    n = len(G.nodes)
    distances_sum = 0
    for i in range(n):
        distances_sum += (1 / distance_mat[index][i])
    return (1/(n-1)) * distances_sum


def relative_degree_dict(G):
    """
    :param G: graph
    :return: normalized nodes' degree dictionary
    """
    degree_dict = {}
    for node in G.nodes:
        degree_dict[node] = G.degree[node]
    degrees_sum = sum(degree_dict.values())
    for node in G.nodes:
        degree_dict[node] = degree_dict[node] / degrees_sum
    return degree_dict


def find_influencers(G, singer):
    """
    :param G: given graph
    :param singer: given singer
    :return: 5 user ids to start publishing the singer from
    """
    measures_dict = {}
    influencers = []
    degree_dict = relative_degree_dict(G)
    for node in G.nodes:
        measures_dict[node] = 0.5 * degree_dict[node] + 0.5 * nx.closeness_centrality(G, node)
    temp_list = sorted(measures_dict.items(), key=lambda x: x[1], reverse=True)[:3]
    temp_list = list(zip(*temp_list))[0]
    influencers.append(temp_list[0])
    influencers.append(temp_list[1])
    influencers.append(temp_list[2])
    infected, infected_num = simulator_per_group(influencers, G, singer)
    for node in infected:
        measures_dict[node] = 0
    temp_list2 = sorted(measures_dict.items(), key=lambda x: x[1], reverse=True)[:2]
    temp_list2 = list(list(zip(*temp_list2))[0])
    influencers = influencers + temp_list2
    return influencers
