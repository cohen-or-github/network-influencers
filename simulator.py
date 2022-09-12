import numpy as np
from friendship_predictor import graph_predictor
from copy import deepcopy


def prob_per_user(num_players, neighbours, infected_list):
    """
    :param user_id: the current user
    :param num_players: the num of artist's players
    :param neighbours: list of user's friends
    :param infected_list: list of the nodes infected
    :return: the probability the user would be infected in the next round
    """
    num_of_neighbours = len(neighbours)
    infected_counter = 0
    for i in range(num_of_neighbours):
        if neighbours[i] in infected_list:
            infected_counter += 1
    if num_players == 0:
        p = infected_counter / float(num_of_neighbours)
    else:
        p = (infected_counter * float(num_players)) / (num_of_neighbours * 1000)
    return p


def prob_dict(G, infected_list):
    """
    :param G: the graph
    :param infected_list: list of infected nodes
    :return: a dict with the probability of each node to get infected
    """
    p_dict = {}
    for node in G.nodes:
        p = prob_per_user(G.nodes[node]["sound"], list(G.adj[node]), infected_list)
        p_dict[node] = p
    return p_dict


def simulator_per_group(influencers_list, G, singer):
    '''
    :param influencers_list: the starting group of influencers to begin with
    :param G: the graph at time 0
    :param singer: the given singer
    :return: the number of people infected by starting with this group
    '''
    infected = deepcopy(influencers_list)
    for i in range(6):
        print(i)
        probability_dict = prob_dict(G, infected)
        for node in G.nodes:
            random_num = np.random.rand()
            if random_num <= probability_dict[node] and node not in infected:
                infected.append(node)
        G = graph_predictor(G, singer)
    return infected, len(infected)

