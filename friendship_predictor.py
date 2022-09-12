from copy import deepcopy
import numpy as np
import networkx as nx
from create_graph import user_artist_graph, graph_per_file


# first try - without defining strong edges
def triadic_closure(G, singer):
    """
    :param G: graph
    :param singer: the singer to find influencers for
    :return: new graph with all the edges that existed before & after the current run
    """
    G1 = deepcopy(G)
    for user in G.nodes:
        if G[user] != {} and user != "userID":
            for conc1 in G[user].keys():
                for conc2 in G[user].keys():
                    if conc2 != conc1 and G.has_edge(conc2, conc1) == False and conc2 != "userID" and conc1 != "userID":
                        G1.add_edge(conc1, conc2)
    for edge in G.edges:
        if edge[0] != "userID" and edge[1] != "userID":
            G1.add_edge(edge[0], edge[1])
    return G1


def new_edges_added(G_t_1, G_t):
    """
    :param G_t_1: the graph at time t-1
    :param G_t: the graph at point t
    :return: list of all new edges between time t-1 and time t
    """
    new_edge = []
    for edge in G_t.edges:
        if edge not in G_t_1.edges:
            new_edge.append(edge)
    return new_edge


def not_added_by_predicted_graph(G1, G0, singer, predicted):
    """
    :param G1: the graph at time -1
    :param G0: the graph at time 0
    :param singer: the singer to find influencers for
    :param predicted: the predicted graph
    :return: a graph with all edges not added by the predicted graph
    """
    new_graph = deepcopy(G1)
    for node in G1.nodes:
        new_graph.add_node(deepcopy(node))
    for edge in G0.edges:
        if edge not in predicted.edges:
            new_graph.add_edge(edge[0], edge[1])
    return new_graph


def added_only_by_predicted_graph(G1, G0, singer, predicted):
    """
    :param G1: the graph at time -1
    :param G0: the graph at time 0
    :param singer: the singer to find influencers for
    :param predicted: the predicted graph
    :return: a graph with all edges not added by the predicted graph
    """
    new_graph = deepcopy(G1)
    for node in G1.nodes:
        new_graph.add_node(deepcopy(node))
    for edge in predicted.edges:
        if edge not in G0.edges:
            new_graph.add_edge(edge[0], edge[1])
    return new_graph


def predicted_error(singer, predicted):
    """
    :param singer: the singer
    :param predicted: the predicted graph
    """
    G1 = graph_per_file(singer, 'instaglam_1.csv')
    G0 = graph_per_file(singer, 'instaglam0.csv')

    # check which edges were in G0 but not predicted
    not_predicted = not_added_by_predicted_graph(G1, G0, singer, predicted)
    error1 = len(list(not_predicted.edges))

    # check which edges were predicted by triadic closure but weren't in G0
    predicted_false = added_only_by_predicted_graph(G1, G0, singer, predicted)
    error2 = len(list(predicted_false.edges))

    return error1, error2


# trying to find connection between edges made and artists played (homogenity)
def find_tendency(singer):
    """
    :param singer: the current singer
    :return: void
    """
    G1 = graph_per_file(singer, 'instaglam_1.csv')
    G0 = graph_per_file(singer, 'instaglam0.csv')
    new_edges = new_edges_added(G1, G0)
    new_graph = user_artist_graph()
    counter1 = 0
    counter2 = 0
    mean_true = 0
    mean_false = 0

    # find the difference in the number of different artist played
    for edge in new_edges:
        if edge[0] in new_graph.nodes:
            counter1 += 1
            mean_true += len(new_graph.nodes[edge[0]]['art']) - len(new_graph.nodes[edge[1]]['art'])
    for edge in triadic_closure(G1, singer).edges:
        if edge not in G0:
            if edge[0] in new_graph.nodes:
                counter2 += 1
                mean_false += len(new_graph.nodes[edge[0]]['art']) - len(new_graph.nodes[edge[1]]['art'])

    # print(mean_true / counter1, mean_false / counter2)
    # got: mean_true = 0.119, mean_false = 0.251

    # find the number of mutual artist
    num_of_mutual_true = 0
    num_of_mutual_false = 0
    num_true = []
    num_false = []
    for edge in new_edges:
        if edge[0] in new_graph.nodes:
            num_of_mutual_true = 0
            for artist in new_graph.nodes[edge[0]]['art']:
                if artist in new_graph.nodes[edge[1]]['art']:
                    num_of_mutual_true += 1
            num_true.append(num_of_mutual_true)
    for edge in triadic_closure(G1, singer).edges:
        if edge not in G0:
            if edge[0] in new_graph.nodes:
                num_of_mutual_false = 0
                for artist in new_graph.nodes[edge[0]]['art']:
                    if artist in new_graph.nodes[edge[1]]['art']:
                        num_of_mutual_false += 1
                num_false.append(num_of_mutual_false)
    # print(median(num_true), median(num_false))
    # got 7, 3 - stronger chance to bond if nodes have around 7 mutual artists

    # find the number of mutual friends
    true_mutual_neighbours = []
    for edge in new_edges:
        true_mutual_neighbours.append(len(list(nx.common_neighbors(G1, edge[0], edge[1]))))

    false_mutual_neighbours = []
    for edge in triadic_closure(G1, singer).edges:
        if edge not in G0:
            false_mutual_neighbours.append(len(list(nx.common_neighbors(G1, edge[0], edge[1]))))

    # print ("true neighbours mean:", mean(true_mutual_neighbours), " false neighbours mean:", mean(false_mutual_neighbours))
    # we got that all the nodes that connected at t0 had at least 2 mutual friends and in average 5,
    # and all the nodes that were predicted by triadic closure but didn't connect had 1-40 friends (mean: 0.08)

    # check the number of friends of each connected node
    true_num_of_neighbours = []
    true_num = 0
    for edge in new_edges:
        true_num = 0
        first_node_neighbours = G1.adj[edge[0]]
        second_node_neighbours = G1.adj[edge[1]]
        for neighbour in first_node_neighbours:
            true_num += 1
        for neighbour in second_node_neighbours:
            true_num -= 1
        true_num_of_neighbours.append(abs(true_num))

    false_num_of_neighbours = []
    false_num = 0
    for edge in triadic_closure(G1, singer).edges:
        if edge not in G0:
            false_num = 0
            first_node_neighbours = G1.adj[edge[0]]
            second_node_neighbours = G1.adj[edge[1]]
            for neighbour in first_node_neighbours:
                false_num += 1
            for neighbour in second_node_neighbours:
                false_num -= 1
            false_num_of_neighbours.append(abs(true_num))

    # we got that the diffrence in the number of friends between connected nodes were 0 - 114
    # and between those who were predicted but didn't connect the diffrence is exactly 7

    # finding if people with large number of friends are more likely to connect
    true_num_of_friends = {}
    general_num_of_friends = {}

    for node1 in G1:
        for node2 in G1:
            general_num_of_friends[len(list(G1.adj[node1])), len(list(G1.adj[node2]))] = 0
            true_num_of_friends[len(list(G1.adj[node1])), len(list(G1.adj[node2]))] = 0

    for edge in new_edges:
        true_num_of_friends[len(list(G1.adj[edge[0]])), len(list(G1.adj[edge[1]]))] += 1

    for node1 in G1:
        for node2 in G1:
            general_num_of_friends[len(list(G1.adj[node1])), len(list(G1.adj[node2]))] += 1
    prob_num_of_friends = {}
    for key in general_num_of_friends.keys():
        if key in true_num_of_friends:
            prob_num_of_friends[key] = true_num_of_friends[key] / general_num_of_friends[key]

    # print(sorted(prob_num_of_friends.items(), key=lambda x:x[1], reverse=True))


# function that adds probability according to nodes' number of friends
# decided to drop since it gave worse results
'''
def prob_num_of_friends(G1, G0, friends_num_one, friends_num_two, num_of_friends_dict):
    """
    :param G1: the graph at time -1
    :param G0: the graph at time 0
    :param friends_num_one: number of one of the nodes friend
    :param friends_num_two: number of the second node friends
    :param num_of_friends_dict: a dictionary with number of apperances for each number of friends tuple
    :return:
    """
    new_edges = new_edges_added(G1, G0)
    true_counter = 0
    for edge in new_edges:
        if (len(list(G1.adj[edge[0]])) == friends_num_one and len(list(G1.adj[edge[1]])) == friends_num_two) \
                or (len(list(G1.adj[edge[0]])) == friends_num_two and len(list(G1.adj[edge[1]])) == friends_num_one):
            true_counter += 1

    if (friends_num_one, friends_num_two) in num_of_friends_dict \
            and num_of_friends_dict[(friends_num_one, friends_num_two)] != 0:
        return true_counter / num_of_friends_dict[(friends_num_one, friends_num_two)]
    elif (friends_num_two, friends_num_one) in num_of_friends_dict \
            and num_of_friends_dict[(friends_num_two, friends_num_one)] != 0:
        return true_counter / num_of_friends_dict[(friends_num_two, friends_num_one)]

    return 1
'''


def create_predictor(G1, G0, singer, prob_dict):
    """
    :param G1: the graph at time -1
    :param G0: the graph at time 0
    :param singer: the given singer
    :param prob_dict: a dict with probabilities for edge to be created
    :return: a predicted graph at time 0 (without edges from time -1)
    """
    predicted = deepcopy(G1)

    # add edges by probability computed
    for node1 in predicted.nodes:
        for node2 in predicted.nodes:
            if [node1, node2] not in G1.edges and node1 != node2:
                num_of_mutual = len(list(nx.common_neighbors(G1, node1, node2)))
                rand = np.random.rand()
                if num_of_mutual > 0:
                    if rand < (np.log(num_of_mutual) / np.log(float(10**27))) - 0.007:
                        predicted.add_edge(node1, node2)
    return predicted


def graph_predictor_zero(singer):
    '''
    :param singer: the given singer
    :return: the dictionary of probabilities found
    this function finds a method that approximately predicts G0's new edges
    '''
    G1 = graph_per_file(singer, "instaglam_1.csv")
    G0 = graph_per_file(singer, "instaglam0.csv")
    mutual_friends_dict = {}
    true_mutual_friends_dict = {}
    prob_dict = {}

    # initializing the dicts
    for i in range(12):
        mutual_friends_dict[i] = 0
        true_mutual_friends_dict[i] = 0
        prob_dict[i] = 0

    # updating dicts values
    for node1 in G1.nodes:
        for node2 in G1.nodes:
            if node1 != node2 and [node1, node2] not in G1.edges and len(list(nx.common_neighbors(G1, node1, node2))) < 12:
                    mutual_friends_dict[(len(list(nx.common_neighbors(G1, node1, node2))))] += 1
                    if [node1, node2] in G0.edges:
                        true_mutual_friends_dict[(len(list(nx.common_neighbors(G1, node1, node2))))] += 1
    for i in range(12):
        if i in mutual_friends_dict:
            prob_dict[i] = true_mutual_friends_dict[i] / mutual_friends_dict[i]
    # predict G0
    graph_zero = create_predictor(G1, G0, singer, prob_dict)


def graph_predictor(Gt, singer):
    """
    :param Gt: the graph at time t
    :param singer: the current singer
    :return: a predicted graph for time t+1
    """
    G_t_plus_1 = deepcopy(Gt)

    # predict which edges are going to be added at time t+1
    for node1 in Gt.nodes:
        for node2 in Gt.nodes:
            if [node1, node2] not in Gt.edges and node1 != node2:
                num_of_mutual = len(list(nx.common_neighbors(Gt, node1, node2)))
                rand = np.random.rand()
                if num_of_mutual > 1:
                    if rand < (np.log(num_of_mutual) / np.log(float(10**27))) - 0.007:
                        G_t_plus_1.add_edge(node1, node2)
    return G_t_plus_1

