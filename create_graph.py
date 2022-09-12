import numpy as np
import networkx as nx
import random


def graph(singer):
    """
    :param singer: the singer's id
    :return: a graph with all friendships and number of players of the given singer for each id
    """
    G = nx.Graph()
    all_user = open('spotifly.csv', 'r')
    file0 = open('instaglam0.csv', "r")
    rows0 = file0.readlines()
    file_1 = open('instaglam_1.csv', "r")
    rows_1 = file_1.readlines()

    # add the number of artist's players per id
    for row in all_user.readlines():
        if row.split(",")[0] == "userID":
            continue
        if row.split(",")[0] not in G.nodes:
            G.add_node(row.split(",")[0])
            G.nodes[row.split(",")[0]]["sound"] = 0
        if row.split(",")[1] == singer:
            G.nodes[row.split(",")[0]]["sound"] = max(G.nodes[row.split(",")[0]]["sound"],
                                                      int(row.split(",")[2].split('\n')[0]))

    # add friendships as edges
    for row in rows_1:
        if row.split(",")[0] == "userID":
            continue
        G.add_edge(row.split(",")[0], row.split(",")[1].split('\n')[0])

    for row in rows0:
        if row.split(",")[0] == "userID":
            continue
        G.add_edge(row.split(",")[0], row.split(",")[1].split('\n')[0])
    return G


def user_artist_graph():
    """
    :return: a graph with the artist connected to the user_ID's who listen to them
    """
    G = nx.Graph()
    file_1 = open('spotifly.csv', 'r')
    rows_1 = file_1.readlines()
    for row in rows_1:
        if row.split(",")[0] == "userID":
            continue
        if row.split(",")[0] not in G.nodes:
            G.add_node(row.split(",")[0])
            G.nodes[row.split(",")[0]]["art"] = [row.split(",")[1]]
        else:
            G.nodes[row.split(",")[0]]["art"].append(row.split(",")[1])
    return G


def sound_error(G, singer):
    """
    :param G: graph
    :param singer: current singer
    :return: a graph without key error for 'sound'
    """
    all_user = open('spotifly.csv', 'r')
    for node in G.nodes:
        G.nodes[node]["sound"] = 0
        for row in all_user.readlines():
            if row.split(",")[0] == node and row.split(",")[1] == singer:
                G.nodes[row.split(",")[0]]["sound"] = max(G.nodes[row.split(",")[0]]["sound"],
                                                          int(row.split(",")[2].split('\n')[0]))
    return G


def graph_per_file(singer, path):
    """
    :param singer: given singer
    :param path: the path to the files
    :return: a graph at given time
    """
    G = nx.Graph()
    file_1 = open(path, "r")
    rows_1 = file_1.readlines()
    for row in rows_1:
        if row.split(",")[0] == "userID":
            continue
        if row.split(",")[0] not in G.nodes:
            G.add_node(row.split(",")[0])
            G.nodes[row.split(",")[0]]["sound"] = 0
        if row.split(",")[1] == singer:
            G.nodes[row.split(",")[0]]["sound"] = max(G.nodes[row.split(",")[0]]["sound"],
                                                      int(row.split(",")[2].split('\n')[0]))
    for row in rows_1:
        if row.split(",")[0] == "userID":
            continue
        G.add_edge(row.split(",")[0], row.split(",")[1].split('\n')[0])
    return G