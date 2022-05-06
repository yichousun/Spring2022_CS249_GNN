import os
from graph import EdgeTable
from collections import defaultdict
import numpy as np


################################ Preprocessed .edge files analysis #################################
file_path_train = os.path.join(os.getcwd(), 'data/' + '{}_user2item.train_p2.edges'.format('ml-1m'))
file_path_eval = os.path.join(os.getcwd(), 'data/' + '{}_user2item.test_p2.edges'.format('ml-1m'))

ui_edges_train = EdgeTable.load(file_path_train)
ui_edges_eval = EdgeTable.load(file_path_eval)

"""
# (idx, val) = (edge_idx, UserID)
print("src: ", max(ui_edges_eval.src))   # length = 300001, ('src: ', array([2445, 4424, 4353, ..., 3264, 5076, 2591]))
# (idx, val) = (edge_idx, MovieID)
print("dest: ", max(ui_edges_eval.dest)) # length = 300001, ('dest: ', array([3248, 1732, 3530, ...,   35, 1032, 2597]))
# print("data: ", ui_edges_eval.data) # ('data: ', None)
# print("n_nodes: ", ui_edges_eval.n_nodes)   # ('n_nodes: ', 6040)

# (idx, val) = (UserID, edge_idx)
print("**TRAIN** _src_index: ", len(ui_edges_train._src_index))
# (idx, val) = (MovieID, edge_idx)
print("**TRAIN** _dest_index: ", len(ui_edges_train._dest_index))

# (idx, val) = (UserID, edge_idx)
print("**EVAL** _src_index: ", len(ui_edges_eval._src_index))
# (idx, val) = (MovieID, edge_idx)
print("**EVAL** _dest_index: NONE")
"""


################################ Without Rating #################################
"""
Only consider edges, can use .edge files & HashGNN structure
"""
# construct the adjacency list
def get_edge_list(ui_edges):
    """
    Construct the adjacency list
    :param ui_edges: EdgeTable.load(file_path)
    :return: adjacency list [(UserID-1, MovieID-1)]
    """
    srcs, dests = ui_edges.src, ui_edges.dest
    edge_list = []
    for i in range(len(srcs)):
        edge_list.append((srcs[i], dests[i]))
    return edge_list


edge_list_train = get_edge_list(ui_edges_train)
edge_list_eval = get_edge_list(ui_edges_eval)


################################ With Rating ###################################
"""
Consider edges, process ratings.data file
"""
# process ratings.data
def get_ratings(filename):
    """
    Process ratings.data file
    :param filename: ratings.dat
    :return: a dict of (key, val) = (UserID, [(MovieID, Rating)])
    """
    maxUserId, maxMovieID = 0, 0
    rating_dict = defaultdict(list)
    with open(filename) as fp:
        line = fp.readline()
        while line:
            UserID, MovieID, Rating, Timestamp = line.split("::")
            maxUserId, maxMovieID = max(maxUserId, int(UserID)), max(maxMovieID, int(MovieID))
            rating_dict[int(UserID)].append((int(MovieID), int(Rating)))
            line = fp.readline()

    rating_matrix = np.zeros((maxUserId+1, maxMovieID+1))
    for uid in range(1, maxUserId+1):
        for iid, rating in rating_dict[uid]:
            rating_matrix[uid][iid] = rating

    return rating_matrix


# test: 1::1193::5::978300760
rating_matrix = get_ratings("ml-1m/ratings.dat")
print("Rating should be 5: ", rating_matrix[1][1193])
print("Rate should be 0: ", rating_matrix[0][1193])

