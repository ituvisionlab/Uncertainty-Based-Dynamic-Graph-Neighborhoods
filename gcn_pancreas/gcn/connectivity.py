import numpy as np
import gcn_pancreas.gcn.utils as gut


def connect_n6_krandom(ref, voxel_node, node_voxel, working_nodes, k_random, weighting, args):
    edges = []
    labels = []
    num_nodes = node_voxel.shape[0]
    tabu_list = {}  # A list to avoid duplicated elements in the adjacency matrix.
    nodes_complete = {}  # A list counting how many neighbors a node already has.
    valid_nodes = np.array(np.where(working_nodes > 0))
    valid_nodes = np.transpose(valid_nodes)

    for node_idx in range(num_nodes):
        y, x, z = node_voxel[node_idx]  # getting the 3d position for current node
        labels.append(ref[y, x, z])  # Labels come from the CNN prediction
        #   Basic n6 connectivity
        for axis in range(3):
            axisy = int(axis == 0)
            axisx = int(axis == 1)
            axisz = int(axis == 2)
            for ne in [-1, 1]:
                neighbor = y + axisy*ne, x + axisx*ne, z + axisz*ne
                if neighbor not in voxel_node:
                    continue
                ne_idx = voxel_node[neighbor]
                if (node_idx, ne_idx) not in tabu_list and (ne_idx, node_idx) not in \
                        tabu_list:
                    tabu_list[(node_idx, ne_idx)] = 1  # adding the edge to the tabu list
                    weighting.weights_for((y, x, z), neighbor, args)  # computing the weight for the current pair.
                    weighting.weights_for(neighbor, (y, x, z), args)  # computing the weight for the current pair.
                    edges.append([node_idx, ne_idx])
#                   Adding the edge in the opposite direction.
                    edges.append([ne_idx, node_idx])
#       Generating random connections to current node.
        for j in range(k_random):
            valid_neigh = False
            if node_idx not in nodes_complete:
                nodes_complete[node_idx] = 0
            elif nodes_complete[node_idx] == k_random:
                break

            while not valid_neigh:
                lu_idx = np.random.randint(low=0, high=num_nodes)  # we look for a random node.
                yl, xl, zl = valid_nodes[lu_idx]  # getting the euclidean coordinates for the voxel.
                lu_idx = voxel_node[yl, xl, zl]  # getting the node index.
                if lu_idx not in nodes_complete:
                    nodes_complete[lu_idx] = 0
                    valid_neigh = True
                elif nodes_complete[lu_idx] < k_random:
                    valid_neigh = True

            if not (node_idx, lu_idx) in tabu_list and not (lu_idx, node_idx) in tabu_list \
                    and node_idx != lu_idx:  # checking if the edge was already generated
                weighting.weights_for((y, x, z), (yl, xl, zl), args)  # computing the weight for the current pair.
                weighting.weights_for((yl, xl, zl), (y, x, z), args)
                tabu_list[(node_idx, lu_idx)] = 1
                edges.append([node_idx, lu_idx])
                #  Adding the weight in the opposite direction
                edges.append([lu_idx, node_idx])
                #  Increasing the amount of neighbors connected to each node
                nodes_complete[node_idx] += 1
                nodes_complete[lu_idx] += 1
    edges = np.asarray(edges, dtype=int)
    pp_args = {
        "edges": edges,
        "num_nodes": num_nodes
    }
    weighting.post_process(pp_args)  # Applying weight post-processing, e.g. normalization
    weights = weighting.get_weights()
    edges, weights, _ = gut.sparse_to_tuple(weights)

    return edges, weights, np.asarray(labels, dtype=np.float32), num_nodes

def connect_n6_final(ref, node_voxel, weighting, args):

    num_nodes = node_voxel.shape[0]

    all_indices = np.arange(0, num_nodes)
    all_indices = node_voxel[all_indices, :]  # all y, x, z

    node_voxel_indices = np.ones((ref.shape[0]+1,ref.shape[1],ref.shape[2]), dtype=np.int) * -1
    node_voxel_indices[all_indices[:, 0], all_indices[:, 1], all_indices[:, 2]] = np.arange(0, num_nodes)

    all_labels = ref[all_indices[:, 0], all_indices[:, 1], all_indices[:, 2]]

    all_neigs_1_0_0 = all_indices + np.array([1, 0, 0]).reshape(1, 3)
    all_neigs_0_1_0 = all_indices + np.array([0, 1, 0]).reshape(1, 3)
    all_neigs_0_0_1 = all_indices + np.array([0, 0, 1]).reshape(1, 3)
    all_neigs_1_0_0_neg = all_indices + np.array([-1, 0, 0]).reshape(1, 3)
    all_neigs_0_1_0_neg = all_indices + np.array([0, -1, 0]).reshape(1, 3)
    all_neigs_0_0_1_neg = all_indices + np.array([0, 0, -1]).reshape(1, 3)

    all_neig_ids = np.concatenate((all_neigs_1_0_0.reshape(num_nodes, 1, 3),
                                   all_neigs_0_1_0.reshape(num_nodes, 1, 3),
                                   all_neigs_0_0_1.reshape(num_nodes, 1, 3),
                                   all_neigs_1_0_0_neg.reshape(num_nodes, 1, 3),
                                   all_neigs_0_1_0_neg.reshape(num_nodes, 1, 3),
                                   all_neigs_0_0_1_neg.reshape(num_nodes, 1, 3)
                                   ), axis=1)  # x,6,3

    all_neig_ids = all_neig_ids.reshape((num_nodes * 6, 3))


    neig_in_roi = node_voxel_indices[all_neig_ids[:, 0], all_neig_ids[:, 1], all_neig_ids[:, 2]]
    neig_in_roi = neig_in_roi.reshape((num_nodes, 6, 1))

    rep_indices = np.tile(np.arange(0, num_nodes).reshape((num_nodes, 1)), (1, 6)).reshape((num_nodes, 6, 1))

    all_edges = np.concatenate((rep_indices, neig_in_roi), 2).reshape((num_nodes * 6, 2))

    real_edges = all_edges[:, 1] > -1

    all_edges = all_edges[real_edges, :] #(x, 2) -> (x,3,2)

    index1 = all_edges[:,0]
    index2 = all_edges[:,1]

    index1 = node_voxel[index1]
    index2 = node_voxel[index2]

    weighting.weights_for(index1,index2, args)

    weighting.post_process(all_edges)  # Applying weight post-processing, e.g. normalization
    weights = weighting.get_weights()

    return all_edges, weights, all_labels, num_nodes

def get_connect_func(cf_id):
    if cf_id == 1:
        return connect_n6_final


    return None



