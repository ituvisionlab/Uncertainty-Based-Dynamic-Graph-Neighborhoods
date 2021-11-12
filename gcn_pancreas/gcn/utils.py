import numpy as np
import scipy.sparse as sp
import torch
import gcn_pancreas.path_config as dirs


def add_feature(add_vol, ft_vol):
    axis = len(add_vol.shape)
    new_fts = np.expand_dims(add_vol, axis=axis)
    ret_fts = np.concatenate((ft_vol, new_fts), axis=axis)
    return ret_fts
def add_feature_torch(add_vol, ft_vol):
    axis = len(add_vol.shape)
    new_fts = torch.unsqueeze(add_vol, dim=axis)
    ret_fts = torch.cat((ft_vol, new_fts), dim=axis)
    return ret_fts


def map_voxel_nodes(shape, include_nodes):
    """
    Assigns one node id to the valid nodes of a volume with certain shape.
    Parameters
    ----------
    shape : tuple
        The original shape of the input volume
    include_nodes : numpy.ndarray
        A binary array used to indicate the voxels that should be considered in the graph

    Returns
    -------
    tuple
        A tuple containing a dictionary tha maps voxels coordinates to node index and an array containing the map from
        node index to 3D coordinates.

    """

    ys, xs, zs = shape
    # N = np.sum(include_nodes.astype(np.int))
    N = np.sum(include_nodes)
    voxel_node = {}
    node_voxel = np.zeros(shape=(N, 3), dtype=np.int)
    node_index = 0
    for y in range(ys):
        for x in range(xs):
            for z in range(zs):
                if not include_nodes[y, x, z]:
                    continue
                node_voxel[node_index] = [y, x, z]
                voxel_node[y, x, z] = node_index
                node_index += 1
    return voxel_node, node_voxel

def reference_to_graph(vol_torch, node_voxel):

    vol = vol_torch.clone().detach().cpu().numpy()
    return vol[node_voxel[:, 0], node_voxel[:, 1], node_voxel[:, 2]]


def graph_fts(fts, node_voxel):

    return fts[node_voxel[:,0],node_voxel[:,1],node_voxel[:,2],:]


def vol_graph_map(shape, include_nodes):
    ys = shape[0]
    xs = shape[1]
    zs = shape[2]
    N = ys * xs * zs
    graph_vox = np.zeros(shape=(N, 3), dtype=np.int32)  # a map between node index and voxel position.
    for z in range(zs):
        for y in range(ys):
            for x in range(xs):
                if include_nodes is not None and not include_nodes[y, x, z]:
                    continue
                node_idx = z * ys * xs + y * xs + x
                graph_vox[node_idx] = [y, x, z]
    return graph_vox


def reconstruct_from_n6(ft_mat, map_vector, shape, dtype=np.uint8):
    ys, xs, zs = shape
    N = map_vector.shape[0]
    rec_vol = np.zeros(shape=(ys, xs, zs), dtype=dtype)
    for i in range(N):
        y, x, z = map_vector[i]
        rec_vol[y, x, z] = dtype(ft_mat[i])
    return rec_vol
def reconstruct_from_n6_fin(ft_mat, map_vector, shape, dtype=np.uint8):
    zs,ys, xs = shape
    N = map_vector.shape[0]
    rec_vol = np.zeros(shape=(zs, ys, xs,), dtype=dtype)
    for i in range(N):
        z, x, y = map_vector[i]
        rec_vol[z, x, y] = dtype(ft_mat[i])
    return rec_vol


def show_mask_graph(y_val, map_vector, mask):
    ys, xs, zs = map_vector[-1] + 1
    N = y_val.shape[0]
    rec_vol = np.zeros(shape=(ys, xs, zs), dtype=np.uint8)
    for i in range(N):
        rec_vol[map_vector[i, 0], map_vector[i, 1], map_vector[i, 2]] = np.uint8(
            int(mask[i] > 0) * (np.uint8(y_val[i]) * 255 + (1 - np.uint8(y_val[i])) * 100))
    return rec_vol


# Getting uncertain elements
def generate_mask(unc_vol, node_voxel, th=0):

    return 1.0*(unc_vol[node_voxel[:,0],node_voxel[:,1],node_voxel[:,2]] > th).squeeze()


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(graph_packet, path=""):
    print("Loading graph data")
    val_portion = 0.2#***

    graph = graph_packet['graph']
    weights = graph_packet['weights']
    features = graph_packet['ft_graph']
    labels = graph_packet['lb']
    test_mask = graph_packet['mask']
    y_test = graph_packet['ref_lb']

    full_mask = 1 - test_mask  #***  # Elements that will be used for training the model

    #labels = np.load(labels_path)  # All the predicted (from model) labels are included here.
    num_nodes = labels.shape[0]
    adj = sp.coo_matrix((weights, (graph[:, 0], graph[:, 1])), shape=(num_nodes, num_nodes))
    features = sp.coo_matrix(features)

    working_nodes = np.where(full_mask != 0)[0]#***
    random_arr = np.random.uniform(low=0, high=1, size=working_nodes.shape)#***

    features = normalize(features)



    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    idx_train = working_nodes[random_arr > val_portion]#***
    idx_val = working_nodes[random_arr <= val_portion]#***
    idx_test = np.where(test_mask != 0)#***

    print("Num nodes: {}".format(num_nodes))
    print("Num of uncertain nodes: {}.".format(np.sum(test_mask)))
    print("Num certain nodes: {}.".format(np.sum(full_mask)))
    print("Num of positive samples: {}".format(torch.sum(labels[np.where(full_mask != 0)[0]] == 1)))
    print("Num of negative samples: {}".format(torch.sum(labels[np.where(full_mask != 0)[0]] == 0)))

    features = torch.FloatTensor(np.array(features.todense()))
    labels = labels.long()
    y_test = torch.FloatTensor(y_test[:, 0])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test[0])

    return adj, features, labels, y_test, idx_train, idx_val, idx_test, torch.from_numpy(graph.T).long(), torch.from_numpy(weights),


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(adj):
    """Row-normalize sparse matrix"""
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def normalize_adj2(adj):
    """Row-normalize sparse matrix for directed graphs"""
    rowsum = np.array(adj.sum(1))
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    return d_mat_inv.dot(adj).tocoo()


def accuracy(output, labels):
    preds = (output > 0.5).type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx
