import numpy as np
import scipy.sparse as sp
import gcn_pancreas.gcn.utils as gut
import gcn_pancreas.path_config as dirs
import torch



device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


class BasicWeighting:
    def __init__(self, w_id):
        self.description = "All edges are weighted as 1"
        self.id = w_id
        self.weights = []


    def weights_for(self, idx1, idx2):
        # self.weights.append(1)
        self.weights.append(0.2)

    def post_process(self,edges):
        global num_nodes
        self.weights = np.asarray(self.weights, dtype=np.float32)
        num_nodes = num_nodes
        w1 = sp.coo_matrix((self.weights, (edges[:, 0], edges[:, 1])), shape=(num_nodes, num_nodes))
        self.weights = w1

    def get_weights(self):
        return self.weights

    def get_id(self):
        return self.id

    @property
    def get_description(self):
        return self.description



def create_weights(graph_packet, edges,node_voxel,prob,vol,path):

    # valid_nodes = np.load(path + dirs.DILATION_PATH)
    valid_nodes = graph_packet['dilated']

    num_nodes = node_voxel.shape[0]
    weighting = Weighting2(w_id=1,prob = prob,vol = vol, num_nodes =num_nodes)
    _, node_voxel = gut.map_voxel_nodes(vol.shape, valid_nodes.astype(np.bool))
    ind1 = edges[0,:]
    ind2 = edges[1, :]
    node_voxel = torch.from_numpy(node_voxel).to(device)


    weighting.weights_for(node_voxel[ind1, :], node_voxel[ind2, :])



    weighting.post_process(edges)  # Applying weight post-processing, e.g. normalization
    weights1, weights2, weights3 = weighting.get_w()

    #edges, weights, _ = gut.sparse_to_tuple(weights)

    return weights1, weights2, weights3




class Weighting2(BasicWeighting):
    def __init__(self, w_id,prob,vol,num_nodes):
        super(Weighting2, self).__init__(w_id=w_id)
        self.description = "l*div + e(int) + e(pos)"
        self.weights1 = []
        self.weights2 = []
        self.weights3 = []
        self.prob = prob #torch.from_numpy(prob).to(device)
        self.vol = vol #torch.from_numpy(vol).to(device)
        self.num_nodes = num_nodes


    def weights_for(self,idx1, idx2):

        old_shape = self.prob.shape


        prob_1d = self.prob.reshape(-1)
        vol_1d = self.vol.reshape(-1)
        idx1_1d = idx1[:,2] + old_shape[2]*idx1[:,1] + old_shape[1]*old_shape[2]*idx1[:,0]
        idx2_1d = idx2[:, 2] + old_shape[2] * idx2[:, 1] + old_shape[1] * old_shape[2] * idx2[:, 0]

        prob1 = prob_1d[idx1_1d] #<class 'tuple'>: (1353580,)
        prob2 = prob_1d[idx2_1d]

        int1 = vol_1d[idx1_1d]
        int2 = vol_1d[idx2_1d]
        ny, nx, nz = self.vol.shape

        dim_array = torch.tensor([ny, nx, nz], dtype=torch.float32).to(device)

        pos1 = torch.tensor(idx1.clone().detach(),dtype=torch.float32).to(device) / dim_array
        pos2 = torch.tensor(idx2.clone().detach(), dtype=torch.float32).to(device)/ dim_array


        # print(pos1.max(),pos1.min(),pos2.max(),pos2.min() )
#       Computing the weight
        int_diff = int1 - int2
        pos_diff = pos1 - pos2
        intensity = int_diff * int_diff
        space = torch.sum(pos_diff * pos_diff,dim=1)
        p = prob1 - prob2
        delta = 1.0e-15
        lambd = p * (torch.log2(prob1 / (prob2 + delta) + delta) - torch.log2((1 - prob1) / ((1 - prob2) + delta) + delta))
        self.weights1.append(lambd) #TODO numpy to tensor
        self.weights2.append(intensity) #TODO numpy to tensor
        self.weights3.append(space)
    '''
    weights = torch.cat((weights1.reshape(-1,1).float(),weights2.reshape(-1,1).float(),weights3.reshape(-1,1).float()),1).float()
    RuntimeError: All input tensors must be on the same device. Received cpu and cuda:0
    '''
    def post_process(self, edges):


        self.weights1 = self.weights1[0]/100
        self.weights2 = self.weights2[0]
        self.weights3 = self.weights3[0]
        num_nodes = self.num_nodes
        ne = float(self.weights1.shape[0])
        muw2 = self.weights2.sum() / ne
        muw3 = self.weights3.sum() / ne


        sig2 = 2 * torch.sum((self.weights2.reshape(-1) - muw2) ** 2) / ne
        sig3 = 2 * torch.sum((self.weights3.reshape(-1) - muw3) ** 2) / ne

        self.weights2 = torch.exp(-self.weights2 / sig2)
        self.weights3 = torch.exp(-self.weights3 / sig3)



    def get_w(self):
        return self.weights1,self.weights2,self.weights3
