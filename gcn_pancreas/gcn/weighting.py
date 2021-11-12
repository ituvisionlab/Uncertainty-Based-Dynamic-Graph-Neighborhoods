import numpy as np
import scipy.sparse as sp
from torch.cuda import get_device_name
import gcn_pancreas.gcn.utils as gut
import torch

class BasicWeighting:
    def __init__(self, w_id):
        self.description = "All edges are weighted as 1"
        self.id = w_id
        self.weights = []

    def weights_for(self, idx1, idx2, args):
        self.weights.append(1)


    def get_weights(self):
        return self.weights 

    def get_id(self):
        return self.id

    @property
    def get_description(self):
        return self.description


class Weighting_v(BasicWeighting):
    def __init__(self, w_id):
        super(Weighting_v, self).__init__(w_id=w_id)
        self.description = "l*div + e(int) + e(pos)"
        self.weights1 = []
        self.weights2 = []
        self.weights3 = []
        self.testlist = []

    def weights_for(self, idx1, idx2, args): 

        prob1 = args["probability"][idx1[:,0],idx1[:,1],idx1[:,2]]
        prob2 = args["probability"][idx2[:,0],idx2[:,1],idx2[:,2]]


        int1 = args["volume"][idx1[:,0],idx1[:,1],idx1[:,2]]
        int2 = args["volume"][idx2[:,0],idx2[:,1],idx2[:,2]]

        device =prob1.device

        print(device)

        ny, nx, nz = args["volume"].shape
        dim_array =  torch.tensor([ny, nx, nz], dtype=torch.float32).to(device)
        pos1 = torch.tensor(idx1).to(device) / dim_array
        pos2 = torch.tensor(idx2).to(device)/ dim_array


        int_diff = int1 - int2
        pos_diff = pos1 - pos2
        intensity = int_diff * int_diff

        space = torch.sum(pos_diff * pos_diff,dim =1)
        p = prob1 - prob2
        delta = 1.0e-5



        lambd = p * (torch.log2(prob1 / (prob2 + delta) + delta) - torch.log2((1 - prob1) / ((1 - prob2) + delta) + delta))

        self.weights1.append(lambd)
        self.weights2.append(intensity)
        self.weights3.append(space)

    def post_process(self, edges): 

        self.weights1 = self.weights1[0]
        self.weights2 = self.weights2[0]
        self.weights3 = self.weights3[0]

        ne = self.weights1.shape[0]*1.0
        muw2 = self.weights2.sum() / ne
        muw3 = self.weights3.sum() / ne

        sig2 = 2 * torch.sum((self.weights2.reshape(-1) - muw2) ** 2) / ne
        sig3 = 2 * torch.sum((self.weights3.reshape(-1) - muw3) ** 2) / ne

        self.weights2 = torch.exp(-self.weights2 / sig2)
        self.weights3 = torch.exp(-self.weights3 / sig3)

        self.weights = 0.5 * self.weights1 + self.weights2 + self.weights3


def get_weighting_func(w_id):
    if w_id == 0:
        return BasicWeighting(w_id)
    if w_id == 1:
        return Weighting_v(w_id=1)
    return None



