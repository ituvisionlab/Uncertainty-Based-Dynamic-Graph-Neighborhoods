import torch
import torch.nn as nn
import torch.nn.functional as F
from gcn_pancreas.gcn.layers import GraphConvolution
from gcn_pancreas.new_weight2 import create_weights
from torch_geometric.utils import degree
from torch_geometric.datasets import ZINC
from torch_geometric.data import DataLoader
from torch_geometric.nn import PNAConv, BatchNorm, global_add_pool, GCNConv, TransformerConv
from torch.nn import Sequential, ReLU, Linear
from torch.nn import Embedding
from torch_geometric.nn import fps

def knn(x, x_neig, k=20):

	orig_node_count = x.shape[0]
	neig_node_count = x_neig.shape[0]
	feature_size = x.shape[1]

	all_neigs = torch.zeros((orig_node_count, k)).cuda()

	step = 1000

	for i in range(orig_node_count//step):

		start = i*step
		end = (i+1)*step


		if i == x.shape[0]//step -1:
			end = x.shape[0]

		x_part = x[start:end,:]#.reshape(-1,1, feature_size)

		inner = -2 * torch.matmul(x_part, x_neig.transpose(1, 0))

		xxp = torch.sum(x_neig ** 2, dim=1, keepdim=True)

		pairwise_distance = -(xxp.transpose(1, 0) + inner + torch.sum(x_part ** 2, dim=1, keepdim=True))

		_, indices = torch.topk(pairwise_distance, dim=-1, sorted=True, k=k)

		all_neigs[start:end, :] = indices

	return all_neigs


def get_graph_feature(x_orig, x_neig, k=10):

	num_points = x_orig.shape[0]

	idx = knn(x_orig, x_neig, k=k) 


	idx += x_orig.shape[0]

	edges_x = torch.arange(0, num_points).reshape((num_points, 1)).repeat((1,k)).cuda()

	idx = idx.reshape(1,-1)
	edges_x = edges_x.reshape(1,-1)

	edgelist = torch.cat((edges_x.long(),idx.long()),0)
	edgelist2 = torch.cat((idx.long(),edges_x.long()),0)

	new_edges = torch.cat((edgelist,edgelist2),1)

	return new_edges, idx

class GCN(nn.Module):
	def __init__(self, nfeat, nhid, degree):
		super(GCN, self).__init__()


		aggregators = ['mean', 'min', 'max', 'std']
		scalers = ['identity', 'amplification', 'attenuation']

		self.conv1 = PNAConv(in_channels=nfeat, out_channels=nhid // 2,
							 aggregators=aggregators, scalers=scalers, deg=degree,
							 towers=1, pre_layers=1, post_layers=1, edge_dim=1,
							 divide_input=False)
		self.conv1_bn = BatchNorm(nhid // 2)

		self.conv1_sec = PNAConv(in_channels=nhid // 2, out_channels=nhid,
								 aggregators=aggregators, scalers=scalers, deg=degree,
								 towers=1, pre_layers=1, post_layers=1, edge_dim=1,
								 divide_input=False)
		self.conv1_sec_bn = BatchNorm(nhid)

		self.conv3 = GCNConv(32, 64)



		self.mlp = Sequential(Linear(64, 32), ReLU(), torch.nn.Dropout(0.5),
							  Linear(32, 1))


	def forward(self, graph_packet, neig_dict):


		x = graph_packet['ft_graph']
		edge_list = graph_packet['graph']
		weights = graph_packet['weights']

		selected_neig_features = []
		selected_neig_img_indices = []
		selected_neig_img_voxel_positions = []

		with torch.no_grad():
			for i in range(len(neig_dict)):

				selected_neig = neig_dict[i]

				neig_x = selected_neig['ft_graph']
				neig_weights = selected_neig['weights']
				neig_edge_list = selected_neig['graph']
				neig_node_voxels = selected_neig['node_voxel']
				index = selected_neig['index']

				x_neig = F.leaky_relu(self.conv1_bn(self.conv1(neig_x, neig_edge_list, neig_weights.reshape(-1,1))))
				x_neig = F.leaky_relu(self.conv1_sec_bn(self.conv1_sec(x_neig, neig_edge_list, neig_weights.reshape(-1,1))))


				if i == 0:
					selected_neig_features = x_neig[index,:]
					selected_neig_img_indices = i * torch.ones((index.shape[0],))
					selected_neig_img_voxel_positions = neig_node_voxels[index,:]
				else:
					selected_neig_features = torch.cat((selected_neig_features,x_neig[index,:]),0)
					selected_neig_img_indices = torch.cat((selected_neig_img_indices,i * torch.ones((index.shape[0],))),0)
					selected_neig_img_voxel_positions = torch.cat((selected_neig_img_voxel_positions,neig_node_voxels[index,:]),0)

		x_orig_node_count = x.shape[0]

		x_orig = F.leaky_relu(self.conv1_bn(self.conv1(x, edge_list, weights.reshape(-1,1))))
		x_orig = F.leaky_relu(self.conv1_sec_bn(self.conv1_sec(x_orig, edge_list, weights.reshape(-1, 1))))


		new_edges, selected_ids = get_graph_feature(x_orig, selected_neig_features, k=5)

		extended_features = torch.cat((x_orig,selected_neig_features),0)
		extended_edges = torch.cat((edge_list,new_edges),1)

		x3 = F.leaky_relu(self.conv3(extended_features, extended_edges))

		extended_features = x3[0:x_orig_node_count,:]

		x = torch.sigmoid(self.mlp(extended_features)).squeeze()

		selected_neig_img_indices = selected_neig_img_indices.cpu().numpy()
		selected_neig_img_voxel_positions = selected_neig_img_voxel_positions.cpu().numpy()

		return x, selected_neig_img_indices, selected_neig_img_voxel_positions, selected_ids.cpu().numpy()
