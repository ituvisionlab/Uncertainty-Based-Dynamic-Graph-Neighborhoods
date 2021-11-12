import numpy as np
import torch
import time
import torch.nn.functional as F
from loader import DataReader
import torch.nn as nn
import os
import datetime
import torch.optim as optim
from unet import UNet as Net
import roi_list as roi
import gcn_pancreas.main as gcn
import gcn_pancreas.utilities.nparrays as arrtools
import cnn_utils as helper
from gcn_pancreas.gcn.models import GCN
import scipy.ndimage as ndimage
import gcn_pancreas.gcn.utils as gut
import gcn_pancreas.gcn.weighting as wfs
import gcn_pancreas.gcn.connectivity as cfs
from gcn_pancreas.gcn.utils import load_data, accuracy, reconstruct_from_n6, map_voxel_nodes, reconstruct_from_n6_fin
import gcn_pancreas.dlm.fcn_tools as tools
import gcn_pancreas.path_config as dirs
from gcn_pancreas.utilities.misc import npy_to_nifti
import gcn_pancreas.global_param as mpar
from torch_geometric.nn import fps

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def crete_graph_components(graph_packet, weight = 1):
	entropy_th = 0.8
	ref_shape = (graph_packet['expectation'] > 0.5).shape

	entropy = graph_packet['entropy']
	probability = graph_packet['expectation']
	roi = graph_packet['roi_limits']

	bin_entropy = (entropy > entropy_th).detach().cpu().numpy().astype(np.uint8)
	bin_prob = (probability > 0.5).detach().cpu().numpy().astype(np.uint8)
	kernel = np.ones(shape=(5, 7, 7), dtype=np.bool)
	dilated = ndimage.binary_dilation(bin_entropy, structure=kernel).astype(np.uint8)
	dilated = ((dilated + bin_prob) > 0).astype(np.int)

	expanded_bin_entropy = np.zeros(shape=ref_shape)
	expanded_bin_entropy[roi[0]:roi[3], roi[1]:roi[4], roi[2]:roi[5]] = bin_entropy[roi[0]:roi[3], roi[1]:roi[4], roi[2]:roi[5]] #TODO bin entropy with roi???


	graph_packet['dilated'] = dilated
	graph_packet['bin_entropy'] = bin_entropy
	graph_packet['expanded_bin_entropy'] = expanded_bin_entropy


	###
	reference = graph_packet['reference']
	vol = graph_packet['volume']
	seg_vol = graph_packet['segmentation']

	# ----- Normalizing volume
	num_vox = vol.shape[0] * vol.shape[1] * vol.shape[2]
	vmu = torch.sum(vol.float()) / num_vox
	vvar = torch.sum((vol.float() - vmu) ** 2) / num_vox

	fts = (vol - vmu) / vvar
	fts = torch.unsqueeze(fts, 3)

	fts = gut.add_feature_torch(probability, fts)
	fts = gut.add_feature_torch(entropy, fts)

	voxel_node, node_voxel = gut.map_voxel_nodes(vol.shape, dilated.astype(np.bool))

	ft_graph = gut.graph_fts(fts, node_voxel)  # convert the feature vol to a graph representation

	args = {
		"volume": (vol - vmu) / vvar,
		"prediction": seg_vol,
		"probability": probability,
		"uncertainty": bin_entropy,
		"entropy_map": entropy,
		"features": fts
	}

	graph, weights, lb, N = cfs.get_connect_func(1)(ref=seg_vol,
													node_voxel=node_voxel,
													weighting=wfs.get_weighting_func(weight),
													args=args)


	mask = gut.generate_mask(bin_entropy, node_voxel)  # Uncertainty mask
	# Volume ground truth are represented as nodes in the graph (reference graph)

	ref_lb = gut.reference_to_graph(reference, node_voxel)

	graph_packet['graph'] = graph.T
	graph_packet['weights'] = weights
	graph_packet['ft_graph'] = ft_graph
	graph_packet['lb'] = lb
	graph_packet['ref_lb'] = ref_lb
	graph_packet['mask'] = mask
	graph_packet['node_voxel'] = node_voxel

	return graph_packet


def make_prediction(divider, img, graph_packet):
	slice_count = img.shape[1]
	pred_tensor = torch.zeros((slice_count, 324, 324)).to(device)

	model.eval()

	#Slicing the 3D representation to batches in order to fit them into the GPU (implementation detail)
	for j in range(slice_count // divider):

		img_part = img[:, j * divider:(j + 1) * divider, :, :].to(device).permute((1, 0, 2, 3)).float()

		with torch.no_grad():
			output, _ = model(img_part)
			pred_tensor[j * divider:(j + 1) * divider, :, :] = output.squeeze()

	if slice_count % divider != 0:

		img_part = img[:, -(slice_count % divider):, :, :].to(device).permute((1, 0, 2, 3)).float()

		with torch.no_grad():
			output, _ = model(img_part)
			pred_tensor[-(slice_count % divider):, :, :] = output.squeeze()

	model.train()

	pred_tensor_expanded = helper.expand_to_size_torch(pred_tensor, img[0, :, :, :])
	seg_vol = torch.where(F.sigmoid(pred_tensor_expanded) > 0.5, 1, 0)
	seg_vol_np = seg_vol.detach().cpu().numpy()
	reduced_vol = arrtools.largest_connected_component3d(vol=seg_vol_np)
	roi_limits = arrtools.bounding_cube(reduced_vol)

	graph_packet['roi_limits'] = roi_limits
	graph_packet['segmentation'] = seg_vol

def forward_from_unet(divider, do_iter, img, img_class, train_flag):

	init_train_flag = train_flag

	slice_count = img.shape[1]

	pred_exp_tensor = torch.zeros((slice_count, 324, 324)).to(device)
	pred_ent_tensor = torch.zeros((slice_count, 324, 324)).to(device)

	vol_tensor = torch.zeros((slice_count, 512, 512)).to(device)
	ref_tensor = torch.zeros((slice_count, 512, 512)).to(device)

	selected_slice_to_tr = np.random.randint(divider)
	selected_bunch_to_tr = np.random.randint(slice_count // divider)


	for j in range(slice_count // divider):

		img_part = img[:, j * divider:(j + 1) * divider, :, :].to(device).permute((1, 0, 2, 3)).float()

		class_part = img_class[:, j * divider:(j + 1) * divider, :, :].to(device).permute((1, 0, 2, 3))
		expectation = torch.zeros((divider, 324, 324)).to(device)

		for k in range(do_iter):

			if not train_flag:
				with torch.no_grad():
					output, _ = model(img_part)

					expectation += (F.sigmoid(output).squeeze() / do_iter).detach()
			else:
				output, _ = model(img_part)

				if j == selected_bunch_to_tr:
					sel_slice = (F.sigmoid(output[selected_slice_to_tr,:,:]).squeeze() / do_iter)

					all_slices = (F.sigmoid(output).squeeze() / do_iter).detach()

					all_slices[selected_slice_to_tr,:,:] = sel_slice

					expectation += all_slices

				else:
					expectation += (F.sigmoid(output).squeeze() / do_iter).detach()


				train_flag = False

		train_flag = init_train_flag

		expectation = torch.clamp(expectation, min=0.0, max=1.0)
		entropy = -expectation * torch.log2(expectation + 1.0e-15) - (1.0 - expectation) * torch.log2(1.0 - expectation + 1.0e-15)
		pred_exp_tensor[j * divider:(j + 1) * divider, :, :] = expectation
		pred_ent_tensor[j * divider:(j + 1) * divider, :, :] = entropy
		vol_tensor[j * divider:(j + 1) * divider, :, :] = img_part.squeeze()
		ref_tensor[j * divider:(j + 1) * divider, :, :] = class_part.squeeze()

	# Remaining
	expectation_temp = torch.zeros((slice_count % divider, 324, 324)).to(device)

	if slice_count % divider != 0:

		img_part = img[:, -(slice_count % divider):, :, :].to(device).permute((1, 0, 2, 3)).float()
		class_part = img_class[:, -(slice_count % divider):, :, :].to(device).permute((1, 0, 2, 3)) 

		for k in range(do_iter):

			with torch.no_grad():
				output, _ = model(img_part)

				expectation_temp += (F.sigmoid(output).squeeze() / do_iter).detach()



		expectation_temp = torch.clamp(expectation_temp, min=0.0, max=1.0)
		entropy_temp = -expectation_temp * torch.log2(expectation_temp + 1.0e-15) - (1.0 - expectation_temp) * torch.log2(1.0 - expectation_temp + 1.0e-15)

		pred_exp_tensor[-(slice_count % divider):, :, :] = expectation_temp
		pred_ent_tensor[-(slice_count % divider):, :, :] = entropy_temp
		vol_tensor[-(slice_count % divider):, :, :] = img_part.squeeze()
		ref_tensor[-(slice_count % divider):, :, :] = class_part.squeeze()



	pred_exp_tensor_expanded = helper.expand_to_size_torch(pred_exp_tensor, img_class[0, :, :, :])
	pred_ent_tensor_expanded = helper.expand_to_size_torch(pred_ent_tensor, img_class[0, :, :, :])
	pred_ent_tensor_expanded[pred_ent_tensor_expanded < 0] = 0.0

	graph_packet = {}
	graph_packet['expectation'] = pred_exp_tensor_expanded
	graph_packet['entropy'] = pred_ent_tensor_expanded
	graph_packet['reference'] = ref_tensor
	graph_packet['volume'] = vol_tensor


	return graph_packet


class DiceLoss(nn.Module):
	def __init__(self, weight=None, size_average=True):
		super(DiceLoss, self).__init__()

	def forward(self, inputs, targets, smooth=0.001):
		inputs = torch.sigmoid(inputs)
		inputs = inputs.view(-1)
		targets = targets.reshape(-1)
		intersection = (inputs * targets).sum()
		dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

		return 1 - dice

def focal_loss(p, y, alpha=0.5, gamma=0.0):
	eps = 1.0e-15
	p0 = torch.ones_like(p) - p
	y0 = torch.ones_like(y) - y

	loss = -1.0*alpha
	loss *= y
	pw = (p0.pow(gamma))
	loss *= pw
	loss *= torch.log(p + eps)

	loss2 = -1.0*(1.0 - alpha)
	loss2 *= y0
	loss2 *= (p ** gamma)
	loss2 *= torch.log(p0 + eps)

	loss += loss2
	return torch.mean(loss)

def mean_vol_dsc(vol, gt_vol):
	val_acc = 0.0
	num_slices = vol.shape[0]
	for i in range(num_slices):
		mat = vol[i, :, :]
		gt_slice = gt_vol[i, :, :]
		acc = tools.dice_score_np(mat, gt_slice)
		val_acc += acc

	val_acc = val_acc / float(num_slices)
	return val_acc

def vol_dsc(vol, gt_vol):
	eps = 1e-9
	ab = np.sum(vol * gt_vol)
	a = np.sum(vol)
	b = np.sum(gt_vol)
	dsc = (2 * ab + eps) / (a + b + eps)
	return dsc


def graph_train(graph_packet, graph_model, neig_dict, optimizer, optimizer_main_model, idx_train, idx_val, valid=True, is_train_main=False):
	t = time.time()
	graph_model.train()
	optimizer.zero_grad()

	if is_train_main == True:
		optimizer_main_model.zero_grad()

	if is_train_main == False:
		graph_packet['ft_graph'] = graph_packet['ft_graph'].detach()
		graph_packet['weights'] = graph_packet['weights'].detach()
		graph_packet['graph'] = graph_packet['graph'].detach()


	output, _, _, _ = graph_model(graph_packet, neig_dict)

	loss_train = focal_loss(output[idx_train], graph_packet['ref_lb'][idx_train])
	acc_train = accuracy(output[idx_train], graph_packet['ref_lb'][idx_train])
	loss_train.backward()

	optimizer.step()

	if is_train_main == True:
		optimizer_main_model.step()


	voxel_node, node_voxel = map_voxel_nodes(graph_packet['volume'].shape, graph_packet['dilated'].astype(np.bool))
	graph_predictions = (output > mpar.gcn_th).cpu().numpy().astype(np.float32)
	refined = reconstruct_from_n6_fin(graph_predictions, node_voxel, graph_packet['volume'].shape)
	gcn_vol_dsc = vol_dsc(refined, graph_packet['reference'].detach().cpu().numpy())
	print("GCN - Train Vol Dsc : ", gcn_vol_dsc)


	if valid:
		with torch.no_grad():
			graph_model.eval()
			output, _, _, _ = graph_model(graph_packet, neig_dict)
			loss_val = focal_loss(output[idx_val], graph_packet['lb'][idx_val])
			acc_val = accuracy(output[idx_val], graph_packet['lb'][idx_val])
			print('loss_train: {:.4f}'.format(loss_train.item()),
				  'acc_train: {:.4f}'.format(acc_train.item()),
				  'loss_val: {:.4f}'.format(loss_val.item()),
				  'acc_val: {:.4f}'.format(acc_val.item()),
				  'time: {:.4f}s'.format(time.time() - t))



def gcn_inference(graph_packet, epoch, graph_model, neig_dict, idx_test, path=None, best_score=0.0):
	roi_limits = graph_packet['roi_limits']
	vol_shape_tuple = (roi_limits[3] - roi_limits[0], roi_limits[4] - roi_limits[1], roi_limits[5] - roi_limits[2])

	segmentation = graph_packet['segmentation']
	gt = graph_packet['reference']
	roi_vol = graph_packet['volume']
	valid_nodes = graph_packet['dilated']
	vol = graph_packet['reference'] 
	valid_nodes = graph_packet['dilated']

	gt[gt != 0] = 1

	graph_model.eval()
	output = graph_model(graph_packet, neig_dict)

	y_test = torch.from_numpy(graph_packet['mask']).to(segmentation.device)

	loss_test = focal_loss(output[idx_test], y_test[idx_test])
	acc_test = accuracy(output[idx_test], y_test[idx_test])

	print("GCN Inference Test Loss {}".format(loss_test))
	print("GCN Inference Test Acc {}".format(acc_test))


	voxel_node, node_voxel = map_voxel_nodes(roi_vol.shape, valid_nodes.astype(np.bool))
	graph_predictions = (output > mpar.gcn_th).cpu().numpy().astype(np.float32)
	refined = reconstruct_from_n6_fin(graph_predictions, node_voxel, roi_vol.shape)  # recovering the volume shape

	# recovering sizes
	segmentation_expanded = np.zeros(vol_shape_tuple, dtype=np.float) 
	segmentation_expanded = segmentation[roi_limits[0]:roi_limits[3], roi_limits[1]:roi_limits[4], roi_limits[2]:roi_limits[5]].detach().cpu().numpy()

	refined_expanded = np.zeros(vol_shape_tuple, dtype=np.float)
	refined_expanded = refined[roi_limits[0]:roi_limits[3], roi_limits[1]:roi_limits[4], roi_limits[2]:roi_limits[5]]

	
	cnn_slice_dsc = mean_vol_dsc(segmentation.detach().cpu().numpy(), gt.detach().cpu().numpy())
	gcn_slice_dsc = mean_vol_dsc(refined, gt.detach().cpu().numpy())

	cnn_vol_dsc = vol_dsc(segmentation.detach().cpu().numpy(), gt.detach().cpu().numpy())
	gcn_vol_dsc = vol_dsc(refined, gt.detach().cpu().numpy())

	np.save(path + dirs.GRAPH_PREDICTION, refined)


	if gcn_vol_dsc > best_score:
		npy_to_nifti(refined_expanded, path + "best_" + dirs.NIFTI_GRAPH_SEG)
		best_score = gcn_vol_dsc
		best_epoch = epoch

	npy_to_nifti(refined_expanded, path + dirs.NIFTI_GRAPH_SEG)


	info = {
		"cnn_slice_dsc": cnn_slice_dsc,
		"gcn_slice_dsc": gcn_slice_dsc,
		"cnn_vol_dsc": cnn_vol_dsc,
		"gcn_vol_dsc": gcn_vol_dsc,
		"loss_test": loss_test,
		"best_score": best_score,
		"best_epoch": best_epoch
	}

	return info



BATCH_SIZE = 1
numworkers = 1

train_loader = DataReader(mode='train', score=False, roi_list=roi.ROI_TRAIN)

in_channel = 1
out_channel = 1
num_epochs = 40

gcn_epochs = 5

tr_score_list = []
val_score_list = []

model = Net(in_channel, out_channel, dropout_rate=0.3)
state = torch.load("checkpoint.pancreas.dsc.pth.tar", map_location=device)
model.load_state_dict(state["state_dict"])
model = model.to(device)


#Average degree for panc.
degree = np.zeros((20,1))
degree[3:7] = np.array([3589.4186046512,	7966.0930232558,	15908.8604651163,	180809.302325581]).reshape((4,1))
degree = torch.from_numpy(degree).to(device)

graph_model = GCN(nfeat=3, nhid=32, degree = degree).to(device)


weight_decay = 1e-5


optimizer2 = optim.Adam(graph_model.parameters(), lr=1e-2, weight_decay = weight_decay)
optimizer3 = optim.Adam(model.parameters(), lr=1e-6)

loss = DiceLoss().to(device)

all_tr_losses = torch.zeros(num_epochs, 1)
all_val_losses = torch.zeros(num_epochs, 1)



divider = 6
do_iter = 20  # number of iteration for dropout


for epoch_id in range(num_epochs):

	model.train()

	train_loader.shuffle_indexes()

	total_loss = 0
	total_true = 0
	total_false = 0
	total_score = 0
	score_to_print = 0.0

	for i, data in enumerate(train_loader):


		print("train - epoch = ", epoch_id, "iter = ", i, end="\r")

		img = data['image'].unsqueeze(0)
		img_class = data['label'].unsqueeze(0)
		pan_orig = data['pname']

		graph_packet = forward_from_unet(divider, do_iter, img, img_class, True)

		make_prediction(divider, img, graph_packet)
		graph_packet = crete_graph_components(graph_packet, weight=1)

		sel_neig_image_indices = np.arange(0,len(train_loader))
		sel_neig_image_indices = np.delete(sel_neig_image_indices, np.argwhere(sel_neig_image_indices == i))
		sel_neig_image_indices = np.random.permutation(sel_neig_image_indices)[0:5]

		all_neig_elements = []


		with torch.no_grad():
			for j in range(len(sel_neig_image_indices)):

				sel_index = sel_neig_image_indices[j]

				img = train_loader[sel_index]['image'].unsqueeze(0)
				img_class = train_loader[sel_index]['label'].unsqueeze(0)
				pan = train_loader[sel_index]['pname']

				neig_gp = forward_from_unet(divider, do_iter, img, img_class, False)
				make_prediction(divider, img, neig_gp)
				neig_gp = crete_graph_components(neig_gp, weight=1)

				neig_dict = {}
				neig_dict['name'] = pan
				neig_dict['ft_graph'] = neig_gp['ft_graph'].detach()
				neig_dict['weights'] = neig_gp['weights'].detach()
				neig_dict['node_voxel'] = torch.from_numpy(neig_gp['node_voxel']).to(device).float().detach()
				neig_dict['graph'] = torch.from_numpy(neig_gp['graph']).to(device).detach()
				neig_dict["index"] = fps(neig_dict["node_voxel"], ratio=1.0 / 40).detach()

				all_neig_elements.append(neig_dict)


		graph_packet['graph'] = torch.from_numpy(graph_packet['graph']).to(device)
		graph_packet['lb'] = graph_packet['lb'].to(device).long()
		graph_packet['ref_lb'] = torch.from_numpy(graph_packet['ref_lb']).to(device)


		val_portion = 0.2
		full_mask = 1 - graph_packet['mask']


		working_nodes = np.where(full_mask != 0)[0]  
		random_arr = np.random.uniform(low=0, high=1, size=working_nodes.shape)

		idx_train = working_nodes[random_arr > val_portion] 
		idx_val = working_nodes[random_arr <= val_portion]  
		idx_test = np.where(graph_packet['mask'] != 0) 

		idx_train = torch.LongTensor(idx_train).to(device).long()
		idx_val = torch.LongTensor(idx_val).to(device).long()
		idx_test = torch.LongTensor(idx_test[0]).to(device).long()

		t_total = time.time()

		graph_packet['graph'] = graph_packet['graph'].to(device)

		print("------- Training GCN")
	
		for epoch in range(gcn_epochs):
			valid = True

			if epoch == 0:
				graph_train(graph_packet, graph_model, all_neig_elements, optimizer2, optimizer3, idx_train, idx_val, valid, True)
			else:
				graph_train(graph_packet, graph_model, all_neig_elements, optimizer2, optimizer3, idx_train, idx_val, valid, False)


			count += 1
		model_save_dir = "cnn_gcn_models_exp0"

		try:
			os.mkdir(model_save_dir)
		except:
			pass

		torch.save(model, model_save_dir + '/model_' + str(epoch_id) + "_" + pan_orig + '.pth')
		torch.save(graph_model.state_dict(), model_save_dir + '/graph_model_' + str(epoch_id) + "_" + pan_orig + '.pth')