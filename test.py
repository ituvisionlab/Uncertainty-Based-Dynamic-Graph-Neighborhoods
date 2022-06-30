import numpy as np
import torch
import torch.nn.functional as F
from loader import DataReader
import torch.nn as nn
import os
import torch.optim as optim
import gcn_pancreas.utilities.nparrays as arrtools
import cnn_utils as helper

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def vol_dsc(vol, gt_vol):
	eps = 1e-9
	ab = np.sum(vol * gt_vol)
	a = np.sum(vol)
	b = np.sum(gt_vol)
	dsc = (2 * ab + eps) / (a + b + eps)
	return dsc

def make_prediction(divider, img):
	print(img.shape)
	slice_count = img.shape[1]
	pred_tensor = torch.zeros((slice_count, 324, 324)).to(device)

	model.eval()

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




	pred_tensor_expanded = helper.expand_to_size_torch(pred_tensor, img[0, :, :, :])

	seg_vol = torch.where(F.sigmoid(pred_tensor_expanded) > 0.5, 1, 0)

	seg_vol_np = seg_vol.detach().cpu().numpy()

	orig_pred = np.zeros((1,slice_count, 512, 512))
	reduced_vol = arrtools.largest_connected_component3d(vol=seg_vol_np)
	roi_limits = arrtools.bounding_cube(reduced_vol)

	orig_pred[0,roi_limits[0]:roi_limits[3], roi_limits[1]:roi_limits[4], roi_limits[2]:roi_limits[5]] = reduced_vol[roi_limits[0]:roi_limits[3], roi_limits[1]:roi_limits[4], roi_limits[2]:roi_limits[5]]
	
	return orig_pred

def get_dsc(divider, img, ref):
	orig_pred = make_prediction(divider, img)
	return vol_dsc(orig_pred, ref.detach().cpu().numpy())

BATCH_SIZE = 1
numworkers = 1

test_loader = DataReader(mode='test', score=False, roi_list="", exp_name="")

num_epochs = 1
divider = 6
scores = []

mp = "pancreas_ct\checkpoint.pth"
print("Flag1")
# print(checkpoint.pth)
model = torch.load(mp, map_location=device)
print("Flag2")
model = model.to(device)
scores = []
print("Flag3")
with torch.no_grad():
	for i, data in enumerate(test_loader):
		# print("Flag: ", i)
		# print("Data", data)
		# exit(0)

		img = data['image'].unsqueeze(0)
		# img_class = data['label'].unsqueeze(0)
		pan_orig = data['pname']

		print(pan_orig)
		score = make_prediction(divider, img)
		print(score.max())
		print("Shape: ", score.shape)
		scores.append(score)

		# print("DSC Score: {}".format(score))

	scores = np.array(scores)
	np.save("Almenara\Pred_npys\ejemplo0013-aquije", scores[0])
	# print("Mean {} Std {}".format(np.mean(scores),np.std(scores)))