import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import shutil
import re

def vol_dsc_torch(vol, gt_vol):
	eps = 1e-9
	ab = torch.sum(vol * gt_vol)
	a = torch.sum(vol)
	b = torch.sum(gt_vol)
	dsc = (2 * ab + eps) / (a + b + eps)
	return dsc.item()

def show_cam_on_image(img, mask):
	heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
	cam = heatmap*0.3 + np.float32(img.reshape((512,512,1)))*0.7
	cam = cam/np.max(cam)

	cv2.imshow("test window", np.uint8(255*cam))
	cv2.waitKey()


def expand_to_size_torch(x, y):

	expanded = torch.zeros(y.shape, dtype=torch.float32).to(x.device)
	x_off = (y.shape[2] - x.shape[2]) // 2
	y_off = (y.shape[1] - x.shape[1]) // 2
	xs = x.shape[2]
	ys = x.shape[1]
	expanded[:,y_off:y_off + ys, x_off:x_off + xs] = x
	return expanded


def save_pred(experiment_name, output, img_class, epoch_id, index, mode):
	# output = torch.sigmoid(output)
	p = output.cpu()
	pred_image = p.detach().numpy()
	# print(pred_image.shape)
	g = img_class.cpu()
	gt_image = g.detach().numpy()

	pred_concat = np.concatenate((pred_image[0, 0], pred_image[1, 0], pred_image[2, 0], pred_image[3, 0]), 1)

	gtrt_concat = np.concatenate((gt_image[0, 0], gt_image[1, 0], gt_image[2, 0], gt_image[3, 0]), 1)

	concat = np.concatenate((pred_concat, gtrt_concat), 0)
	concat = np.where(concat > 0.5, 1, 0)

	fig, ((ax1), (ax3)) = plt.subplots(2, 1, constrained_layout=True)
	fpath = experiment_name + '/' + str(mode) + '_results/' + str(epoch_id) + '/test_' + str(epoch_id) + '-' + str(index) + '.png'

	plt.imsave(fpath, concat, cmap='gray')
	plt.clf()
	plt.close()

'''
def save_config(res_name, model_name, lr, score_func, score_eps, loss_func, loss_eps, optimizer_name, num_epochs):
	with open(res_name, "a") as f:
		f.write("Model Name: ")
		f.write(model_name)
		f.write("\n")

		f.write("Learning Rate: ")
		f.write(lr)
		f.write("\n")

		f.write("Score Function: ")
		f.write(score_func)
		f.write("\n")

		f.write("Epsilon of Score Function: ")
		f.write(score_eps)
		f.write("\n")

		f.write("Loss Function: ")
		f.write(loss_func)
		f.write("\n")

		f.write("Epsilon of Loss Function: ")
		f.write(loss_eps)
		f.write("\n")

		f.write("Optimizer Name: ")
		f.write(optimizer_name)
		f.write("\n")

		f.write("Number of Epoch: ")
		f.write(num_epochs)
		f.write("\n")
		f.write("--------------------------------------")
		f.write("\n")

def save_res(epoch_id, total_loss, loader_len, total_score, time_start, res_name, mode):
	with open(res_name, "a") as f:
		f.write(mode)
		f.write(": ")
		f.write(str(datetime.datetime.now()))
		f.write("\n")

		f.write("Epoch ")
		# f.write(str(i))
		f.write(" scores: ")
		f.write(str(epoch_id))
		f.write("\n")

		f.write("Loss: ")
		f.write(str((total_loss / loader_len)))
		f.write("\n")

		f.write("Score: ")
		f.write(str(total_score))
		f.write("\n")
		# f.write("Score: ")
		# f.write(str((total_score / loader_len)))
		# f.write("\n")

		f.write("Time (s): ")
		f.write(str(time.time() - time_start))
		f.write("\n")
		f.write("--------------------------------------")
		f.write("\n")
'''

def assemble_vol(vol):
	z = len(vol)
	y, x = vol[0].shape
	np_vol = np.empty(shape=(y, x, z), dtype=float)
	for i in range(z):
		np_vol[:, :, i] = vol[i]
	return np_vol

# region Dice Score training
def crop_tensor_to_size_reference(x1, x2):
	"""A center-based crop of x1 (``shaped [batch, c, rows1, cols1]``) to the x2's (shaped ``[batch, c, rows2, cols2]``)
	size. It is assumed that rows1/cos1 >= rows2/cos2.

	Parameters
	----------
	x1 : torch.Tensor
		The tensor that will be cropped.
	x2 : torch.Tensor
		The reference tensor. The batch size and number of channels must be the same as `x1`.

	Returns
	-------
	torch.Tensor
		A tensor with the content of x1 cropped to the rows and cols of x2.
	"""
	x_off = (x1.size()[3] - x2.size()[3]) // 2
	y_off = (x1.size()[2] - x2.size()[2]) // 2
	xs = x2.size()[3]
	ys = x2.size()[2]
	x = x1[:, :, y_off:y_off + ys, x_off:x_off + xs]
	return x

def prepare_experiment(project_path = ".", experiment_name = None):
	next_exp_num = 0

	for directory in os.listdir(project_path):
		res = re.search("experiment_(.*)", directory)

		if res and next_exp_num < int(res[1]) + 1:
			next_exp_num = int(res[1]) + 1

	if experiment_name is None:
		experiment_name = "experiment_{}".format(next_exp_num)

	os.mkdir(experiment_name)
	os.mkdir(experiment_name + '/val_results/')
	os.mkdir(experiment_name + '/train_results/')
	os.mkdir(experiment_name + '/graphs/')
	os.mkdir(experiment_name + '/models/')
	os.mkdir(experiment_name + '/code/')

	all_python_files = os.listdir('.')

	for i in range(len(all_python_files)):
		if '.py' in all_python_files[i]:
			os.system('cp ' + all_python_files[i] + ' ' + experiment_name + '/code/')

	return  experiment_name