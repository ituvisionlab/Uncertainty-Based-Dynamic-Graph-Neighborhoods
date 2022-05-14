import dicom2nifti
import nibabel as nib
import numpy as np
import glob, os
import sys
from PIL import Image
import torch
import random
import re

def atoi(text):
	return int(text) if text.isdigit() else text

def natural_keys(text):

	return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def assemble_vol(vol):
	z = len(vol)
	y, x = vol[0].shape
	np_vol = np.empty(shape=(y, x, z), dtype=float)
	for i in range(z):
		np_vol[:, :, i] = vol[i]
	return np_vol
#v = np.load("/home/ufuk/Desktop/graph_based_cn/gcn_refinement/vol.0012.nii.gz.npy", allow_pickle=True)
#r = np.load("/home/ufuk/Desktop/graph_based_cn/gcn_refinement/ref.0012.nii.gz.npy", allow_pickle=True)


pancreas_dir = "./pancreas_train_with_graph/pancreas_ct/Pancreas-CT/"
label_dir = "./pancreas_train_with_graph/pancreas_ct/TCIA_pancreas_labels-02-05-2017/"
temp_dir = []
data_dir = []

pancreas_list = os.listdir(pancreas_dir) #"/home/ufuk/Desktop/pancreas_ct/Pancreas-CT/Pancreas1"
pancreas_list = sorted(pancreas_list)

label_list = os.listdir(label_dir) #/home/ufuk/Desktop/pancreas_ct/TCIA_pancreas_labels-02-05-2017/label0001.nii.gz
label_list = sorted(label_list)


for fldr in pancreas_list:
	temp_dir.append(pancreas_dir + fldr + "/")

for dirr in temp_dir:
	temp_dir2 = os.listdir(dirr)#AAA
	folders_2 = os.listdir(dirr + temp_dir2[0] + "/")#AAA
	print(temp_dir2)
	print(folders_2)
	exit(0)
	data_dir.append(dirr + temp_dir2[0] + "/" + folders_2[0])#BBB


pancreas_nifti = "./pancreas_train_with_graph/pancreas_ct/pancreas_nifti/"
pancreas_npy = "./pancreas_train_with_graph_0106/pancreas_ct/pancreas_npy_3d/"
pancreas_label_npy = "./pancreas_train_with_graph_0106/pancreas_ct/pancreas_label_npy_3d/"


#
# '''
#
# for i in range (len(data_dir)):
#
# 	directory = pancreas_list[i]
# 	path = os.path.join(pancreas_nifti, pancreas_list[i])
# 	os.mkdir(path)
#
# 	new_dir = pancreas_nifti + directory
#
# 	image = dicom2nifti.convert_directory(data_dir[i], new_dir, compression=True, reorient=True)
#
# '''
#
#

pancreas_nifti_list = os.listdir(pancreas_nifti)
#pancreas_nifti_list = sorted(pancreas_nifti_list)
pancreas_nifti_list.sort(key=natural_keys)


print("input-start")
for i in range(len(pancreas_nifti_list)):
	nifti_example_dir = pancreas_nifti + pancreas_nifti_list[i] + "/none_pancreas.nii.gz"
	nifti = nib.load(nifti_example_dir)
	img = (np.array(nifti.get_data())*1.0).astype(np.float32)

	directory = pancreas_list[i]
	path = os.path.join(pancreas_npy, pancreas_nifti_list[i])
	os.mkdir(path)

	temp_list = []

	for j in range(img.shape[2]-1,-1,-1):
		temp = img[:,:,j]
		old_min = img.min()
		old_max = img.max()
		new_min = 0
		new_max = 255

		old_range = (old_max - old_min)
		if old_range == 0:
			old_range = 1
		new_range = (new_max - new_min)
		new_value = np.float32((((temp - old_min) * new_range) / old_range) + new_min)


		new_value = np.round(new_value)
		temp_list.append(new_value)


		#png_path = path + "/" + pancreas_nifti_list[i] + "_" + str(j) + ".png"
		#npy_path = path + "/" + pancreas_nifti_list[i] + "_" + str(j) + ".npy"
		#print(npy_path)
		#np.save(npy_path, new_value)
		#im = np.save(npy_path, new_value)
		#im = Image.fromarray(new_value)
		#im.save(png_path)

	temp_vol = assemble_vol(temp_list)
	npy_path = path + "/" + pancreas_nifti_list[i] + ".npy"
	np.save(npy_path, temp_vol)

print("input-end")




pancreas_label_nifti_list = os.listdir(label_dir)
#pancreas_label_nifti_list = sorted(pancreas_label_nifti_list)
pancreas_label_nifti_list.sort(key=natural_keys)
#
# print("label-start")
# for i in range(len(pancreas_label_nifti_list)):
# 	if i < 9 :
# 		pan_name = "PANCREAS_000" + str(i+1)
# 	else:
# 		pan_name = "PANCREAS_00" + str(i+1)
# 	if (i == 24):
# 		continue
# 	if (i == 69):
# 		continue
# 	nifti_example_dir = label_dir + pancreas_label_nifti_list[i]
# 	nifti = nib.load(nifti_example_dir)
# 	img = np.array(nifti.dataobj)
# 	#directory = pancreas_list[i]
# 	#path = os.path.join(pancreas_label_npy, pancreas_list[i])
# 	path = os.path.join(pancreas_label_npy, pan_name)
# 	os.mkdir(path)
# 	print(str(i+1) + "/" + str(len(pancreas_label_nifti_list)))
# 	for j in range(img.shape[2]):
# 		temp = img[:,:,img.shape[2] - j - 1]
# 		#png_path = path + "/" + pancreas_label_nifti_list[i] + "_" + str(j) + ".png"
# 		npy_path = path + "/" + pancreas_label_nifti_list[i] + "_" + str(j) + ".npy"
# 		np.save(npy_path, temp)
# 		#im = Image.fromarray(temp)
# 		#im.save(png_path)
#
# print("label-end")


#### 3D ###


# print("label-start")
# for i in range(len(pancreas_label_nifti_list)):
# 	if i < 9 :
# 		pan_name = "PANCREAS_000" + str(i+1)
# 	else:
# 		pan_name = "PANCREAS_00" + str(i+1)
# 	if (i == 24):
# 		continue
# 	if (i == 69):
# 		continue
# 	nifti_example_dir = label_dir + pancreas_label_nifti_list[i]
# 	nifti = nib.load(nifti_example_dir)
# 	img = np.array(nifti.dataobj)
# 	path = os.path.join(pancreas_label_npy, pan_name)
# 	os.mkdir(path)
# 	print(str(i+1) + "/" + str(len(pancreas_label_nifti_list)))
# 	npy_path = path + "/" + pancreas_label_nifti_list[i] + ".npy"
# 	np.save(npy_path, img)
#
#
# print("label-end")