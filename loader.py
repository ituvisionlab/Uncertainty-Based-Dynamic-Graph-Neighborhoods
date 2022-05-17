import torch
import glob, os
import torch.utils.data
from PIL import Image
from torchvision import transforms
import numpy as np
import random
import re
import roi_list as roi
import matplotlib.pyplot as plt

def atoi(text):
	return int(text) if text.isdigit() else text

def natural_keys(text):

	return [ atoi(c) for c in re.split(r'(\d+)', text) ]

class DataReader(torch.utils.data.Dataset):
	def __init__(self, mode, score, roi_list, pancreas_name="",exp_name = ""):
		super(DataReader, self).__init__()

		self.input_img_paths = []
		self.mask_paths = []
		self.mode = mode
		self.pname = []
		self.score = score
		self.roi_list = roi_list


		pancreas_dir = "our_pancreas_ct/pancreas_npy_3d"
		label_dir = "pancreas_ct/pancreas_label_npy_3d"


		if mode == 'train':

			train_input_dir = pancreas_dir + "/train"
			train_label_dir = label_dir + "/train"
			if pancreas_name != "":
				train_input_folder = []
				train_input_folder.append(pancreas_name)


				train_label_folder = []
				train_label_folder.append(pancreas_name)

			else:
				train_input_folder = os.listdir(train_input_dir)
				train_input_folder.sort(key=natural_keys)
				self.pname = train_input_folder


				train_label_folder = os.listdir(train_label_dir)
				train_label_folder.sort(key=natural_keys)


			for i in range(len(train_input_folder)):
				if pancreas_name != "":
					temp_in_path = train_input_dir + "/" + pancreas_name
					temp_label_path = train_label_dir + "/" + pancreas_name
				else:
					temp_in_path = train_input_dir + "/" + train_input_folder[i]
					temp_label_path = train_label_dir + "/" + train_label_folder[i]


				in_img_list = os.listdir(temp_in_path)

				label_img_list = os.listdir(temp_label_path)

				in_path = temp_in_path + "/" + in_img_list[0]
				label_path = temp_label_path + "/" + label_img_list[0]
				self.input_img_paths.append(in_path)
				self.mask_paths.append(label_path)

		else:
			val_input_dir = pancreas_dir + "/" + self.mode
			val_label_dir = label_dir + "/" + self.mode
			if pancreas_name != "":
				val_input_folder = []
				val_input_folder.append(pancreas_name)

				val_label_folder = []
				val_label_folder.append(pancreas_name)

			else:
				val_input_folder = os.listdir(val_input_dir)
				print(val_input_dir)
				val_input_folder.sort(key=natural_keys)
				self.pname = val_input_folder

				# val_label_folder = os.listdir(val_label_dir)
				# val_label_folder.sort(key=natural_keys)

			for i in range(len(val_input_folder)):
				if pancreas_name != "":
					temp_in_path = val_input_dir + "/" + pancreas_name
					temp_label_path = val_label_dir + "/" + pancreas_name
				else:
					temp_in_path = val_input_dir + "/" + val_input_folder[i]
					# temp_label_path = val_label_dir + "/" + val_label_folder[i]


				in_img_list = os.listdir(temp_in_path)

				# label_img_list = os.listdir(temp_label_path)

				in_path = temp_in_path + "/" + in_img_list[0]
				#  label_path = temp_label_path + "/" + label_img_list[0]
				self.input_img_paths.append(in_path)
				# self.mask_paths.append(label_path)


		self.indexes = np.arange(0,len(self.input_img_paths))

	def shuffle_indexes(self):

		self.indexes = np.random.permutation(self.indexes)

	def load_input_img(self, filepath):
		img = Image.open(filepath).convert('RGB')
		return img

	def __getitem__(self, index):
		pancreas_dir = "our_pancreas_ct/pancreas_npy_3d"
		# label_dir = "pancreas_ct/pancreas_label_npy_3d"
		if self.mode == 'train':

			vol = np.load(pancreas_dir + "/" + self.mode + "/" + self.pname[self.indexes[index]] + "/" + self.pname[self.indexes[index]] + ".npy")
			ref = np.load(label_dir + "/" + self.mode + "/" + self.pname[self.indexes[index]] + "/" + self.pname[self.indexes[index]] + ".npy")


			in_arr = vol[:,:,self.roi_list[self.pname[self.indexes[index]]][0]:self.roi_list[self.pname[self.indexes[index]]][1]]
			msk_arr = ref[:, :, self.roi_list[self.pname[self.indexes[index]]][0]:self.roi_list[self.pname[self.indexes[index]]][1]].astype(np.uint8)

			input_img = torch.from_numpy(in_arr)
			msk = torch.from_numpy(msk_arr)

			input_img = input_img.permute((2,0,1))
			msk = msk.permute((2, 0, 1))


			x = random.uniform(1, 1.1)
			y = random.uniform(1, 1.1)
			_, w, h = input_img.shape
			new_w = int(x*w)
			new_h = int(y*h)

			input_resize = transforms.Resize(size=(new_h, new_w), interpolation= Image.BILINEAR)
			mask_resize = transforms.Resize(size=(new_h, new_w), interpolation= Image.NEAREST)
			image = input_resize(input_img)
			mask = mask_resize(msk)

			i, j, he, wi = transforms.RandomCrop.get_params(image, output_size=(512, 512))
			image = transforms.functional.crop(image, i, j, he, wi)
			mask = transforms.functional.crop(mask, i, j, he, wi)


			sp, ep = transforms.RandomPerspective.get_params(width = 512, height = 512, distortion_scale = 0.1)

			image = transforms.functional.perspective(image, sp, ep, interpolation=Image.BILINEAR)
			mask = transforms.functional.perspective(mask, sp, ep, interpolation=Image.NEAREST)

			data = {}
			data['image'] = image
			data['label'] = mask
			data['pname'] = self.pname[self.indexes[index]]

			return data

		else:			
			in_arr = np.load(pancreas_dir + "/" + self.mode + "/" + self.pname[self.indexes[index]] + "/" + self.pname[self.indexes[index]] + ".npy")
			# msk_arr = np.load(label_dir + "/" + self.mode + "/" + self.pname[self.indexes[index]] + "/" + self.pname[self.indexes[index]] + ".npy").astype(np.uint8)

			input_img = torch.from_numpy(in_arr)
			# msk = torch.from_numpy(msk_arr)

			input_img = input_img.permute((2,0,1))
			# msk = msk.permute((2, 0, 1))

			data = {}
			data['image'] = input_img
			# data['label'] = msk
			data['pname'] = self.pname[self.indexes[index]]

			return data
	def __len__(self):
		return len(self.input_img_paths)