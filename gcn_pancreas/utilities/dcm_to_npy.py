import dicom2nifti
import nibabel as nib
import numpy as np
import glob, os
import sys
from PIL import Image


pancreas_dir = "/mnt/disk3/gcn_ref/pancreas/Pancreas-CT/"
label_dir = "/mnt/disk3/gcn_ref/pancreas/TCIA_pancreas_labels-02-05-2017/"
pancreas_nifti = "/mnt/disk3/gcn_ref/pancreas/pancreas_nifti/"
pancreas_npy = "/mnt/disk3/gcn_ref/pancreas/pancreas_npy/"
pancreas_label_npy = "/mnt/disk3/gcn_ref/pancreas/pancreas_label_npy/"

#"/mnt/disk3/gcn_ref/pancreas/pancreas_label_npy/ref.PANCREAS_0042.nii.gz.npy"
#"cc

os.mkdir(pancreas_nifti)
os.mkdir(pancreas_npy)
os.mkdir(pancreas_label_npy)

temp_dir = []
data_dir = []
dcm_pancreas_dir = []


pancreas_name = ["PANCREAS_0042",
                "PANCREAS_0058",
                "PANCREAS_0056",
                "PANCREAS_0060",
                "PANCREAS_0062",
                "PANCREAS_0032",
                "PANCREAS_0051",
                "PANCREAS_0035",
                "PANCREAS_0031",
                "PANCREAS_0071",
                "PANCREAS_0076",
                "PANCREAS_0059",
                "PANCREAS_0065",
                "PANCREAS_0012",
                "PANCREAS_0057",
                "PANCREAS_0029",
                "PANCREAS_0027",
                "PANCREAS_0080",
                "PANCREAS_0018",
                "PANCREAS_0068"]




pancreas_list = [pancreas_dir + x + "/" for x in pancreas_name]


for path in pancreas_list:
    foo_dir1 = os.listdir(path)
    foo_dir2 = os.listdir(path + foo_dir1[0] + "/")
    dcm_pancreas_dir.append(path + foo_dir1[0] + "/" + foo_dir2[0])


for i in range (len(pancreas_list)):

	directory = pancreas_list[i]
	path = os.path.join(pancreas_nifti, pancreas_name[i])
	os.mkdir(path)

	image = dicom2nifti.convert_directory(dcm_pancreas_dir[i], path, compression=True, reorient=True)


pancreas_nifti_list = os.listdir(pancreas_nifti)

print("input-start")
for i in range(len(pancreas_nifti_list)):
	nifti_example_dir = pancreas_nifti + pancreas_nifti_list[i] + "/none_pancreas.nii.gz"
	nifti = nib.load(nifti_example_dir)
	img = (np.array(nifti.get_data())*1.0).astype(np.float32)

	directory = pancreas_list[i]

	old_min = img.min()
	old_max = img.max()

	new_min = 0
	new_max = 255

	old_range = (old_max - old_min)
	if old_range == 0:
		old_range = 1
	new_range = (new_max - new_min)
	new_value = np.float32((((img - old_min) * new_range) / old_range) + new_min)


	npy_path = pancreas_npy + "vol." + pancreas_nifti_list[i] + ".nii.gz.npy"
	new_value = np.round(new_value)
	np.save(npy_path, new_value)

print("input-end")



label_list = os.listdir(label_dir)

print("label-start")
for pname in pancreas_name:
	for label_name in label_list:
		if pname[11:] in label_name:
			path = label_dir + label_name
			nifti = nib.load(path)
			img = np.array(nifti.dataobj)
			npy_path = pancreas_label_npy + "ref." + pname + ".nii.gz.npy"
			np.save(npy_path, img)

print("label-end")