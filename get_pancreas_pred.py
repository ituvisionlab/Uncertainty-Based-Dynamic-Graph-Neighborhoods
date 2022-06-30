import pydicom
from pydicom import dcmread
import numpy as np

# Dicom in npy
#url = "/content/drive/MyDrive/Colab Notebooks/images-test/tomografias1/1/patient11-19.dcm"
#ds_1 = dcmread(url)
#arr_1 = ds_1.pixel_array
#print(arr_1)

# Test mask
#x = np.array([[1,2],[2,3],[3,4]])
#mask = np.array([[0,1],[0,1],[0,1]]).astype(bool)
#x[np.array(mask)]


# Real mask
# path=r"C:\Users\usuario\Desktop\Pancreas-CT\PANCREAS_00 01\11-24-2015-PANCREAS0001-Pancreas-18957\Pancreas-99667"
import matplotlib.pyplot as plt
import pydicom
from pydicom import dcmread
from pydicom.data import get_testdata_file
import os
npy_base_empty=np.empty([219,512,512])
i=0

# for filename in os.listdir(path):
#     f = os.path.join(path,filename)
#     ds = dcmread(f)
#     npy_base[i]=ds.pixel_array
#     i+=1
input_image = np.load("./our_pancreas_ct/pancreas_npy_3d/test/IMG-0013/IMG-0013.npy")
input_image = np.moveaxis(input_image, 0, -1)
input_image = np.moveaxis(input_image, 0, -1)

no_segmentado = input_image
segmentado = np.load("./Almenara/Pred_npys/ejemplo0013-aquije.npy")
print("Dimensiones segmentado", len(segmentado), len(segmentado[0]), len(segmentado[0][0]))
print("Dimensiones no_segmentado", len(no_segmentado), len(no_segmentado[0]), len(no_segmentado[0][0]))

npy_segment= np.array(segmentado[0])
npy_with_masking=np.multiply(input_image,npy_segment)
print(np.sum(npy_with_masking))

path_base=os.getcwd()
save_path=os.path.join(path_base,"./Almenara/Pred_npys/results")
os.mkdir(save_path)
it = 0

for shape in npy_with_masking:
    name = str(it) + ".npy"
    filename=os.path.join(save_path,name)
    np.save(filename, shape)
    it += 1
    print("File " + filename + " generated.")