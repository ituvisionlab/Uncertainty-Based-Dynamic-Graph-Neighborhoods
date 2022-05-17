from PIL import Image
import numpy as np

data = np.load("ejemplo0013-aquije.npy")
print(data.shape)
# print(data[0][0].shape)
x = 0
d = 0
for i in data[0][110]:
    x+=1
    d = 0
    for j in i:
        d+=1
        if(j!=0):
            print(j)
            print(x, d)
            print("yes")
            exit(0)
# img = Image.fromarray(data[0][110], mode="L")
# img.save('ejemplo0013-aquije.png')
# img.show()