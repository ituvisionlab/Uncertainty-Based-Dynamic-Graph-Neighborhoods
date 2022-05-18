from PIL import Image
import numpy as np

data = np.load("Almenara\Pred_npys\ejemplo0013-aquije.npy")
img = Image.fromarray((data[0][120] * 255).astype('uint8') , 'L')
img.save('Almenara\Pred_npys\ejemplo0013-aquije.jpg')
img.show()