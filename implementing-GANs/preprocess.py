import numpy as np
from PIL import Image

images = []
for i in range(10000):
    file_name = "images/img_%05d.png" % (i)
    img = Image.open(file_name)
    images.append(np.reshape(np.array(img.getdata()), (128,128,3)))

data = np.array(images)
print(data.shape)
np.save("data.npy", data)
