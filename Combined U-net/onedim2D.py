import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
from skimage.measure import label
#from skimage import data, util,measure
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data
from matplotlib import cm
from skimage import morphology



results_dir = 'new_result/'


data = np.load(results_dir+'2D1dim.npz')

data = data

x_all=data['x'][0]
y_all=data['y'][0]
pred_all=data['pred'][0]

x=x_all[80]
y=y_all[80]
pred=pred_all[80]


y = y.reshape(1, 128, 128, 2)
pred = pred.reshape(1, 128, 128, 2)

y= np.argmax(y, axis=3)
pred = np.argmax(pred, axis=3)
print(x[0,:,:,0].shape)
print(y[0].shape)
print(pred[0].shape)

pred=pred[0]

label_image = label(pred, connectivity=2)
pred_rev = morphology.remove_small_objects(label_image, min_size=100, connectivity=2)
pred_rev = pred_rev.reshape(y.size)
for i in range(len(pred_rev)):
    if pred_rev[i] > 0:
        pred_rev[i] = 1
pred = pred_rev.reshape(pred.shape)


##   2D mask and net work
image =x[0,:,:,0]
pred_mask = pred
true_mask = y[0]

#mask = true_mask
mask = pred_mask
cart_cor=np.array(np.where(mask>0)).T
for i in cart_cor:
    image[i[0], i[1]] =3000

np.save(results_dir + 'one.npy',image)
#plt.imshow(image[::-1,:])
#plt.savefig(result_dir + "2d1.png", bbox_inches='tight')
#plt.show()
