import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.utils import plot_model
from keras import callbacks as cbks
from data_prep import *
from unet_model import *
from dataloader import *
from keras.callbacks import TensorBoard

results_dir = 'new_result/'
try:
    os.mkdir(results_dir)
except:
    pass



batch_size = 5
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
n_labels = train_data[0][1].shape[-1]

print(n_labels)
print(len(train_loader))

model = create_unet_model3D(input_image_size=train_data[0][0].shape, output_image_size=train_data[0][1].shape[0], n_labels=n_labels,layers=4, lowest_resolution=16,
                            mode='classification',init_lr=0.0001)




model.summary()






model.load_weights(results_dir+'segmentation-weights_3D3.h5')
x_all=[]
y_all=[]
pred_all=[]
ROI_points = []
sca = []
mas = []



for i in range(10):
    real_x = val_data[i][0][np.newaxis,...]
    real_y = val_data[i][1][np.newaxis,...]
    ROI_points.append(ROI_test[i])
    sca.append(scan_train[i])
    mas.append(mask_train[i])
    print(real_x.shape)
    print(real_y.shape)
    real_y_pred = model.predict(real_x)
    print(real_y_pred.shape)
    acc = model.evaluate(real_x,real_y)
    print(acc)
    x_all.append(real_x)
    y_all.append(real_y)
    pred_all.append(real_y_pred)

np.savez_compressed(results_dir+'3D3.npz',x=x_all, y=y_all, pred = pred_all,ROI = ROI_points)
