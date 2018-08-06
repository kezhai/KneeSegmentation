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
n_labels = 2

print(n_labels)
print(len(train_loader))

model = create_unet_model3D(input_image_size=train_data[0][0].shape,output_image_size=train_data[0][1].shape[0], n_labels=n_labels,layers=4, lowest_resolution=16,
                            mode='classification')



model.summary()



callbacks = [cbks.ModelCheckpoint(results_dir+'segmentation-weights_3D3.h5' , monitor='val_loss', save_best_only=True),
            cbks.ReduceLROnPlateau(monitor='val_loss', factor=0.1),TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)]

model.fit_generator(generator=iter(train_loader), steps_per_epoch=np.ceil(len(train_data)/batch_size),
                    epochs=100, verbose=1, callbacks=callbacks,
                    shuffle=True,
                    validation_data=iter(val_loader), validation_steps=np.ceil(len(val_data)/batch_size),
                    class_weight=None, max_queue_size=10,
                    workers=1, use_multiprocessing=False,  initial_epoch=0)
