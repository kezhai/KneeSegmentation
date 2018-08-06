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


model = create_unet_model2D(input_image_size=train_data[0][0].shape, output_image_size=train_data[0][1].shape[0],n_labels=n_labels, layers=4)

model.summary()
#plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)



# callbacks = [cbks.ModelCheckpoint(results_dir+'segmentation-weights.h5' , monitor='val_loss', save_best_only=True),
#             cbks.ReduceLROnPlateau(monitor='val_loss', factor=0.1),TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)]
#
# model.fit_generator(generator=iter(train_loader), steps_per_epoch=np.ceil(len(train_data)/batch_size),
#                     epochs=5, verbose=1, callbacks=callbacks,
#                     shuffle=True,
#                     validation_data=iter(val_loader), validation_steps=np.ceil(len(val_data)/batch_size),
#                     class_weight=None, max_queue_size=10,
#                     workers=1, use_multiprocessing=False,  initial_epoch=0)


model.load_weights(results_dir+'segmentation-weights_2d11.h5')
x_all=[]
y_all=[]
pred_all=[]


# 2D
for j in range(len(index_train)):
    x_t=[]
    y_t=[]
    pred_t=[]
    for i in range(index_train[j]):
        real_x = train_data[i][0][np.newaxis,...]
        real_y = train_data[i][1][np.newaxis,...]
        print(real_x.shape)
        print(real_y.shape)
        real_y_pred = model.predict(real_x)
        print(real_y_pred.shape)
        acc = model.evaluate(real_x,real_y)
        print(acc)
        x_t.append(real_x)
        y_t.append(real_y)
        pred_t.append(real_y_pred)
    x_all.append(x_t)
    y_all.append(y_t)
    pred_all.append(pred_t)
#np.save(results_dir+'prediction_2D.npy',real_val_y_pred)

#np.savez_compressed(results_dir+'2D.npz', x=real_val_x, y=real_val_y, pred = real_val_y_pred)
np.savez_compressed(results_dir+'2D11.npz', x=x_all, y=y_all, pred = pred_all)
