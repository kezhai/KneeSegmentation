import numpy as np
import h5py
import os, random
import keras.backend.tensorflow_backend as K
from keras.models import load_model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, AveragePooling2D
from keras.optimizers import adam,SGD
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
from itertools import product
from functools import partial

#####section 2 train CNN models with patches

np.random.seed(7)

results_dir = 'new_result/'

train = np.load(results_dir+'train_again.npz')
patch_train = train['patch']
y_train = train['label']

val = np.load(results_dir+'val_again.npz')
patch_val = val['patch']
y_val = val['label']

test = np.load(results_dir+'test_again.npz')
patch_test = test['patch']
y_test = test['label']



print(patch_train.shape)
print(y_train.shape)

print(patch_val.shape)
print(y_val.shape)




# batch_size to train
batch_size = 64
# number of output classes
nb_classes = 2
# number of epochs to train
nb_pool = 2
# number of convolutional filters to use
nb_filters = 20
# convolution kernel size
nb_conv = 5





model = Sequential()
model.add(Convolution2D(nb_filters, nb_conv, nb_conv, dim_ordering='th', border_mode='valid',input_shape=(3, 28, 28)))
convout1 = Activation('relu')
model.add(convout1)
model.add(AveragePooling2D(pool_size=(nb_pool, nb_pool), dim_ordering="th"))

model.add(Convolution2D(nb_filters, nb_conv, nb_conv, dim_ordering='th', border_mode='valid'))
convout2 = Activation('relu')
model.add(convout2)

model.add(Convolution2D(nb_filters, nb_conv, nb_conv, dim_ordering='th', border_mode='valid'))
convout3 = Activation('relu')
model.add(convout3)

model.add(Flatten())
model.add(Dense(1000, batch_input_shape=(None, 3, 28, 28)))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(2))
model.add(Activation('softmax'))

adam = adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)


#sgd = SGD(lr=1e-5, decay=0.0, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])


xx = np.concatenate((patch_test,patch_train))
ll = np.concatenate((y_test,y_train))

print(xx.shape)
print(ll.shape)



#earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')

model.fit(xx,ll, nb_epoch=60, batch_size=256 ,class_weight={0:1,1:5},  validation_data=(patch_val,y_val),shuffle=True)

model.save_weights(results_dir+'knee_2dcnn_3.h5')
