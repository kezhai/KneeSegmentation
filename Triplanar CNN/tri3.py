import h5py
import numpy as np
from keras.utils import to_categorical
from sklearn.utils import shuffle
import os, random
import keras.backend.tensorflow_backend as K
from keras.models import load_model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, AveragePooling2D
from keras.optimizers import adam,SGD
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
import time


####section 3 predict with the trained models


random.seed(7)

base_path = 'data/'
results_dir = 'new_result/'


train_path_left = base_path + 'train/left'
train_path_right = base_path + 'train/right'
val_path_left = base_path + 'val/left'
val_path_right = base_path + 'val/right'
test_path_left = base_path + 'test/left'
test_path_right = base_path + 'test/right'



def padwithzeros(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = 0
    vector[-pad_width[1]:] = 0
    return vector

def reverse(scan):
    scan=scan[:,:,::-1]
    return scan

patch_size = 14
predict_all = []



nb_classes = 2
# number of epochs to train
nb_epoch = 5
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
sgd = SGD(lr=1e-5, decay=0.0, momentum=0.9, nesterov=True)

model.load_weights(results_dir+ 'knee_2dcnn_3.h5')
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])




        



def eva(scan,CartFM):
    """
    big function for evaluation
    :param scan:
    :param CartFM:
    :return: predicted segmented image
    """

    CartFM_cor=np.array(np.where(CartFM>0)).T
    one_cor=np.array(CartFM_cor)
    loss_all=[]
    predict_image=[]


    scan= np.lib.pad(scan, patch_size, padwithzeros)
    print("----padding finished----")

    print(scan.shape[2])
    print(CartFM.shape[2])
    for z in range(CartFM.shape[2]):
        print(z)
        image = scan[:,:,z+patch_size]
        cart = CartFM[:,:,z]
        patch_all=np.zeros(cart.size*28*28*3).reshape(cart.size,3,28,28)
    #    label = np.ones(cart.size)

        for i in range(cart.size):
            patch_all[i,0,:,:]=scan[i//170+patch_size,i%170:i%170+2*patch_size,z:z+2*patch_size]
            patch_all[i,1,:,:]=scan[i//170:i//170+2*patch_size,i%170+patch_size,z:z+2*patch_size]
            patch_all[i,2,:,:]=scan[i//170:i//170+2*patch_size,i%170:i%170+2*patch_size,z+patch_size]
    #        print([i,cart[i//170,i%170]])
    #        if cart[i//170,i%170]==1:
    #            label[i]=0
        #    label[i] = cart[i//170,i%170,71]

        predict = model.predict(patch_all)
        predict_image.append(predict)
    return predict_image

text_files = [f for f in os.listdir(val_path_left) if f.endswith('.mat')]
for files in text_files:
    print('%s/%s'%(val_path_left,files))
    with h5py.File('%s/%s'%(val_path_left,files),'r') as file:
        print(list(file.keys()))
        scan = np.array(file['scan'])
        CartFM = np.array(file['CartTM'])
        predict_image = eva(scan,CartFM)
        predict_all.append(predict_image)

text_files = [f for f in os.listdir(val_path_right) if f.endswith('.mat')]
for files in text_files:
    print('%s/%s'%(val_path_right,files))
    with h5py.File('%s/%s'%(val_path_right,files),'r') as file:
        print(list(file.keys()))
        scan = reverse(np.array(file['scan']))
        CartFM = reverse(np.array(file['CartTM']))
        predict_image = eva(scan,CartFM)
        predict_all.append(predict_image)


np.save(results_dir +'predict_2dcnn_3',predict_all)
