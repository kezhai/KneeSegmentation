import h5py
import numpy as np
import os, random
from skimage.measure import label
from skimage import morphology

random.seed(7)

patch_size = 14

base_path = 'Data/'

results_dir = 'Results/'


train_path_left = base_path + 'train/left'
train_path_right = base_path + 'train/right'
val_path_left = base_path + 'val/left'
val_path_right = base_path + 'val/right'

#data = np.load(results_dir + '2D716.npz')
#index_train = np.load(results_dir+'ind_train.npy')

pred = np.load(results_dir + 'pred_2d_3dim.npy')
DSC = np.load(results_dir + 'dsc_2d_3dim2.npy')
real = np.load(results_dir+'ROI_train.npy')


DSC_mean =[]
for i in range(len(DSC)):
    DSC_mean.append(np.mean(DSC[i]))
print(DSC_mean)


def post(pred):
    """regular size ROI"""
    x = int(round((pred[0] + pred[1])/2))
    y = int(round((pred[2] + pred[3])/2))
    z = int(round((pred[4] + pred[5])/2))
#    print(x,y,z)
    neo = [x-16,x+16, y-40,y+40, z-24,z+24]
    return neo

def post_1dim(pred):
    x = int(round(pred[0] + pred[1]/2))
    y = int(round(pred[2] + pred[3]/2))
    print(x,y)
    neo = [x-40,x+40, y-24,y+24]
    return neo

pred_neo =[]
for i in range(len(real)):
    pred_neo.append(post(pred[i]))
#print(pred_neo)





def IOU(real,pred):
    up = (min(real[1],pred[1])-max(real[0],pred[0]))* (min(real[3],pred[3])-max(real[2],pred[2]))* (min(real[5],pred[5])-max(real[4],pred[4]))
    down = (max(real[1], pred[1]) - min(real[0], pred[0])) * (max(real[3], pred[3]) - min(real[2], pred[2])) * (
                max(real[5], pred[5]) - min(real[4], pred[4]))
    return up/down

def IOT(real,pred):
    up = (min(real[1],pred[1])-max(real[0],pred[0]))* (min(real[3],pred[3])-max(real[2],pred[2]))* (min(real[5],pred[5])-max(real[4],pred[4]))
    down = (real[1]-real[0])*(real[3]-real[2])*(real[5]-real[4])
    return up/down

def IOT_1dim(real,pred):
    up=(min(real[3],pred[1])-max(real[2],pred[0]))* (min(real[5],pred[3])-max(real[4],pred[2]))
    down =(real[3]-real[2])*(real[5]-real[4])
    return up/down


IOU_all =[]
for i in range(len(real)):
    IOU_all.append(IOT(real[i],pred_neo[i]))

print(IOU_all)
