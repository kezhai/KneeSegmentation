import h5py
import numpy as np
import os
from keras.utils import to_categorical
from sklearn.utils import class_weight
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
from mrcnn import model as modellib
from mrcnn.model import log
from mpl_toolkits.mplot3d import Axes3D



# from keras.models import Model
# from keras.layers import *
# from keras.optimizers import Adam
# from keras.regularizers import l2
# from keras.preprocessing.image import ImageDataGenerator
# import keras.backend as K
# from keras.callbacks import LearningRateScheduler, ModelCheckpoint




# with h5py.File('Data/020926-002-L-Turbo 3D T1, 1-2004.mat', 'r') as file:
#     print(list(file.keys()))
#     scan = np.array(file['scan'])
#     CartTM = np.array(file['CartTM'])


base_path = '/home/hzd551/3dknee/data/'
results_dir = 'new_result/'


train_path_left = base_path + 'train/left'
train_path_right = base_path + 'train/right'
val_path_left = base_path + 'val/left'
val_path_right = base_path + 'val/right'
test_path_left = base_path + 'test/left'
test_path_right = base_path + 'test/right'

ROI1 = np.save(results_dir+'pred_2d_3dim1.npy')
ROI2 = np.save(results_dir+'pred_2d_3dim2.npy')
ROI3 = np.save(results_dir+'pred_2d_3dim3.npy')



def onehot(X, y=None):
    """
    Assumes channel dim is last dimension
    """
    xshape = list(X.shape)
#    print(xshape)
    xx = to_categorical(X,num_classes=2)
#    print(xx.shape)
    xx = xx.reshape(xshape + [xx.shape[-1]])
#    print(xx.shape)
    return xx

def padwithzeros(vector, pad_width, iaxis, kwargs):
    """
    Pad with zero
    """
    vector[:pad_width[0]] = 0
    vector[-pad_width[1]:] = 0
    return vector


def pad(image,x):
    all = np.zeros(170*170*(image.shape[2]+2*x)).reshape(170,170,(image.shape[2]+2*x))
    all[:,:,x:image.shape[2]+x] = image
    return all

def reverse(scan):
    """
    reverse a image from z axis
    """
    scan=scan[:,:,::-1]
    return scan



def data_split_2d(scan,CartTM,cart_cor,left=True):
    """
    2D slices generate slice, mask and weights
    """
    scan = scan[..., np.newaxis]
    mask = onehot(CartTM).reshape(scan.size, 2)
    weights = np.zeros((scan.size))
    for i in range(scan.size):
        if mask[i][0] == 0:
            weights[i] = 1 - len(cart_cor) / scan.size
        else:
            weights[i] = len(cart_cor) / scan.size
    return scan, mask, weights

def ROI(file,left=True):
    """
    finding the center of cartilage
    :param file: 3D scan
    :param left: if the image is left or not
    :return: the coordinate of center point
    """
    scan = np.array(file['scan'])
    CartTM = np.array(file['CartTM'])
    if left is False:
               scan = reverse(scan)
               CartTM = reverse(CartTM)
    scan = np.lib.pad(scan, 2, padwithzeros)
    CartTM = np.lib.pad(CartTM, 2, padwithzeros)
    CartTM_cor = np.array(np.where(CartTM > 0)).T
    x = int(round(max(np.array(CartTM_cor.T[0])) + min(np.array(CartTM_cor.T[0])))/2)
    y = int(round(max(np.array(CartTM_cor.T[1])) + min(np.array(CartTM_cor.T[1])))/2)
    z = int(round(max(np.array(CartTM_cor.T[2])) + min(np.array(CartTM_cor.T[2])))/2)
    print(x,y,z)
    return [x,y,z]

def data_split(file,number,left=True):
    """
    3D scan generate ROI, ROI mask and weights
    :param file:scan
    :param left:if the scan is left or not
    :return:ROI,ROI_mask,weights
    """
    scan = np.array(file['scan'])
    CartTM = np.array(file['CartTM'])
    pred = ROI1[number]
    if left is False:
               scan = reverse(scan)
               CartTM = reverse(CartTM)
    scan = np.lib.pad(scan, 2, padwithzeros)
    CartTM = np.lib.pad(CartTM, 2, padwithzeros)
    CartTM_cor = np.array(np.where(CartTM > 0)).T
    print(len(CartTM_cor))
    # x = int(round(max(np.array(CartTM_cor.T[0])) + min(np.array(CartTM_cor.T[0])))/2)
    # y = int(round(max(np.array(CartTM_cor.T[1])) + min(np.array(CartTM_cor.T[1])))/2)
    # z = int(round(max(np.array(CartTM_cor.T[2])) + min(np.array(CartTM_cor.T[2])))/2)
    x = int(round((pred[0] + pred[1])/2))
    y = int(round((pred[2] + pred[3])/2))
    z = int(round((pred[4] + pred[5])/2))
    # print(x,y,z)
    ROI = scan[x-16:x+16, y-40:y+40, z-24:z+24]
    ROI_mask = CartTM[x-16:x+16, y-40:y+40, z-24:z+24]
    print(scan.shape)
    print(CartTM.shape)
    bbx = utils.extract_bboxes(CartTM)
    print(bbx.shape)
#     ROI = scan[x-8:x+8, y-32:y+32, z-16:z+16]
#     ROI_mask = CartTM[x-8:x+8, y-32:y+32, z-16:z+16]
#    print(ROI_mask.shape)
    cart_cor = np.array(np.where(ROI_mask > 0)).T
#    back_cor = np.array(np.where(ROI_mask == 0)).T
    print(ROI.shape)
    print(len(cart_cor))
    print(len(cart_cor)/len(CartTM_cor))
    print(len(cart_cor)/ROI.size)
#    print(back_cor.shape)
#    ROI_mask = np.array([np.array(np.where(ROI_mask > 0))*8, np.array(np.where(ROI_mask ==0))])
    ROI = ROI[...,np.newaxis]
    ROI_mask = onehot(ROI_mask).reshape(ROI.size,2)
    weights = np.zeros((ROI.size))
    for i in range(ROI.size):
        if ROI_mask[i][0] == 0:
            weights[i] = 1-len(cart_cor)/ROI.size
        else:
            weights[i] = len(cart_cor)/ROI.size
    return ROI,ROI_mask,weights

def data_split_duo(file, left=True):
    """the function for training on whole scan but not ROI"""
    scan = np.array(file['scan'])
    CartTM = np.array(file['CartTM'])
    print(scan.shape)
    if scan.shape[2]>104:
        scan = scan[:,:,int(scan.shape[2]/2-52):int(scan.shape[2]/2+52)]
        CartTM = CartTM[:, :, int(scan.shape[2] / 2 - 52):int(scan.shape[2] / 2 + 52)]
    CartTM_cor = np.array(np.where(CartTM > 0)).T
    print(scan.shape)
    print(len(CartTM_cor) / scan.size)
    scan = scan[...,np.newaxis]
    CartTM = onehot(CartTM).reshape(CartTM.size,2)
    weights = np.zeros((scan.size))
    for i in range(scan.size):
        if CartTM[i][0] == 0:
            weights[i] = 1-len(CartTM_cor)/scan.size
        else:
            weights[i] = len(CartTM_cor)/scan.size
    return scan, CartTM, weights
    #ROI_mask = CartTM[57:73, 33:97, 54:86]


def data_generate_2D_1dim(p, path_left, path_right,path_left2=None, path_right2=None, train = False):
    """sagittal slice generation as input of 2D U-net"""
    number = 0
    number_slice = 0
    data_all = []
    index = []

    text_files = [f for f in os.listdir(path_left) if f.endswith('.mat')]
    for files in text_files:
        print('%s/%s'%(path_left,files))
        print(number)
        each_x=0
        each_y=0
        each_z=0
        with h5py.File('%s/%s'%(path_left,files),'r') as file:
            scan = np.array(file['scan'])
            CartTM = np.array(file['CartTM'])
            scan = pad(scan, p)
            CartTM = pad(CartTM, p)

            scan = scan[0:int(104 + 2 * p),
                0:int(104 + 2 * p),
                int(scan.shape[2] - (104 + 2 * p)):int(scan.shape[2])]
            CartTM = CartTM[0:int(104 + 2 * p),
                0:int(104 + 2 * p),
                int(scan.shape[2] - (104 + 2 * p)):int(scan.shape[2])]

            for i in range(scan.shape[0]):
                slice = scan[i, :, :]
                slice_mask = CartTM[i, :, :]
                cart_cor = np.array(np.where(slice_mask > 0)).T
                if len(cart_cor) / scan.size>0:
                    number_slice+=1
                    each_x+=1
                    data = data_split_2d(slice,slice_mask,cart_cor,left=True)
                    data_all.append(data)
            number+=1

        print(each_x)
        index.append(each_x+each_y+each_z)

    text_files = [f for f in os.listdir(path_right) if f.endswith('.mat')]
    for files in text_files:
        print('%s/%s'%(path_right,files))
        print(number)
        each_x=0
        each_y=0
        each_z=0
        with h5py.File('%s/%s'%(path_right,files),'r') as file:
            scan = np.array(file['scan'])
            CartTM = np.array(file['CartTM'])
            scan = reverse(scan)
            CartTM = reverse(CartTM)
            scan = pad(scan, p)
            CartTM = pad(CartTM, p)

            scan = scan[0:int(104 + 2 * p),
                0:int(104 + 2 * p),
                int(scan.shape[2] - (104 + 2 * p)):int(scan.shape[2])]
            CartTM = CartTM[0:int(104 + 2 * p),
                0:int(104 + 2 * p),
                int(scan.shape[2] - (104 + 2 * p)):int(scan.shape[2])]

            for i in range(scan.shape[0]):
                slice = scan[i, :, :]
                slice_mask = CartTM[i, :, :]
                cart_cor = np.array(np.where(slice_mask > 0)).T
                if len(cart_cor) / scan.size>0:
                    number_slice+=1
                    each_x+=1
                    data = data_split_2d(slice,slice_mask,cart_cor,left=False)
                    data_all.append(data)
            number+=1
        print(each_x)
        index.append(each_x+each_y+each_z)


    if train == True:

        text_files = [f for f in os.listdir(path_left2) if f.endswith('.mat')]
        for files in text_files:
            print('%s/%s'%(path_left2,files))
            print(number)
            each_x=0
            each_y=0
            each_z=0
            with h5py.File('%s/%s'%(path_left2,files),'r') as file:
                scan = np.array(file['scan'])
                CartTM = np.array(file['CartTM'])
                scan = pad(scan, p)
                CartTM = pad(CartTM, p)

                scan = scan[0:int(104 + 2 * p),
                    0:int(104 + 2 * p),
                    int(scan.shape[2] - (104 + 2 * p)):int(scan.shape[2])]
                CartTM = CartTM[0:int(104 + 2 * p),
                    0:int(104 + 2 * p),
                    int(scan.shape[2] - (104 + 2 * p)):int(scan.shape[2])]

                for i in range(scan.shape[0]):
                    slice = scan[i, :, :]
                    slice_mask = CartTM[i, :, :]
                    cart_cor = np.array(np.where(slice_mask > 0)).T
                    if len(cart_cor) / scan.size>0:
                        number_slice+=1
                        each_x+=1
                        data = data_split_2d(slice,slice_mask,cart_cor,left=True)
                        data_all.append(data)
                number+=1

            print(each_x)
            index.append(each_x+each_y+each_z)

        text_files = [f for f in os.listdir(path_right2) if f.endswith('.mat')]
        for files in text_files:
            print('%s/%s'%(path_right2,files))
            print(number)
            each_x=0
            each_y=0
            each_z=0
            with h5py.File('%s/%s'%(path_right2,files),'r') as file:
                scan = np.array(file['scan'])
                CartTM = np.array(file['CartTM'])
                scan = reverse(scan)
                CartTM = reverse(CartTM)
                scan = pad(scan, p)
                CartTM = pad(CartTM, p)

                scan = scan[0:int(104 + 2 * p),
                    0:int(104 + 2 * p),
                    int(scan.shape[2] - (104 + 2 * p)):int(scan.shape[2])]
                CartTM = CartTM[0:int(104 + 2 * p),
                    0:int(104 + 2 * p),
                    int(scan.shape[2] - (104 + 2 * p)):int(scan.shape[2])]

                for i in range(scan.shape[0]):
                    slice = scan[i, :, :]
                    slice_mask = CartTM[i, :, :]
                    cart_cor = np.array(np.where(slice_mask > 0)).T
                    if len(cart_cor) / scan.size>0:
                        number_slice+=1
                        each_x+=1
                        data = data_split_2d(slice,slice_mask,cart_cor,left=False)
                        data_all.append(data)
                number+=1
            print(each_x)
            index.append(each_x+each_y+each_z)
    print(number_slice)
    return data_all,index

def data_generate_2D(p, path_left, path_right,path_left2=None, path_right2=None, train = False):
    """triplanar slices generating as input of 2D U-net"""
    number = 0
    number_slice = 0
    data_all = []
    index = []
    full_index=[]

    text_files = [f for f in os.listdir(path_left) if f.endswith('.mat')]
    for files in text_files:
        print('%s/%s'%(path_left,files))
        print(number)
        each_x=0
        each_y=0
        each_z=0
        with h5py.File('%s/%s'%(path_left,files),'r') as file:
            scan = np.array(file['scan'])
            CartTM = np.array(file['CartTM'])
            scan = pad(scan, p)
            CartTM = pad(CartTM, p)

            scan = scan[0:int(104 + 2 * p),
                0:int(104 + 2 * p),
                int(scan.shape[2] - (104 + 2 * p)):int(scan.shape[2])]
            CartTM = CartTM[0:int(104 + 2 * p),
                0:int(104 + 2 * p),
                int(scan.shape[2] - (104 + 2 * p)):int(scan.shape[2])]

            for i in range(scan.shape[0]):
                slice = scan[i, :, :]
                slice_mask = CartTM[i, :, :]
                cart_cor = np.array(np.where(slice_mask > 0)).T
                if len(cart_cor) / scan.size>0:
                    number_slice+=1
                    each_x+=1
                    data = data_split_2d(slice,slice_mask,cart_cor,left=True)
                    data_all.append(data)
            for j in range(scan.shape[1]):
                slice = scan[:, j, :]
                slice_mask = CartTM[:, j, :]
                cart_cor = np.array(np.where(slice_mask > 0)).T
                if len(cart_cor) / scan.size>0:
                    number_slice+=1
                    each_y+=1
                    data = data_split_2d(slice,slice_mask,cart_cor,left=True)
                    data_all.append(data)
            for z in range(scan.shape[2]):
                slice = scan[:, :, z]
                slice_mask = CartTM[:, :, z]
                cart_cor = np.array(np.where(slice_mask > 0)).T
                if len(cart_cor) / scan.size>0:
                    number_slice+=1
                    each_z+=1
                    data = data_split_2d(slice,slice_mask,cart_cor,left=True)
                    data_all.append(data)
            number+=1

        print(each_x,each_y,each_z)
        index.append(each_x+each_y+each_z)
        full_index.append([each_x,each_y,each_z])

    text_files = [f for f in os.listdir(path_right) if f.endswith('.mat')]
    for files in text_files:
        print('%s/%s'%(path_right,files))
        print(number)
        each_x=0
        each_y=0
        each_z=0
        with h5py.File('%s/%s'%(path_right,files),'r') as file:
            scan = np.array(file['scan'])
            CartTM = np.array(file['CartTM'])
            scan = reverse(scan)
            CartTM = reverse(CartTM)
            scan = pad(scan, p)
            CartTM = pad(CartTM, p)

            scan = scan[0:int(104 + 2 * p),
                0:int(104 + 2 * p),
                int(scan.shape[2] - (104 + 2 * p)):int(scan.shape[2])]
            CartTM = CartTM[0:int(104 + 2 * p),
                0:int(104 + 2 * p),
                int(scan.shape[2] - (104 + 2 * p)):int(scan.shape[2])]

            for i in range(scan.shape[0]):
                slice = scan[i, :, :]
                slice_mask = CartTM[i, :, :]
                cart_cor = np.array(np.where(slice_mask > 0)).T
                if len(cart_cor) / scan.size>0:
                    number_slice+=1
                    each_x+=1
                    data = data_split_2d(slice,slice_mask,cart_cor,left=False)
                    data_all.append(data)
            for j in range(scan.shape[1]):
                slice = scan[:, j, :]
                slice_mask = CartTM[:, j, :]
                cart_cor = np.array(np.where(slice_mask > 0)).T
                if len(cart_cor) / scan.size>0:
                    number_slice+=1
                    each_y+=1
                    data = data_split_2d(slice,slice_mask,cart_cor,left=False)
                    data_all.append(data)
            for z in range(scan.shape[2]):
                slice = scan[:, :, z]
                slice_mask = CartTM[:, :, z]
                cart_cor = np.array(np.where(slice_mask > 0)).T
                if len(cart_cor) / scan.size>0:
                    number_slice+=1
                    each_z+=1
                    data = data_split_2d(slice,slice_mask,cart_cor,left=False)
                    data_all.append(data)
            number+=1
        print(each_x,each_y,each_z)
        index.append(each_x+each_y+each_z)
        full_index.append([each_x,each_y,each_z])


    if train == True:

        text_files = [f for f in os.listdir(path_left2) if f.endswith('.mat')]
        for files in text_files:
            print('%s/%s'%(path_left2,files))
            print(number)
            each_x=0
            each_y=0
            each_z=0
            with h5py.File('%s/%s'%(path_left2,files),'r') as file:
                scan = np.array(file['scan'])
                CartTM = np.array(file['CartTM'])
                scan = pad(scan, p)
                CartTM = pad(CartTM, p)

                scan = scan[0:int(104 + 2 * p),
                    0:int(104 + 2 * p),
                    int(scan.shape[2] - (104 + 2 * p)):int(scan.shape[2])]
                CartTM = CartTM[0:int(104 + 2 * p),
                    0:int(104 + 2 * p),
                    int(scan.shape[2] - (104 + 2 * p)):int(scan.shape[2])]

                for i in range(scan.shape[0]):
                    slice = scan[i, :, :]
                    slice_mask = CartTM[i, :, :]
                    cart_cor = np.array(np.where(slice_mask > 0)).T
                    if len(cart_cor) / scan.size>0:
                        number_slice+=1
                        each_x+=1
                        data = data_split_2d(slice,slice_mask,cart_cor,left=True)
                        data_all.append(data)
                for j in range(scan.shape[1]):
                    slice = scan[:, j, :]
                    slice_mask = CartTM[:, j, :]
                    cart_cor = np.array(np.where(slice_mask > 0)).T
                    if len(cart_cor) / scan.size>0:
                        number_slice+=1
                        each_y+=1
                        data = data_split_2d(slice,slice_mask,cart_cor,left=True)
                        data_all.append(data)
                for z in range(scan.shape[2]):
                    slice = scan[:, :, z]
                    slice_mask = CartTM[:, :, z]
                    cart_cor = np.array(np.where(slice_mask > 0)).T
                    if len(cart_cor) / scan.size>0:
                        number_slice+=1
                        each_z+=1
                        data = data_split_2d(slice,slice_mask,cart_cor,left=True)
                        data_all.append(data)
                number+=1

            print(each_x,each_y,each_z)
            index.append(each_x+each_y+each_z)
            full_index.append([each_x,each_y,each_z])

        text_files = [f for f in os.listdir(path_right2) if f.endswith('.mat')]
        for files in text_files:
            print('%s/%s'%(path_right2,files))
            print(number)
            each_x=0
            each_y=0
            each_z=0
            with h5py.File('%s/%s'%(path_right2,files),'r') as file:
                scan = np.array(file['scan'])
                CartTM = np.array(file['CartTM'])
                scan = reverse(scan)
                CartTM = reverse(CartTM)
                scan = pad(scan, p)
                CartTM = pad(CartTM, p)

                scan = scan[0:int(104 + 2 * p),
                    0:int(104 + 2 * p),
                    int(scan.shape[2] - (104 + 2 * p)):int(scan.shape[2])]
                CartTM = CartTM[0:int(104 + 2 * p),
                    0:int(104 + 2 * p),
                    int(scan.shape[2] - (104 + 2 * p)):int(scan.shape[2])]

                for i in range(scan.shape[0]):
                    slice = scan[i, :, :]
                    slice_mask = CartTM[i, :, :]
                    cart_cor = np.array(np.where(slice_mask > 0)).T
                    if len(cart_cor) / scan.size>0:
                        number_slice+=1
                        each_x+=1
                        data = data_split_2d(slice,slice_mask,cart_cor,left=False)
                        data_all.append(data)
                for j in range(scan.shape[1]):
                    slice = scan[:, j, :]
                    slice_mask = CartTM[:, j, :]
                    cart_cor = np.array(np.where(slice_mask > 0)).T
                    if len(cart_cor) / scan.size>0:
                        number_slice+=1
                        each_y+=1
                        data = data_split_2d(slice,slice_mask,cart_cor,left=False)
                        data_all.append(data)
                for z in range(scan.shape[2]):
                    slice = scan[:, :, z]
                    slice_mask = CartTM[:, :, z]
                    cart_cor = np.array(np.where(slice_mask > 0)).T
                    if len(cart_cor) / scan.size>0:
                        number_slice+=1
                        each_z+=1
                        data = data_split_2d(slice,slice_mask,cart_cor,left=False)
                        data_all.append(data)
                number+=1
            print(each_x,each_y,each_z)
            index.append(each_x+each_y+each_z)
            full_index.append([each_x,each_y,each_z])
    print(number_slice)
    return data_all,index,full_index

def data_generate_3D(path_left, path_right,path_left2=None, path_right2=None,train=False):
    """data preparation before input 3D U-net"""
    number = 0
#    data=np.zeros(2*2*16*64*32*2).reshape(2,2,16,64,32,2)
    data_all = []
    ROI_points = []
    scan = []
    mask = []

    text_files = [f for f in os.listdir(path_left) if f.endswith('.mat')]
    for files in text_files:
        print('%s/%s'%(path_left,files))
        print(number)
        with h5py.File('%s/%s'%(path_left,files),'r') as file:
            data = data_split(file,number, left=True)
            ROI_points.append(ROI(file,left = True))
            scan.append(np.array(file['scan']))
            mask.append(np.array(file['CartTM']))
            data_all.append(data)
            number+=1

    text_files = [f for f in os.listdir(path_right) if f.endswith('.mat')]
    for files in text_files:
        print('%s/%s'%(path_right,files))
        print(number)
        with h5py.File('%s/%s'%(path_right,files),'r') as file:
            data = data_split(file,number, left=False)
            ROI_points.append(ROI(file,left = False))
            scan.append(reverse(np.array(file['scan'])))
            mask.append(reverse(np.array(file['CartTM'])))
            data_all.append(data)
            number+=1

    if train ==True:
        text_files = [f for f in os.listdir(path_left2) if f.endswith('.mat')]
        for files in text_files:
            print('%s/%s' % (path_left2, files))
            print(number)
            with h5py.File('%s/%s' % (path_left2, files), 'r') as file:
                data = data_split(file, number,left=True)
                ROI_points.append(ROI(file, left=True))
                scan.append(np.array(file['scan']))
                mask.append(np.array(file['CartTM']))
                data_all.append(data)
                number += 1

        text_files = [f for f in os.listdir(path_right2) if f.endswith('.mat')]
        for files in text_files:
            print('%s/%s' % (path_right2, files))
            print(number)
            with h5py.File('%s/%s' % (path_right2, files), 'r') as file:
                data = data_split(file, number, left=False)
                ROI_points.append(ROI(file, left=False))
                scan.append(reverse(np.array(file['scan'])))
                mask.append(reverse(np.array(file['CartTM'])))
                data_all.append(data)
                number += 1
    return data_all,ROI_points,scan,mask





##3-fold cross validation

#2D
#train_data,index_train,full_index_train=data_generate_2D(12,test_path_left,test_path_right,val_path_left,val_path_right,train=True)
#val_data,index_val,full_index_val=data_generate_2D(12,train_path_left,train_path_right)





#2D_1dim
#train_data,index_train=data_generate_2D_1dim(12,test_path_left,test_path_right,train_path_left,train_path_right,train=True)
#val_data,index_val=data_generate_2D_1dim(12,val_path_left,val_path_right)




#3D
#1
#train_data,ROI_train,scan_train,mask_train=data_generate_3D(train_path_left,train_path_right,val_path_left,val_path_right,train=True)
#val_data,ROI_test,scan_test,mask_test=data_generate_3D(test_path_left,test_path_right)

#2
#train_data,ROI_train,scan_train,mask_train=data_generate_3D(test_path_left,test_path_right,val_path_left,val_path_right,train=True)
#val_data,ROI_test,scan_test,mask_test=data_generate_3D(train_path_left,train_path_right)

#3
train_data,ROI_train,scan_train,mask_train=data_generate_3D(test_path_left,test_path_right,train_path_left,train_path_right,train=True)
val_data,ROI_test,scan_test,mask_test=data_generate_3D(val_path_left,val_path_right)










print('data load complete')



