import numpy as np
import h5py
import os

#### The purpose of this file is to generate a a regular size ROI corresponding to the true mask for calculate DSC and IOT


def padwithzeros(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = 0
    vector[-pad_width[1]:] = 0
    return vector

def pad(image,x):
    all = np.zeros(170*170*(104+2*x)).reshape(170,170,(104+2*x))
    all[:,:,x:104+x] = image
    return all

# with h5py.File('Data/val/left/101280-001-L-Turbo 3D T1, 1-2004.mat', 'r') as file:
#     print(list(file.keys()))
#     scan = np.array(file['scan'])
#     CartTM = np.array(file['CartTM'])

def reverse(scan):
    scan=scan[:,:,::-1]
    return scan


base_path = 'data/'
results_dir = 'Results/'

train_path_left = base_path + 'train/left'
train_path_right = base_path + 'train/right'
val_path_left = base_path + 'val/left'
val_path_right = base_path + 'val/right'


def ROI(file,left=True):
    scan = np.array(file['scan'])
    CartTM = np.array(file['CartTM'])
    if left is False:
               scan = reverse(scan)
               CartTM = reverse(CartTM)
    scan = np.lib.pad(scan, 2, padwithzeros)
    CartTM = np.lib.pad(CartTM, 2, padwithzeros)
    CartTM_cor = np.array(np.where(CartTM > 0)).T
    x = [min(np.array(CartTM_cor.T[0])) , max(np.array(CartTM_cor.T[0]))]
    y = [min(np.array(CartTM_cor.T[1])) , max(np.array(CartTM_cor.T[1]))]
    z = [min(np.array(CartTM_cor.T[2])) , max(np.array(CartTM_cor.T[2]))]
    print([x[0],x[1],y[0],y[1],z[0],z[1]])
    return [x[0],x[1],y[0],y[1],z[0],z[1]]

def data_split(file,left=True):
    scan = np.array(file['scan'])
    CartTM = np.array(file['CartTM'])
    if left is False:
               scan = reverse(scan)
               CartTM = reverse(CartTM)
    scan = np.lib.pad(scan, 2, padwithzeros)
    CartTM = np.lib.pad(CartTM, 2, padwithzeros)
    CartTM_cor = np.array(np.where(CartTM > 0)).T
    print(len(CartTM_cor))
    x = int(round(max(np.array(CartTM_cor.T[0])) + min(np.array(CartTM_cor.T[0])))/2)
    y = int(round(max(np.array(CartTM_cor.T[1])) + min(np.array(CartTM_cor.T[1])))/2)
    z = int(round(max(np.array(CartTM_cor.T[2])) + min(np.array(CartTM_cor.T[2])))/2)
    print([x-16,x+16, y-40,y+40, z-24,z+24])
    return [x-16,x+16, y-40,y+40, z-24,z+24]


def data_generate_3D(path_left, path_right):
    number = 0
#    data=np.zeros(2*2*16*64*32*2).reshape(2,2,16,64,32,2)
    data_all = []
    ROI_points = []
    scan = []
    mask = []

    text_files = [f for f in os.listdir(path_left) if f.endswith('.mat')]
    for files in text_files:
        # print('%s/%s'%(path_left,files))
        # print(number)
        with h5py.File('%s/%s'%(path_left,files),'r') as file:
            ROI_points.append(data_split(file,left = True))
            scan.append(np.array(file['scan']))
            mask.append(np.array(file['CartTM']))
            number+=1

    text_files = [f for f in os.listdir(path_right) if f.endswith('.mat')]
    for files in text_files:
        # print('%s/%s'%(path_right,files))
        # print(number)
        with h5py.File('%s/%s'%(path_right,files),'r') as file:
            ROI_points.append(data_split(file,left = False))
            scan.append(reverse(np.array(file['scan'])))
            mask.append(reverse(np.array(file['CartTM'])))
            number+=1
    return ROI_points

ROI_train=data_generate_3D(train_path_left,train_path_right)
#ROI_test=data_generate_3D(val_path_left,val_path_right)

print(ROI_train)
np.save(results_dir+'ROI_train.npy',ROI_train)
