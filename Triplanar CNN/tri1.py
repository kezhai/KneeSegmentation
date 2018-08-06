import h5py
import numpy as np
import random
import os
from keras.utils import to_categorical
from sklearn.utils import shuffle


base_path = 'data/'
results_dir = 'new_result/'


train_path_left = base_path + 'train/left'
train_path_right = base_path + 'train/right'
val_path_left = base_path + 'val/left'
val_path_right = base_path + 'val/right'
test_path_left = base_path + 'test/left'
test_path_right = base_path + 'test/right'



random.seed(7)
patch_size = 14
patch_twenty=[]
label_twenty=[]
patch_val=[]
label_val=[]
patch_test=[]
label_test=[]



###scetion 1
### sample and patch among the ROI


def reverse(scan):
    scan=scan[:,:,::-1]
    return scan

def ROI_finding(scan,CartTM):
    scan = np.lib.pad(scan, 2, padwithzeros)
    CartTM = np.lib.pad(CartTM, 2, padwithzeros)
    CartTM_cor = np.array(np.where(CartTM > 0)).T
    x = int(round(max(np.array(CartTM_cor.T[0])) + min(np.array(CartTM_cor.T[0])))/2)
    y = int(round(max(np.array(CartTM_cor.T[1])) + min(np.array(CartTM_cor.T[1])))/2)
    z = int(round(max(np.array(CartTM_cor.T[2])) + min(np.array(CartTM_cor.T[2])))/2)
    ROI = scan[x-16:x+16, y-40:y+40, z-24:z+24]
    ROI_mask = CartTM[x-16:x+16, y-40:y+40, z-24:z+24]
    return ROI,ROI_mask


def sample(CartFM):
    """random sample"""
    CartFM_cor=np.array(np.where(CartFM>0)).T
    print(CartFM.shape)
    print(CartFM_cor.shape)
    ran_num = random.sample(range(len(CartFM_cor)),20)
    ran = CartFM_cor[ran_num]
    #CartFM_cor = np.repeat(CartFM_cor,8,axis=0)
    CartFM_cor2 =  np.array(np.where(CartFM==0)).T
    dist_all = []

    for i in CartFM_cor2:
        dist_old = 10000
        for a in range(len(ran)):
            dist = np.linalg.norm(i - ran[a])
            if dist<dist_old:
                dist_old = dist
        dist_all.append(dist_old)
    prob = 2/len(dist_all)-dist_all/sum(dist_all)
    random_cor= random.choices(CartFM_cor2, weights=prob,k=5*len(CartFM_cor))
    print(len(CartFM_cor))
    print(len(random_cor))
    return CartFM_cor,random_cor

def padwithzeros(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = 0
    vector[-pad_width[1]:] = 0
    return vector

def patch(scan,zero_cor_old,one_cor_old,patch_size):
    """draw patches according sampling result"""
    scan = np.lib.pad(scan, patch_size, padwithzeros)
    zero_cor_old = np.array(zero_cor_old)
    one_cor_old = np.array(one_cor_old)
    patch_all = np.zeros((len(one_cor_old) + len(zero_cor_old)) * 28 * 28 * 3).reshape((len(one_cor_old) + len(zero_cor_old)), 3, 28,28)
    one_cor = one_cor_old + patch_size
    zero_cor = zero_cor_old + patch_size

    for i in range(len(one_cor)):
        patch_all[i, 0, :, :] = scan[one_cor[i][0], one_cor[i][1] - patch_size:one_cor[i][1] + patch_size,
                                one_cor[i][2] - patch_size:one_cor[i][2] + patch_size]
        patch_all[i, 1, :, :] = scan[one_cor[i][0] - patch_size:one_cor[i][0] + patch_size, one_cor[i][1],
                                one_cor[i][2] - patch_size:one_cor[i][2] + patch_size]
        patch_all[i, 2, :, :] = scan[one_cor[i][0] - patch_size:one_cor[i][0] + patch_size,
                                one_cor[i][1] - patch_size:one_cor[i][1] + patch_size, one_cor[i][2]]

    for i in range(len(zero_cor)):
        patch_all[i + len(one_cor), 0, :, :] = scan[zero_cor[i][0],
                                               zero_cor[i][1] - patch_size:zero_cor[i][1] + patch_size,
                                               zero_cor[i][2] - patch_size:zero_cor[i][2] + patch_size]
        patch_all[i + len(one_cor), 1, :, :] = scan[zero_cor[i][0] - patch_size:zero_cor[i][0] + patch_size,
                                               zero_cor[i][1], zero_cor[i][2] - patch_size:zero_cor[i][2] + patch_size]
        patch_all[i + len(one_cor), 2, :, :] = scan[zero_cor[i][0] - patch_size:zero_cor[i][0] + patch_size,
                                               zero_cor[i][1] - patch_size:zero_cor[i][1] + patch_size, zero_cor[i][2]]

    label = np.concatenate((np.ones(len(one_cor)), np.zeros(len(zero_cor))))
    label = to_categorical(label)
    patch_all, label = shuffle(patch_all, label)
    return patch_all,label





text_files = [f for f in os.listdir(train_path_left) if f.endswith('.mat')]
for files in text_files:
    print('%s/%s'%(train_path_left,files))
    with h5py.File('%s/%s'%(train_path_left,files),'r') as file:
        print(list(file.keys()))
        scan = np.array(file['scan'])
        CartFM = np.array(file['CartTM'])
        scan,CartFM = ROI_finding(scan,CartFM)
        CartFM_cor, random_cor = sample(CartFM)
        patch_all,label = patch(scan,random_cor,CartFM_cor,patch_size)
        patch_twenty.extend(patch_all)
        label_twenty.extend(label)

text_files = [f for f in os.listdir(train_path_right) if f.endswith('.mat')]
for files in text_files:
    print('%s/%s'%(train_path_right,files))
    with h5py.File('%s/%s'%(train_path_right,files),'r') as file:
        print(list(file.keys()))
        scan = reverse(np.array(file['scan']))
        CartFM = reverse(np.array(file['CartTM']))
        scan,CartFM = ROI_finding(scan,CartFM)
        CartFM_cor, random_cor = sample(CartFM)
        patch_all,label = patch(scan,random_cor,CartFM_cor,patch_size)
        patch_twenty.extend(patch_all)
        label_twenty.extend(label)

text_files = [f for f in os.listdir(val_path_left) if f.endswith('.mat')]
for files in text_files:
    print('%s/%s'%(val_path_left,files))
    with h5py.File('%s/%s'%(val_path_left,files),'r') as file:
        print(list(file.keys()))
        scan = np.array(file['scan'])
        CartFM = np.array(file['CartTM'])
        scan,CartFM = ROI_finding(scan,CartFM)
        CartFM_cor, random_cor = sample(CartFM)
        patch_all,label = patch(scan,random_cor,CartFM_cor,patch_size)
        patch_val.extend(patch_all)
        label_val.extend(label)

text_files = [f for f in os.listdir(val_path_right) if f.endswith('.mat')]
for files in text_files:
    print('%s/%s'%(val_path_right,files))
    with h5py.File('%s/%s'%(val_path_right,files),'r') as file:
        print(list(file.keys()))
        scan = reverse(np.array(file['scan']))
        CartFM = reverse(np.array(file['CartTM']))
        scan,CartFM = ROI_finding(scan,CartFM)
        CartFM_cor, random_cor = sample(CartFM)
        patch_all,label = patch(scan,random_cor,CartFM_cor,patch_size)
        patch_val.extend(patch_all)
        label_val.extend(label)

text_files = [f for f in os.listdir(test_path_left) if f.endswith('.mat')]
for files in text_files:
    print('%s/%s'%(test_path_left,files))
    with h5py.File('%s/%s'%(test_path_left,files),'r') as file:
        print(list(file.keys()))
        scan = np.array(file['scan'])
        CartFM = np.array(file['CartTM'])
        scan,CartFM = ROI_finding(scan,CartFM)
        CartFM_cor, random_cor = sample(CartFM)
        patch_all,label = patch(scan,random_cor,CartFM_cor,patch_size)
        patch_test.extend(patch_all)
        label_test.extend(label)

text_files = [f for f in os.listdir(test_path_right) if f.endswith('.mat')]
for files in text_files:
    print('%s/%s'%(test_path_right,files))
    with h5py.File('%s/%s'%(test_path_right,files),'r') as file:
        print(list(file.keys()))
        scan = reverse(np.array(file['scan']))
        CartFM = reverse(np.array(file['CartTM']))
        scan,CartFM = ROI_finding(scan,CartFM)
        CartFM_cor, random_cor = sample(CartFM)
        patch_all,label = patch(scan,random_cor,CartFM_cor,patch_size)
        patch_test.extend(patch_all)
        label_test.extend(label)



        
        
print(len(patch_twenty))
np.savez_compressed(results_dir+'train_again.npz', patch=patch_twenty, label=label_twenty)
np.savez_compressed(results_dir+'val_again.npz', patch=patch_val, label=label_val)
np.savez_compressed(results_dir+'test_again.npz', patch=patch_test, label=label_test)
