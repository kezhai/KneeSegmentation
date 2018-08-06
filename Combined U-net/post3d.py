import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from skimage.measure import label
#from skimage import data, util,measure
from skimage import morphology
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data
from matplotlib import cm


results_dir = 'new_result/'

base_path = '/home/hzd551/3dknee/data/'

train_path_left = base_path + 'train/left'
train_path_right = base_path + 'train/right'
val_path_left = base_path + 'val/left'
val_path_right = base_path + 'val/right'
test_path_left = base_path + 'test/left'
test_path_right = base_path + 'test/right'



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

def data_split(file,left=True):
    scan = np.array(file['scan'])
    CartTM = np.array(file['CartTM'])
    CartTM_cor = np.array(np.where(CartTM > 0)).T
#    print(len(CartTM_cor))
    x = int(round(max(np.array(CartTM_cor.T[0]))+ min(np.array(CartTM_cor.T[0])))/2)
    y = int(round(max(np.array(CartTM_cor.T[1]))+ min(np.array(CartTM_cor.T[1])))/2)
    z = int(round(max(np.array(CartTM_cor.T[2]))+ min(np.array(CartTM_cor.T[2])))/2)
    # if left is False:
    #     z = (CartTM.shape[2]) - z
#    print(x,y,z)
    ROI = scan[x-8:x+8, y-32:y+32, z-16:z+16]
    ROI_mask = CartTM[x-8:x+8, y-32:y+32, z-16:z+16]
    return ROI,ROI_mask


def padwithzeros(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = 0
    vector[-pad_width[1]:] = 0
    return vector

def pad(image,x):
    all = np.zeros(170*170*(image.shape[2]+2*x)).reshape(170,170,(image.shape[2]+2*x))
    all[:,:,x:image.shape[2]+x] = image
    return all

def reverse(scan):
    scan=scan[:,:,::-1]
    return scan


def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]==1:
           TP += 1
    for i in range(len(y_hat)):
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]==0:
           TN += 1
    for i in range(len(y_hat)):
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1
    return(TP, FP, TN, FN)



def dice_coefficient(a, b):
    """ use python list comprehension, preferred over list.append() """
    a_bigram_list = a.flatten()
    b_bigram_list = b.flatten()

    # assignments to save function calls
    lena = len(a_bigram_list)
    one_a = 0
    one_b = 0
    # initialize match counters
    matches = i = 0
    while (i < lena):
        if (a_bigram_list[i] ==1):
            one_a += 1
        if (b_bigram_list[i]==1):
            one_b += 1
        if (a_bigram_list[i] == b_bigram_list[i]==1):
            matches += 2
        i += 1

    score = float(matches) / float(one_a + one_b)
    return score,one_a,one_b,matches

def data_generate_3D(path_left, path_right):
    number = 0
    scan = []
    mask = []

    text_files = [f for f in os.listdir(path_left) if f.endswith('.mat')]
    for files in text_files:
        with h5py.File('%s/%s'%(path_left,files),'r') as file:
            scan.append(np.array(file['scan']))
            mask.append(np.array(file['CartTM']))
            number+=1

    text_files = [f for f in os.listdir(path_right) if f.endswith('.mat')]
    for files in text_files:
        with h5py.File('%s/%s'%(path_right,files),'r') as file:
            scan.append(reverse(np.array(file['scan'])))
            mask.append(reverse(np.array(file['CartTM'])))
            number+=1
    return scan,mask


#scan = data_generate_3D(test_path_left,test_path_right)[0]
#mask = data_generate_3D(test_path_left,test_path_right)[1]


scan = data_generate_3D(train_path_left,train_path_right)[0]
mask = data_generate_3D(train_path_left,train_path_right)[1]


#scan = data_generate_3D(val_path_left,val_path_right)[0]
#mask = data_generate_3D(val_path_left,val_path_right)[1]






print(len(scan))


def multi_Tgraph():
    data = np.load(results_dir+'3D2.npz')
    DSC_all=[]
    Sen_all=[]
    Spe_all=[]
    for n in range(10):
        all=scan[n]
        all_mask=mask[n]
        all = np.lib.pad(all, 2, padwithzeros)
        all_mask = np.lib.pad(all_mask, 2, padwithzeros)
        pred=data['pred'][n]
        [x,y,z] = data['ROI'][n]
        print(x,y,z)

        pred = pred.reshape(1,32,80,48,2)
        DSC = []
        SEN = []
        SPE = []
        for i in range(10):
            thr = 0.05*(i+10)
            print(thr)
            pred[:,:,:,:,0]=pred[:,:,:,:,0]/(1-thr)
            pred[:,:,:,:,1]=pred[:,:,:,:,1]/thr
            pred_rev = np.argmax(pred,axis=4)
            print(pred_rev.shape)
            pred_cor=np.array(np.where(pred_rev[0]>0)).T
            print(len(pred_cor))
            real_cor=np.array(np.where(all_mask>0)).T
            print(len(real_cor))




            pred_all = np.zeros(all.size).reshape(all.shape)
            pred_all[x-16:x+16, y-40:y+40, z-24:z+24] = pred_rev[0]

            cart_s=all_mask.reshape(all.size)
            pred_rev_s=pred_all.reshape(all.size)

            TP, FP, TN, FN=perf_measure(cart_s,pred_rev_s)
    #        print(TP, FP, TN, FN)
            Sena=TP/(TP+FN)
            Spea=TN/(TN+FP)
            SEN.append(Sena)
            SPE.append(Spea)
            # print(Sen,Spe)
            # print(2*TP/(2*TP+FP+FN))
            # print((TN+TP)/(TP+TN+FN+FP))
            #print(dice_coefficient(ROI_mask,pred_rev[0]))
            print(dice_coefficient(all_mask,pred_all)[0])

            # Append cartilage volume together
            #DSC.append([len(pred_cor),len(real_cor)])

            # Append DSC together

            DSC.append(dice_coefficient(all_mask,pred_all)[0])
        DSC_all.append(DSC)
        Sen_all.append(SEN)
        Spe_all.append(SPE)

    print(DSC_all)
   # print(Sen_all)
   # print(Spe_all)
   # np.save(results_dir+'DSC_3D3.npy',DSC_all)
   # np.save(results_dir+'Sen_3D3.npy',Sen_all)
   # np.save(results_dir+'Spe_3D3.npy',Spe_all)



multi_Tgraph()
