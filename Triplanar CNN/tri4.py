import h5py
import numpy as np
import os, random
from skimage.measure import label
from skimage import morphology

#### section 4 calculate DSC, Sensitivity and Specificity



random.seed(7)

patch_size = 14

base_path = 'data/'

results_dir = 'new_result/'


train_path_left = base_path + 'train/left'
train_path_right = base_path + 'train/right'
val_path_left = base_path + 'val/left'
val_path_right = base_path + 'val/right'
test_path_left = base_path + 'test/left'
test_path_right = base_path + 'test/right'


pred_image = np.load(results_dir +'predict_2dcnn_3.npy')


def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 1:
            TP += 1
    for i in range(len(y_hat)):
        if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
            FP += 1
    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 0:
            TN += 1
    for i in range(len(y_hat)):
        if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
            FN += 1
    return (TP, FP, TN, FN)


def reverse(scan):
    scan=scan[:,:,::-1]
    return scan





def post(scan,CartFM,pred_all,t):
    pred_cor_all = []
    cart_cor_all = []
    pred_last = []
    TP_all = 0
    TN_all = 0
    FP_all = 0
    FN_all = 0

    for z in range(scan.shape[2]):
        image = scan[:, :, z]
        cart = CartFM[:, :, z]
        pred = pred_all[z]
        pred_rev = np.zeros(image.size)

        for i in range(len(pred)):
            thr = 0.05 * (t + 10)
            pred[i][0] = pred[i][0] / (1 - thr)
            pred[i][1] = pred[i][1] / thr
            pred_rev[i] = np.argmax(pred[i])
        pred_rev = pred_rev.reshape(170, 170)
        label_image = label(pred_rev, connectivity=2)
        pred_rev = morphology.remove_small_objects(label_image, min_size=200, connectivity=2)
        pred_rev = pred_rev.reshape(image.size)
        for i in range(len(pred_rev)):
            if pred_rev[i] > 0:
                pred_rev[i] = 1
        pred_rev = pred_rev.reshape(170, 170)
        pred_last.append(pred_rev)
        pred_cor = np.array(np.where(pred_rev > 0)).T
        pred_cor_all.extend(pred_cor)

        cart_cor = np.array(np.where(cart > 0)).T
        cart_cor_all.extend(cart_cor)

        cart_s = cart.reshape(image.size)
        pred_rev_s = pred_rev.reshape(image.size)
        TP, FP, TN, FN = perf_measure(cart_s, pred_rev_s)

        TP_all = TP_all + TP
        FP_all = FP_all + FP
        TN_all = TN_all + TN
        FN_all = FN_all + FN

    Sen = TP_all / (TP_all + FN_all)
    Spe = TN_all / (TN_all + FP_all)
    Dsc = 2 * TP_all / (2 * TP_all + FP_all + FN_all)

    print(Sen)
    print(Spe)
    print(Dsc)
    return Dsc,Sen,Spe

D=[]
Se=[]
Sp=[]
n=0
text_files = [f for f in os.listdir(val_path_left) if f.endswith('.mat')]
for files in text_files:
    print('%s/%s'%(val_path_left,files))
    with h5py.File('%s/%s'%(val_path_left,files),'r') as file:
        Dsc_all =[]
        Sen_all=[]
        Spe_all=[]
        scan = np.array(file['scan'])
        CartFM = np.array(file['CartTM'])
        print(n)
        for t in range(10):
            Dsc, Sen, Spe = post(scan,CartFM,pred_image[n],t)
            Dsc_all.append(Dsc)
            Sen_all.append(Sen)
            Spe_all.append(Spe)
        D.append(Dsc_all)
        Se.append(Sen_all)
        Sp.append(Spe_all)
        n=n+1

text_files = [f for f in os.listdir(val_path_right) if f.endswith('.mat')]
for files in text_files:
    print('%s/%s'%(val_path_right,files))
    with h5py.File('%s/%s'%(val_path_right,files),'r') as file:
        Dsc_all =[]
        Sen_all=[]
        Spe_all=[]


        scan = reverse(np.array(file['scan']))
        CartFM = reverse(np.array(file['CartTM']))
        print(n)
        for t in range(1):
            Dsc, Sen, Spe = post(scan,CartFM,pred_image[n],t)
            Dsc_all.append(Dsc)
            Sen_all.append(Sen)
            Spe_all.append(Spe)
        D.append(Dsc_all)
        Se.append(Sen_all)
        Sp.append(Spe_all)
        n=n+1

print(D)
print(Se)
print(Sp)
np.save(results_dir + 'DSC_trp3.npy', D)
np.save(results_dir + 'Sen_trp3.npy', Se)
np.save(results_dir + 'Spe_trp3.npy', Sp)
