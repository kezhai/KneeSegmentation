import h5py
import numpy as np
import os, random
from skimage.measure import label
from skimage import morphology
from data_prep import *

random.seed(7)

patch_size = 14

base_path = 'Data/'

results_dir = 'new_result/'


train_path_left = base_path + 'train/left'
train_path_right = base_path + 'train/right'
val_path_left = base_path + 'val/left'
val_path_right = base_path + 'val/right'
test_path_left = base_path + 'test/left'
test_path_right = base_path + 'test/right'



## 3 fold cross validation
data = np.load(results_dir + '2D11.npz')
#data = np.load(results_dir + '2D12.npz')
#data = np.load(results_dir + '2D13.npz')
##sagittal slice approach
#data = np.load(results_dir + '2D21.npz')
#data = np.load(results_dir + '2D22.npz')
#data = np.load(results_dir + '2D23.npz')

print(index_train)


def dice_coefficient(a, b):
    """function to calculate DSC"""
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



def Dgraph_udim():
    """ROI genetating of sagittal path"""
    fin=[]
    DSC_all=[]
    for z in range(20):

        print(z)
        x_all=data['x']
        y_all=data['y']
        pred_all=data['pred']
        dsc=[]
    #26 67 37
        qq = []
        ww = []
        ee = []
        rr = []

        for i in range(index_train[z][0]):
            x = x_all[z][i]
            y = y_all[z][i]
            pred = pred_all[z][i]
            y = y.reshape(1, 128, 128, 2)
            pred = pred.reshape(1, 128, 128, 2)

            y= np.argmax(y, axis=3)
            pred = np.argmax(pred, axis=3)


            label_image = label(pred[0], connectivity=2)
            pred_rev = morphology.remove_small_objects(label_image, min_size=100, connectivity=2)
            pred_rev = pred_rev.reshape(pred.size)
            for th in range(len(pred_rev)):
                if pred_rev[th] > 0:
                    pred_rev[th] = 1
            pred = pred_rev.reshape(pred.shape)

            y_cor = np.array(np.where(y[0] > 0)).T
            pred_cor = np.array(np.where(pred[0] > 0)).T

#                print(len(y_cor))
#                print(len(pred_cor))

            try:
                    x1=[min(np.array(y_cor.T[0])),max(np.array(y_cor.T[0]))]
                    x2 = [min(np.array(y_cor.T[1])), max(np.array(y_cor.T[1]))]
#                    print(x1,x2)
                    if dice_coefficient(y[0][x1[0]:x1[1], x2[0]:x2[1]], pred[0][x1[0]:x1[1], x2[0]:x2[1]])[0] !=0:
                      print(dice_coefficient(y[0][x1[0]:x1[1], x2[0]:x2[1]], pred[0][x1[0]:x1[1], x2[0]:x2[1]])[0])
                      dsc.append(dice_coefficient(y[0][x1[0]:x1[1], x2[0]:x2[1]], pred[0][x1[0]:x1[1], x2[0]:x2[1]])[0])
            except:
                    continue




            try:
                qq.append(min(np.array(pred_cor.T[0])))
                ww.append(max(np.array(pred_cor.T[0])))
                ee.append(min(np.array(pred_cor.T[1])))
                rr.append(max(np.array(pred_cor.T[1])))
        #        print(min(np.array(pred_cor.T[0])),max(np.array(pred_cor.T[0])),min(np.array(pred_cor.T[1])),max(np.array(pred_cor.T[1])))
            except:
                continue

        try:
            print(min(qq),max(ww),min(ee),max(rr))
            fin.append([min(qq),max(ww),min(ee),max(rr)])
        except:
            continue
        DSC_all.append(dsc)
    print(DSC_all)
    np.save(results_dir+'pred_2d_1dim.npy',fin)
    np.save(results_dir+'dsc_2d_1dim.npy',DSC_all)



def Dgraph_tdim2():
    """ROI generating of triplanar path without post-processing"""
    fin=[]
    DSC_all=[]
    for z in range(20):

        print(z)
        x_all=data['x']
        y_all=data['y']
        pred_all=data['pred']
        dsc=[]
        tem =[]
    #26 67 37
        for j in range(3):
            qq = []
            ww = []
            ee = []
            rr = []

            if j == 0:
                start = 0
                end = index_train[z][j]
            if j == 1:
                start = index_train[z][j-1]
                end = index_train[z][j-1]+index_train[z][j]
            if j == 2:
                start = index_train[z][j-2]+index_train[z][j-1]
                end = index_train[z][j-2]+index_train[z][j-1]+index_train[z][j]

            print(j)
            print(start,end)

            for i in range(start,end):
                x = x_all[z][i]
                y = y_all[z][i]
                pred = pred_all[z][i]
                y = y.reshape(1, 128, 128, 2)
                pred = pred.reshape(1, 128, 128, 2)

                y= np.argmax(y, axis=3)
                pred = np.argmax(pred, axis=3)

                y_cor = np.array(np.where(y[0] > 0)).T
                pred_cor = np.array(np.where(pred[0] > 0)).T

#                print(len(y_cor))
#                print(len(pred_cor))

                try:
                    x1=[min(np.array(y_cor.T[0])),max(np.array(y_cor.T[0]))]
                    x2 = [min(np.array(y_cor.T[1])), max(np.array(y_cor.T[1]))]
#                    print(x1,x2)
                    if dice_coefficient(y[0][x1[0]:x1[1], x2[0]:x2[1]], pred[0][x1[0]:x1[1], x2[0]:x2[1]])[0] !=0:
                      print(dice_coefficient(y[0][x1[0]:x1[1], x2[0]:x2[1]], pred[0][x1[0]:x1[1], x2[0]:x2[1]])[0])
                      dsc.append(dice_coefficient(y[0][x1[0]:x1[1], x2[0]:x2[1]], pred[0][x1[0]:x1[1], x2[0]:x2[1]])[0])
                except:
                    continue


                try:
                    qq.append(min(np.array(pred_cor.T[0])))
                    ww.append(max(np.array(pred_cor.T[0])))
                    ee.append(min(np.array(pred_cor.T[1])))
                    rr.append(max(np.array(pred_cor.T[1])))
            #        print(min(np.array(pred_cor.T[0])),max(np.array(pred_cor.T[0])),min(np.array(pred_cor.T[1])),max(np.array(pred_cor.T[1])))
                except:
                    continue

            try:
                print(min(qq),max(ww),min(ee),max(rr))
                tem.append([min(qq),max(ww),min(ee),max(rr)])
            except:
                continue
        DSC_all.append(dsc)
        fin.append([max(tem[1][0],tem[2][0]),min(tem[1][1],tem[2][1]),max(tem[0][0],tem[2][2]),min(tem[0][1],tem[2][3]),max(tem[0][2],tem[1][2]),min(tem[0][3],tem[1][3])])
    print(fin)
    print(DSC_all)
    np.save(results_dir+'pred_2d_3dim2.npy',fin)
    np.save(results_dir+'dsc_2d_3dim2.npy',DSC_all)


def Dgraph_tdim():
    """ROI generating of triplanar path with post-processing"""
    fin=[]
    DSC_all=[]
    for z in range(20):

        print(z)
        x_all=data['x']
        y_all=data['y']
        pred_all=data['pred']
        dsc=[]
        tem =[]
    #26 67 37
        for j in range(3):
            qq = []
            ww = []
            ee = []
            rr = []

            if j == 0:
                start = 0
                end = index_train[z][j]
            if j == 1:
                start = index_train[z][j-1]
                end = index_train[z][j-1]+index_train[z][j]
            if j == 2:
                start = index_train[z][j-2]+index_train[z][j-1]
                end = index_train[z][j-2]+index_train[z][j-1]+index_train[z][j]

           # print(j)
           # print(start,end)

            for i in range(start,end):
                x = x_all[z][i]
                y = y_all[z][i]
                pred = pred_all[z][i]
                y = y.reshape(1, 128, 128, 2)
                pred = pred.reshape(1, 128, 128, 2)

                y= np.argmax(y, axis=3)
                pred = np.argmax(pred, axis=3)


                label_image = label(pred[0], connectivity=2)
                pred_rev = morphology.remove_small_objects(label_image, min_size=100, connectivity=2)
                pred_rev = pred_rev.reshape(pred.size)
                for th in range(len(pred_rev)):
                    if pred_rev[th] > 0:
                        pred_rev[th] = 1
                pred = pred_rev.reshape(pred.shape)

                y_cor = np.array(np.where(y[0] > 0)).T
                pred_cor = np.array(np.where(pred[0] > 0)).T

                try:
                    x1=[min(np.array(y_cor.T[0])),max(np.array(y_cor.T[0]))]
                    x2 = [min(np.array(y_cor.T[1])), max(np.array(y_cor.T[1]))]
#                    print(x1,x2)
                    if dice_coefficient(y[0][x1[0]:x1[1], x2[0]:x2[1]], pred[0][x1[0]:x1[1], x2[0]:x2[1]])[0] !=0:
                      print(dice_coefficient(y[0][x1[0]:x1[1], x2[0]:x2[1]], pred[0][x1[0]:x1[1], x2[0]:x2[1]])[0])
                      dsc.append(dice_coefficient(y[0][x1[0]:x1[1], x2[0]:x2[1]], pred[0][x1[0]:x1[1], x2[0]:x2[1]])[0])
                except:
                    continue



#                print(len(y_cor))
#                print(len(pred_cor))

                try:
                    qq.append(min(np.array(pred_cor.T[0])))
                    ww.append(max(np.array(pred_cor.T[0])))
                    ee.append(min(np.array(pred_cor.T[1])))
                    rr.append(max(np.array(pred_cor.T[1])))
            #        print(min(np.array(pred_cor.T[0])),max(np.array(pred_cor.T[0])),min(np.array(pred_cor.T[1])),max(np.array(pred_cor.T[1])))
                except:
                    continue

            try:
            #    print(min(qq),max(ww),min(ee),max(rr))
                tem.append([min(qq),max(ww),min(ee),max(rr)])
            except:
                continue
        DSC_all.append(dsc)
#        fin.append([max(tem[1][0],tem[2][0]),min(tem[1][1],tem[2][1]),max(tem[0][0],tem[2][2]),min(tem[0][1],tem[2][3]),max(tem[0][2],tem[1][2]),min(tem[0][3],tem[1][3])])
    print(fin)
    print(DSC_all)
#    np.save(results_dir+'pred_2d_3dim1.npy',fin)
#    np.save(results_dir+'pred_2d_3dim2.npy',fin)
#    np.save(results_dir+'pred_2d_3dim3.npy',fin)
    np.save(results_dir+'dsc_2d_3dim.npy',DSC_all)

Dgraph_tdim2()


