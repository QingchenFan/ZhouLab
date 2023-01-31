import nibabel as nib
import numpy as np
import scipy.io
import os

for i in range(10, 168):
    if i<100:
        sub = 'sub-0'+str(i)
    else:
        sub = 'sub-' + str(i)
    path_g = '/home/cuizaixu_lab/fanqingchen/DATA/data/BIDS_IPCAS_xcpd/derivatives/xcp_abcd/'+sub+'/func/'+sub+'_task-rest_space-fsLR_atlas-Gordon_den-91k_den-91k_bold.ptseries.nii'
    path_s = '/home/cuizaixu_lab/fanqingchen/DATA/data/BIDS_IPCAS_xcpd/derivatives/xcp_abcd/'+sub+'/func/'+sub+'_task-rest_space-fsLR_atlas-subcortical_den-91k_den-91k_bold.ptseries.nii'
    savepath = '/home/cuizaixu_lab/fanqingchen/DATA/data/IPCAS_GordonSubFC/'+sub
    if not os.path.exists(path_g) or not os.path.exists(path_s):
        print('--not os.path.exists--', path_g)
        print('--not os.path.exists--', path_s)

        continue

    if not os.path.exists(savepath):
       os.mkdir(savepath)

    gordondata = nib.load(path_g)
    print('--path_g--', path_g)
    gordondata = gordondata.get_fdata()
    gordondata = np.array(gordondata)

    subdata = nib.load(path_s)
    print('--path_g--', path_s)
    subdata = subdata.get_fdata()
    subdata = np.array(subdata)

    resdata = np.append(gordondata, subdata, axis=1)

    resFC = np.corrcoef(resdata, rowvar=False)
    print(resFC, resFC.shape)

    scipy.io.savemat(savepath+'/'+'func_gordonsubcortical.mat', mdict={'data': resFC})