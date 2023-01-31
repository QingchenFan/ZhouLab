import nibabel as nib
import os
import glob
from scipy.io import savemat
import numpy as np
fcfile = glob.glob('/Users/fan/Documents/Data/zhouTestData/IPCAS_Schaefer/*.nii')

newpath = '/Users/fan/Documents/Data/zhouTestData/IPCAS_SchaeferFCmat'

for i in fcfile:
    subname = i[54:61]
    print('subname', subname)
    print('--i--', i)
    if not os.path.exists(newpath+'/'+subname):
       os.mkdir(newpath+'/'+subname)
    data = nib.load(i)
    temp = data.get_data()

    #temp = np.array(temp)
    print('--path--', newpath+'/'+subname+'/')
    savemat(newpath+'/'+subname+'/'+'func_schaefer.mat', {'data':temp})