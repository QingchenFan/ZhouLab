
import scipy.io as scio
import glob
import pandas as pd
import numpy as np

label_files_all = pd.read_csv('/Users/fan/Documents/Data/zhouTestData/beh_info.csv')
label = label_files_all['ID']
path = '/Users/fan/Documents/Data/zhouTestData/IPCAS_Gordonsubcortical_gradients_aligned/'
files_data = []
for i in label:
    temp = str(i)[0:-2]
    print('temp', temp)
    if int(temp) < 100:
        temp = '0' + temp
    print(path+'sub-'+temp+'/aligned_gradient.mat')
    data = scio.loadmat(path+'sub-'+temp+'/aligned_gradient.mat')
    temp = data['aligned_gradient']
    box = temp[:, 0:1]
    box = box.T
    res = np.array(box).flatten()
    files_data.append(res)
x_data = np.asarray(files_data)
print(x_data.shape)
np.savetxt('./Gordonsubcortical_gradients1.txt', x_data)