import numpy as np
import nibabel as nib
import glob
import pandas as pd
import scipy.io as scio
import ToolBox as tb

def makeniiFeature(datapath,labepath):
    mark = 'ID'
    # Loading Data
    data_files_all = sorted(glob.glob(datapath))
    label_files_all = pd.read_csv(labepath)
    label = label_files_all[mark]

    # Label
    y_label = np.array(label)

    files_data = []
    for i in label:
        temp = str(i)[0:-2]
        print('temp-', temp)
        if int(temp) < 100:
            temp = '0'+temp
        for j in data_files_all:
          num = j.index('/sub')
          print(j[num+1:60])
          if temp in j[num+1:63]:
              print('--j--', j)
              img_data = nib.load(j)
              img_data = img_data.get_data()
              img_data_reshape = tb.upper_tri_indexing(img_data)
              files_data.append(img_data_reshape)
              break

    x_data = np.asarray(files_data)
    print(x_data.shape)
    np.savetxt('./IPCAS_GordonsubcorticalFC.txt', x_data)
def makeFeaturewithsubcortical(datapath,labelpath):

    label_files_all = pd.read_csv(labelpath)
    label = label_files_all['ID']
    files_data = []
    for i in label:
        temp = str(i)[0:-2]

        if int(temp) < 100:
            temp = '0' + temp
        sub = 'sub-'+temp
        print('sub-', sub)
        datapath = datapath+sub+'/'
        print('datapath-', datapath)
        data = scio.loadmat(datapath+'func_gordonsubcortical.mat')['data']
        img_data_reshape = tb.upper_tri_indexing(data)
        files_data.append(img_data_reshape)
    x_data = np.asarray(files_data)
    print(x_data.shape)
    np.savetxt('./IPCAS_GordonsubcorticalFC.txt', x_data)
# datapath = "/Users/fan/Documents/Data/zhouTestData/IPCAS_Shaefer/*.nii"
# labepath = "/Users/fan/Documents/Data/zhouTestData/IPCAS_Shaefer/beh_info.csv"
# def makeniiFeature(datapath, labepath)