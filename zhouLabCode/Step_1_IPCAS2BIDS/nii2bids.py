from shutil import copy
import os
import glob
import json

def bidspre(funcfile, T1file, funcpath, T1path):
    if not os.path.exists(newpathpre):
        os.mkdir(newpathpre)
    if not os.path.exists(funcpath):
         os.mkdir(funcpath)
         print('--in--')
         for i in funcfile:
                indexnum = i.index('sub')
                funcnum = i[indexnum + 3:indexnum + 6]  # 把每个被试编码取出
                print('funcnum', funcnum)
                if not os.path.exists(funcpath+'/'+'sub-'+funcnum):
                       os.mkdir(funcpath+'/'+'sub-'+funcnum)
                mark = 'task-rest'
                print('--mark--', mark)
                funcnewname = 'sub-' + funcnum
                aimpath = funcpath+'/'+'sub-'+funcnum+'/'
                # double check
                print('--i--', i)
                print('--aim--', aimpath + funcnewname + '_' + mark + '_bold.nii')
                print('---------------------------------------------')
                # Copy data to the bisd path
                copy(i, aimpath + funcnewname + '_' + mark + '_bold.nii')
    if not os.path.exists(T1path):
        os.mkdir(T1path)
        for j in T1file:
            indexnum = j.index('sub')
            anatnum = j[indexnum + 3:indexnum + 6]
            print('--funcnum--', anatnum)
            if not os.path.exists(T1path + '/' + 'sub-' + anatnum):
                os.mkdir(T1path + '/' + 'sub-' + anatnum)
                #mark = 'task-rest'
                #print('--mark--', mark)
                anatnewname = 'sub-' + anatnum
                aimpath = T1path + '/' + 'sub-' + anatnum + '/'
                # double check
                print('--i--', j)
                print('--aim--', aimpath + anatnewname + '_T1w.nii')
                print('---------------------------------------------')
                # Copy data to the bisd path
                copy(j, aimpath + anatnewname + '_T1w.nii')

def copyanat(T1file):
    '''
        copy anat files
    '''
    # TODO return
    if os.path.exists(newpath+'/sub-010/anat/sub-010_T1w.json'):
        print('Already exists anat')
        return
    for i in T1file:
        indexnum = i.index('sub')
        anatnum = i[indexnum+3:indexnum + 6]  # Take the three-digit number of the file name

        anatnewname = 'sub-'+anatnum
        aimpath = newpath + anatnewname + '/' + 'anat/'
        #print(aimpath)
        # double check
        print('--i--', i)
        print('--aim--', aimpath+anatnewname+'_T1w.nii')
        print('---------------------------------------------')
        # Copy data to the bisd path
        copy(i, aimpath+anatnewname+'_T1w.nii')
        jsonStruc = {'SkullStripped': 'false',
                 'Project': 'IPCASTest'}
        box = json.dumps(jsonStruc, indent=1)
        with open(aimpath+anatnewname+'_T1w.json', 'w', newline='\n') as f:
            f.write(box)

def copyfunc(funcfile):
    '''
        copy func files
    '''
    # TODO return
    if os.path.exists(newpath+'sub-011_task-rest_bold.json'):
        print('Already exists func')
        return
    for i in funcfile:
        indexnum = i.index('sub')
        funcnum = i[indexnum+3:indexnum + 6]
        mark = 'task-rest'
        funcnewname = 'sub-'+funcnum
        aimpath = newpath + funcnewname + '/' + 'func/'
        #print(aimpath)
        # double check
        print('--i--', i)
        print('--aim--', aimpath+funcnewname+'_'+mark+'_bold.nii')
        print('---------------------------------------------')
        # Copy data to the bisd path
        copy(i, aimpath+funcnewname+'_'+mark+'_bold.nii')
        jsonStruc = {
	            "RepetitionTime": 2,
	            "SkullStripped": "false",
	            "TaskName": "rest"
                }
        box = json.dumps(jsonStruc, indent=1)
        with open(aimpath+funcnewname+'_'+mark+ '_bold.json', 'w', newline='\n') as f:
            f.write(box)

def mkdirFile(newpath,FunImg_path):
    '''
        make  folder
    '''
    if not os.path.exists(newpath):
      os.mkdir(newpath)
      for i in FunImg_path:
         tem = i[16:19]                            # Take the three-digit number of the file name
         new_name = 'sub-'+tem
         print(i + '--' + new_name)                # Calibration
         if not os.path.exists(newpath+new_name):
             os.mkdir(newpath+new_name)            # Create a file for each subject
             os.mkdir(newpath+new_name+'/'+'anat')
             os.mkdir(newpath+new_name+'/'+'func')

if __name__ == '__main__':
    # source data
    FunImgname = '/Users/fan/Documents/Data/zhouTestData/forTest/IPCASFunImg'
    T1Imagname = '/Users/fan/Documents/Data/zhouTestData/forTest/IPCAST1Img'
    #bisd path
    newpath = '/Users/fan/Documents/Data/zhouTestData/BIDS_IPCAS_test/'
    newpathpre = '/Users/fan/Documents/Data/zhouTestData/BIDS_IPCAS_test/'

    # source path
    funcfile = glob.glob('/Users/fan/Documents/Data/zhouTestData/forTest/IPCASFunImg/*/*')
    T1file = glob.glob('/Users/fan/Documents/Data/zhouTestData/forTest/IPCAST1Img/*/2*')
    # new func/T1 path
    funcpath = '/Users/fan/Documents/Data/zhouTestData/BIDS_IPCAS_Pre/Func'
    T1path = '/Users/fan/Documents/Data/zhouTestData/BIDS_IPCAS_Pre/T1'

    FunImg_path = os.listdir(FunImgname)
    T1Imag_path = os.listdir(T1Imagname)
    #bidspre(funcfile, T1file, funcpath, T1path)
    mkdirFile(newpath, FunImg_path)
    copyanat(T1file)
    copyfunc(funcfile)