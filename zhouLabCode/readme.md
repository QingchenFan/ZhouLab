**Step_1_IPCAS2BIDS**

nii2bids.py 将nii文件生成标准的BIDS格式

Nii2mat.py 将nii文件转为.mat保存



**Step_2_Imageprocessing**

fmriprep.sh fmriprep预处理脚本

fmriprep_star.sh 使用fmriprep批量处理被试的脚本

xcpd_postprocess.sh xcp-d数据后处理脚本

xcpd_star.sh 使用xcpd批量处理被试的脚本



**Step_3_getfeature**

 Gradient文件夹存放计算功能连接梯度代码

featureselection.py  特征选择代码

gordonsubcorticalFC.py 计算带有subcortical的功能连接

makefeaturematrixFC.py 将所有被试构成一个大的功能矩阵作为特征数据

makefeaturematrixGradient.py 将所有被试的功能连接梯度构成一个大的矩阵作为特征数据

ToolBox.py 封装的一个工具箱代码



**Step_4_Behaviorindexprediction**

step1_PLSr_IPCAS.py 计算模型的主程序

fc_kfold_svr_prediction.py SVR模型代码

fc_kfold_plsr_prediction_bagging.py 基于bagging的PLS-R模型代码

fc_kfold_plsr_prediction.py PLS-R模型代码



