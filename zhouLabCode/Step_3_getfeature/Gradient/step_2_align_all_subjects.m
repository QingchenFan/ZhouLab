clear;clc;
%导入功能连接矩阵的node的label文件
labeling = load('/Users/fan/Documents/Data/zhouTestData/Schaefer_label.mat');
root_dir = '/Users/fan/Documents/Data/zhouTestData/IPCAS_Schaefer_gradients';
out_path = '/Users/fan/Documents/Data/zhouTestData/IPCAS_Schaefer_gradients_aligned';
if ~exist(out_path,'dir')
    mkdir(out_path);
end
data_all = dir([root_dir filesep 'sub-*']);
% 汇总所有被试的gradient map到一个cell文件里,不指定reference
n_sub= numel(data_all);
for sub = 1:n_sub
   sub_func = load([root_dir filesep data_all(sub).name '/gredient.mat']); 
   all_sub_gradients{sub} = sub_func.gm.gradients{1};
   sub_func.gm.aligned
   sub
end
%使用procrustes方法对两个被试的梯度map进行对齐，使用100次迭代，https://www.biorxiv.org/content/10.1101/2020.10.24.352153v2
[aligned, xfms] = procrustes_alignment(all_sub_gradients,'nIterations', 100);%返回每个被试对齐后的gradient map：aligned和转换矩阵xfms. 
for sub = 1:n_sub
   aligned_gradient = aligned{sub};
   aligned_gradient_all(:,:,sub)=aligned_gradient;
   out_name = [out_path filesep data_all(sub).name];
   if ~exist(out_name,'dir')
       mkdir(out_name)
   end
   save([out_name,'/aligned_gradient.mat'],'aligned_gradient');
   sub
end

%% 检查对齐
%导入功能连接矩阵的node的label文件
labeling = load('/Users/fan/Documents/Data/zhouTestData/Schaefer_label.mat');
labeling = labeling.schaeferdata; 
%导入surface的vertex信息用于图示化
[surf_lh, surf_rh] = load_conte69();
plot_hemispheres([aligned{10}(:,1),aligned{2}(:,1)], ...
    {surf_lh,surf_rh}, 'parcellation', labeling, ...
    'labeltext',{'aligned sub 10','aligned sub 2'});

