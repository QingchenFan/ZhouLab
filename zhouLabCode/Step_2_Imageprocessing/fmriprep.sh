#!/bin/bash
#SBATCH --job-name=fanqingchen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu 8G
#SBATCH -p q_cn
#SBATCH -o /home/cuizaixu_lab/fanqingchen/DATA/Res/fmriprep/job.%j.out
#SBATCH -e /home/cuizaixu_lab/fanqingchen/DATA/Res/fmriprep/job.%j.error.txt
module load singularity/3.7.0


#!/bin/bash
#User inputs:
bids_root_dir=/home/cuizaixu_lab/fanqingchen/DATA/data/BIDS_tep
bids_root_dir_output_wd4singularity=/home/cuizaixu_lab/fanqingchen/DATA/data/BIDS_tep_wk

mkdir $bids_root_dir_output_wd4singularity
subj=$1
nthreads=40

mkdir $bids_root_dir/derivatives
mkdir $bids_root_dir_output_wd4singularity/derivatives
#Run fmriprep
echo ""
echo "Running fmriprep on participant: sub-$subj"
echo ""

#Make fmriprep directory and participant directory in derivatives folder
if [ ! -d $bids_root_dir/derivatives/fmriprep ]; then
    mkdir $bids_root_dir/derivatives/fmriprep
fi

if [ ! -d $bids_root_dir/derivatives/fmriprep/sub-${subj} ]; then
    mkdir $bids_root_dir/derivatives/fmriprep/sub-${subj}
fi
if [ ! -d $bids_root_dir_output_wd4singularity/derivatives/fmriprep ]; then
    mkdir $bids_root_dir_output_wd4singularity/derivatives/fmriprep
fi

if [ ! -d $bids_root_dir_output_wd4singularity/derivatives/fmriprep/sub-${subj} ]; then
    mkdir $bids_root_dir_output_wd4singularity/derivatives/fmriprep/sub-${subj}
fi

#Run fmriprep
export SINGULARITYENV_TEMPLATEFLOW_HOME=/home/cuizaixu_lab/fanqingchen/DATA/templateflow  #将templateflow加载到singularity_fmriprep
#修改python环境
unset PYTHONPATH; singularity run --cleanenv -B $bids_root_dir_output_wd4singularity/derivatives/fmriprep/sub-${subj}:/wd \
    -B $bids_root_dir:/inputbids \
    -B $bids_root_dir/derivatives/fmriprep/sub-${subj}:/output \
    -B /GPFS/cuizaixu_lab_temp/wuguowei/code:/freesurfer_license \
    /usr/nzx-cluster/apps/fmriprep/singularity/fmriprep-20.2.1.simg \  # 镜像文件的命令，将fMRIprep镜像文件导入
    /inputbids /output participant\                                    # /inputbids  ->输入文件夹 /output  —>输出文件夹 participant ->表示一个被试一个被试的跑
    --participant_label ${subj} \                                      # 指定要跑哪一个被试
    -w /wd \                                                           # 挂载临时文件路径，产生的临时文件存放到哪里，有助于检查报错
    --nthreads $nthreads \
    --omp-nthreads $nthreads \
    --mem-mb 160000 \                                                  # 设定资源
    --fs-license-file /freesurfer_license/license.txt \
    --output-spaces T1w MNI152NLin6Asym MNI152NLin2009cAsym  \         #个体空间图像需要往标准空间配准的时候，需要往哪些标准空间去配。如果需要产生皮层文件，就需要T1w、MNI152NLin6Asym
    --return-all-components \
    --notrack --verbose \
    --skip-bids-validation --debug all --stop-on-first-crash --use-syn-sdc --resource-monitor --cifti-output 91k  #--cifti-output 91k 表示输出的文件，需不需要映射到皮层空间，因为这些处理的都是volume空间，小立方块。而这一步就是映射到皮层空间

 #  以下都是资源监控的指令，默认选项
 #   --return-all-components \
 #    --notrack --verbose \
 #    --skip-bids-validation --debug all --stop-on-first-crash --use-syn-sdc --resource-monitor