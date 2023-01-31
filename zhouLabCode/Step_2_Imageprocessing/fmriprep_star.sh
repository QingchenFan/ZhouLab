#!/bin/bash
#SBATCH --job-name=fanqingchen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu 2000
#SBATCH -p q_cn
#SBATCH -o /home/cuizaixu_lab/fanqingchen/DATA/Res/fmriprep/other/job.%j.out
#SBATCH -e /home/cuizaixu_lab/fanqingchen/DATA/Res/fmriprep/other/job.%j.error.txt

for((i=10;i<168;i++))
do
   if (($i<100))
   then
        echo 0$i
        sbatch fmri_prep.sh 0$i
   else
        echo $i
        sbatch fmri_prep.sh $i
   fi
done