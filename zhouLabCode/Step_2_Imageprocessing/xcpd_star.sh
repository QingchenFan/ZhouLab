#!/bin/bash
#SBATCH --job-name=fanqingchen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu 2000
#SBATCH -p q_cn
#SBATCH -o /home/cuizaixu_lab/fanqingchen/DATA/Res/fmriprep/other/job.%j.out
#SBATCH -e /home/cuizaixu_lab/fanqingchen/DATA/Res/fmriprep/other/job.%j.error.txt

function contains() {
    local n=$#
    local value=${!n}
    for ((i=1;i < $#;i++)) {
        if [ "${!i}" == "${value}" ]; then
            echo "y"
            return 0
        fi
    }
    echo "n"
    return 1
}

A=(20 28 35 38 50 56 61 72 76 78 119 133)

for((i=10;i<168;i++))
do
   if [ $(contains "${A[@]}" $i) == "y" ]; then

        continue
   fi
   if (($i<100))
   then
        echo 0$i
        sbatch xcpd_postprocess.sh 0$i
   else
        echo $i
        sbatch xcpd_postprocess.sh $i
   fi
done