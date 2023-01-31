#!/bin/bash
#SBATCH --job-name=fanqingchen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu 8G
#SBATCH -p q_cn
#SBATCH -o /home/cuizaixu_lab/fanqingchen/DATA/Res/xcpd_cifti/job.%j.out
#SBATCH -e /home/cuizaixu_lab/fanqingchen/DATA/Res/xcpd_cifti/job.%j.error.txt
##### END OF JOB DEFINITION  #####
echo ""
echo "Running fmriprep on participant: sub-$1"
echo ""
module load singularity_xcp_abcd/0.0.4
singularity run -B /home/cuizaixu_lab/fanqingchen/DATA/data/BIDS_IPCAS/derivatives/fmriprep/sub-$1/fmriprep/:/data \
 -B /home/cuizaixu_lab/fanqingchen/DATA/data/BIDS_IPCAS_xcpd/derivatives/:/out \
 -B /home/cuizaixu_lab/fanqingchen/.cache:/home/xcp_abcd/.cache /home/cuizaixu_lab/wuguowei/DATA/aconda_envirment/xcp_abcd.sif \
 /data /out participant -w /out --participant_label $1 --cifti -p 36P --despike --lower-bpf 0.01 --upper-bpf 0.08 --smoothing 6

