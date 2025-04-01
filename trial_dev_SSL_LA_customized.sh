#!/bin/sh
#SBATCH --job-name=hemlata_dev
#SBATCH --out=log_hemlata_dev
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:tesla_a100:1

# Print the node name
/bin/hostname

# Print the job ID
echo "Job ID: $SLURM_JOB_ID"

source /home/smg/xuecliu/envs/miniconda3/bin/activate SSL_Spoofing

datadir=ultra_deepfake_data/asvspoof2019/LA
scoredir=ultra_deepfake_scores

mkdir -p $scoredir

# WO short
python3 eval_SSL_LA_customized.py --wav_format flac \
    --database_path $datadir/dev \
    --protocols_path $datadir/dev \
    --eval_output $scoredir/dev_wo_short.txt

# WO
python3 eval_SSL_LA_customized.py \
    --database_path $datadir/mc_p3/dev \
    --protocols_path $datadir/mc_p3/dev \
    --eval_output $scoredir/dev_wo.txt

# SEG4
python3 eval_SSL_LA_customized.py \
    --database_path $datadir/mc_p3/dev/SEG4 \
    --protocols_path $datadir/mc_p3/dev/SEG4 \
    --eval_output $scoredir/dev_seg4.txt
