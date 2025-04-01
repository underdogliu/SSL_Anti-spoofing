#!/bin/sh
#SBATCH --job-name=hemlata_test
#SBATCH --out=log_hemlata_test
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
    --database_path $datadir/eval \
    --protocols_path $datadir/eval \
    --eval_output $scoredir/wo_short.txt

# WO
python3 eval_SSL_LA_customized.py \
    --database_path $datadir/mc_p3/eval \
    --protocols_path $datadir/mc_p3/eval \
    --eval_output $scoredir/wo.txt

# SEG4
python3 eval_SSL_LA_customized.py \
    --database_path $datadir/mc_p3/eval/SEG4 \
    --protocols_path $datadir/mc_p3/eval/SEG4 \
    --eval_output $scoredir/seg4.txt

# partialspoof
python3 eval_SSL_LA_customized.py \
    --database_path ultra_deepfake_data/partialspoof \
    --protocols_path ultra_deepfake_data/partialspoof \
    --eval_output $scoredir/partialspoof.txt

# partialspoof
python3 eval_SSL_LA_customized.py \
    --database_path ultra_deepfake_data/partialspoof/SEG \
    --protocols_path ultra_deepfake_data/partialspoof/SEG \
    --eval_output $scoredir/partialspoof_seg0_16.txt
