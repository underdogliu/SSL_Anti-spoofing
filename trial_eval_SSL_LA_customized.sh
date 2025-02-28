#!/bin/sh

datadir=ultra_deepfake_data/asvspoof2019/LA
scoredir=ultra_deepfake_scores

# WO short
python3 eval_SSL_LA_customized.py \
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
