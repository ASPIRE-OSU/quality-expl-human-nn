#!/bin/bash
# include statement to activate conda environment here
# e.g., source activate quality-expl-human-nn

BINS=(0 1 2 3 4 5 31 40 41)
SCALES=(0.25 0.5 10 50 100 1000)
FILES=(sid5513-noisy-COSINE-7551-F_array1_75.wav \
       sid5566-clean-COSINE-8112-F_close_126.wav \
       sid5571-noisy-COSINE-8112-F_array1_130.wav \
       sid5575-noisy-COSINE-8112-F_array1_135.wav \
       sid5604-anchor-COSINE-8112-F_close_166.wav \
       sid5610-anchor-COSINE-8112-F_close_171.wav \
       sid5615-anchor-COSINE-8112-F_close_180.wav \
       sid5783-clean-COSINE-9472-F_close_44.wav \
       sid5802-noisy-COSINE-9987-F_shoulder_105.wav \
       sid5812-noisy-COSINE-9987-F_shoulder_115.wav \
       sid5891-noisy-COSINE-9987-F_shoulder_191.wav \
       sid5907-clean-COSINE-9987-F_close_206.wav \
       sid5977-noisy-COSINE-9987-F_shoulder_76.wav \
       sid5993-noisy-COSINE-9987-F_shoulder_91.wav)

for b in "${BINS[@]}"; do
    printf "${b}: "
    for s in "${SCALES[@]}"; do
        for f in "${FILES[@]}"; do
            python3 perturb_audio.py -f ${f} -b ${b} -s ${s}
            printf "-"
        done
    done
    printf "\n"
done