#!/bin/bash
#SBATCH --account PAS2301
#SBATCH --job-name SCOREQ
#SBATCH --mem=51200mb
#SBATCH --time=18:00:00
#SBATCH --mail-type=FAIL,END

# stop if any errors
set -e

# load python environment
source activate scoreq
cd /fs/ess/PAS2301/Data/Student_Data/abarach/quality/omnixai/scoreq

time python3 predict_eval.py