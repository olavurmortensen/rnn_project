#!/bin/bash
#PBS -l nodes=1:ppn=8
#PBS -l walltime=24:00:00
#PBS -q visual

# Run as "qsub submit_script.sh"

cd $PBS_O_WORKDIR

python data_generator.py
python word_embeddings.py
