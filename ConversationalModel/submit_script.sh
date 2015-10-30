#!/bin/bash
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -l walltime=24:00:00
#PBS -q visual

# Run as "qsub submit_script.sh"

cd $PBS_O_WORKDIR

module load cuda

THEANO_FLAGS="cuda.root=/appl/cuda/6.5/,mode=FAST_RUN,device=gpu,floatX=float32,base_compiledir=$PBS_O_WORKDIR,compiledir_format=$$"

THEANO_FLAGS=device=gpu,floatX=float32 python LasagneEstimator.py
