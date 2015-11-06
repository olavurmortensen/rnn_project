#!/bin/bash
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -l walltime=00:10:00
#PBS -l vmem=8gb
#PBS -q visual

# Run as "qsub submit_script.sh"

cd $PBS_O_WORKDIR

module load cuda

THEANO_FLAGS="cuda.root=/appl/cuda/6.5/,mode=FAST_RUN,device=gpu,floatX=float32,base_compiledir=$PBS_O_WORKDIR,compiledir_format=$$"

# python data_generator.py

THEANO_FLAGS=device=gpu,floatX=float32 python LasagneEstimator.py
