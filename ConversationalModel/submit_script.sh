#!/bin/bash
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -l vmem=16gb
#PBS -l walltime=24:00:00
#PBS -q visual
#PBS -N conv0.00001

# Run as "qsub submit_script.sh"

cd $PBS_O_WORKDIR

module load cuda

THEANO_FLAGS="cuda.root=/appl/cuda/6.5/,mode=FAST_RUN,device=gpu,floatX=float32,base_compiledir=$PBS_O_WORKDIR,compiledir_format=$$"

THEANO_FLAGS=device=gpu,floatX=float32 python LasagneEstimator.py 0.00001
