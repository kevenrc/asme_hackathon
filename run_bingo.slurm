#!/bin/bash
#SBATCH --time=10:00:00 # walltime, abbreviated by -t
#SBATCH --nodes=1 # number of cluster nodes, abbreviated by -N
#SBATCH -o test-%j.out-%N # name of the stdout, using the job number (%j) and the first node (%N)
#SBATCH --ntasks=1 # number of MPI tasks, abbreviated by -n # additional information for allocated clusters
#SBATCH --account=hochhalter-np # account - abbreviated by -A
#SBATCH --partition=hochhalter-shared-np # partition, abbreviated by -p # # set data and working directories
#SABTCH --qos=hochhalter-high-np
#SBATCH --gres=gpu:v100:2
#SBATCH --nodelist=notch204
#
# set data and working directories
export WORKDIR=/uufs/chpc.utah.edu/common/home/u0823391/Desktop/asme_hackathon
export SCRDIR=/scratch/kingspeak/serial/u0823391/bingo/asme_hackathon

mkdir -p $SCRDIR
cp -r * $SCRDIR
cd $SCRDIR

module load intel mpich anaconda
source activate bingo

mpirun -np $SLURM_NTASKS python bingo_evolution.py
