#!/bin/sh
#
# Example SGE script for running mpi jobs
#
# Submit job with the command: qsub myscript
#
# Note: A line of the form "#$ qsub_option" is interpreted
#       by qsub as if "qsub_option" was passed to qsub on
#       the commandline.
#
# Set the hard runtime (aka wallclock) limit for this job,
# default is 2 hours. Format: -l h_rt=HH:MM:SS
#
#$ -l h_rt=04:00:00
#
# Merge stderr into the stdout file, to reduce clutter.
#
#$ -j y
#
# Invoke the mpi Parallel Environment for N processors.

#$ -P frgeeeph
#$ -N K 
#$ -pe mpi_8_tasks_per_node 8
#$ -m e
# end of qsub options

export MPI_IMPLEMENTATION=openmpi

#mpirun -np $NSLOTS /usr3/graduate/zenanqi/lammps-8May13/src/lmp_linux < in.lj

mpirun -np $NSLOTS /project/frgeeeph/lammpsMoS2/src/lmp_mpi < in.graphene_kirigami

#You can override the SGE batch resource parameters such as 
#number of processors and walltime limit, pre-defined in myscript, as follows:

#katana:~ % qsub -pe mpi 8 -l h_rt=24:00:00 lammps_katana.sh
#usually, it is submitted as:
#katana:~% qsub lammps_katana.sh
