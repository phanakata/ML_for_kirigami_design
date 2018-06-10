#!/bin/ksh
for i in {1..203}
do
 echo $i
 mkdir $i
 cp in.graphene_kirigami $i
 cp lammps_scc.sh $i 
 mv geo.kirigami_d0.73_sAC_$i $i
 cd $i 
 cp geo.kirigami_d0.73_sAC_$i geo.kirigami
 qsub < lammps_scc.sh
 cd ..
done
