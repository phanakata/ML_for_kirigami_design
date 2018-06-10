#!/bin/ksh
module load python/3.6.2
rm -rf junki
rm -rf junkstress0
rm -rf junkstress1
for i in {0..203}
do
 echo $i
 echo "$i" >> junki
 cd $i 
 paste output.txt | awk '{if (NR!=1) {print $1, $2*$7/3.7}}' > stress.dat 
 python3.6 ../interpolate.py
 #use interpolation data
 python3.6 ../get-firstMaxima.py > junk0
 python3.6 ../get-firstMaxima-inter.py > junk1
 #python2.6 ../get-maxima.py > junk
 #paste junk | awk -v a="$i" '{print $a, $1, $2}'>>../stress-strain.dat
 paste junk0>>../junkstress0
 paste junk1>>../junkstress1
 cd ..
done
paste junki junkstress0 | awk '{print $1, $2, $3, $4}'> ss.dat
paste junki junkstress1 | awk '{print $1, $2, $3, $4}'> ss-i.dat
