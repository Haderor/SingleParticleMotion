#!/bin/bash

for (( i=0; i < 10; i++))
do
    echo i=$i
    mpirun -n 16 python Particles_motion.py $i
done

echo finished

