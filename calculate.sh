#!/bin/bash
#SBATCH --nodes=16
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --time=5:00:00

n=10 # Corresponds to w0
m=16 # Number of processes
for (( i=0; i < $n; i++))
do
    echo $i
    mpirun -n $m python3.7 Particles_motion.py $i
done

output_dir=~/OAM_transfer/Single_particle_effects/output

for (( i=0; i < $n; i++))
do
    cat $output_dir/pr1_$i* > $output_dir/pr_$i.txt
    cat $output_dir/pr2_$i* >> $output_dir/pr_$i.txt
    cat $output_dir/pz1_$i* > $output_dir/pz_$i.txt
    cat $output_dir/pz2_$i* >> $output_dir/pz_$i.txt
    cat $output_dir/Lz1_$i* > $output_dir/Lz_$i.txt
    cat $output_dir/Lz2_$i* >> $output_dir/Lz_$i.txt
done

rm -rf $output_dir/pr1* $output_dir/pr2* $output_dir/pz1* $output_dir/pz2* $output_dir/Lz1* $output_dir/Lz2*

for (( i=0; i < $n; i++))
do
    python3.7 average.py $output_dir/pr_$i.txt
    python3.7 average.py $output_dir/pz_$i.txt
    python3.7 average.py $output_dir/Lz_$i.txt
done

touch  $output_dir/pr.txt $output_dir/pz.txt $output_dir/Lz.txt

for (( i=0; i < $n; i++))
do
    cat $output_dir/pr_$i.txt >> $output_dir/pr.txt
    #echo '\n' >> $output_dir/pr.txt
    cat $output_dir/pz_$i.txt >> $output_dir/pz.txt
    #echo '\n' >> $output_dir/pz.txt
    cat $output_dir/Lz_$i.txt >> $output_dir/Lz.txt
    #echo '\n' >> $output_dir/Lz.txt
done

rm -rf  $output_dir/pr_* $output_dir/pz_* $output_dir/Lz_*
