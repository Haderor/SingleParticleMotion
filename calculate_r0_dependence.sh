#!/bin/bash
#SBATCH --nodes=16
#SBATCH --tasks-per-node=1
#SBATCH --time=5:00:00

nb_r0=20 # number of r0
m=16    # Number of processes
output_dir=/home/bearlune/OAM_transfer/Single_particle_effects/output_r0
for (( i=0; i < $nb_r0; i++))
do
    echo $i
    mpirun -n $m python3.7 Particles_motion.py $i $nb_r0 0 1 $output_dir
done

for (( i=0; i < $nb_r0; i++))
do
    cat $output_dir/pr1_$i* > $output_dir/pr_$i.txt
    cat $output_dir/pr2_$i* >> $output_dir/pr_$i.txt
    cat $output_dir/pz1_$i* > $output_dir/pz_$i.txt
    cat $output_dir/pz2_$i* >> $output_dir/pz_$i.txt
    cat $output_dir/Lz1_$i* > $output_dir/Lz_$i.txt
    cat $output_dir/Lz2_$i* >> $output_dir/Lz_$i.txt
done

rm -rf $output_dir/pr1* $output_dir/pr2* $output_dir/pz1* $output_dir/pz2* $output_dir/Lz1* $output_dir/Lz2*

for (( i=0; i < $nb_r0; i++))
do
    python3.7 average_file_content.py $output_dir/pr_$i.txt
    python3.7 average_file_content.py $output_dir/pz_$i.txt
    python3.7 average_file_content.py $output_dir/Lz_$i.txt
done

touch  $output_dir/pr.txt $output_dir/pz.txt $output_dir/Lz.txt

for (( i=0; i < $nb_r0; i++))
do
    if [ -s "$output_dir/pr_$i.txt" ]; then
        echo -n "$i    " >> $output_dir/pr.txt
        cat $output_dir/pr_$i.txt >> $output_dir/pr.txt
    fi
    if [ -s "$output_dir/pz_$i.txt" ]; then
        echo -n "$i    " >> $output_dir/pz.txt
        cat $output_dir/pz_$i.txt >> $output_dir/pz.txt
    fi
    if [ -s "$output_dir/Lz_$i.txt" ]; then
        echo -n "$i    " >> $output_dir/Lz.txt
        cat $output_dir/Lz_$i.txt >> $output_dir/Lz.txt
    fi
done

rm -rf  $output_dir/pr_* $output_dir/pz_* $output_dir/Lz_*
