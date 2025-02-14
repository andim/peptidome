#BSUB -W 12:00
#BSUB -n 1
#BSUB -e /home/levinej4/tmp/distance_pairs_%J_%I.err
#BSUB -o /home/levinej4/tmp/distance_pairs_%J_%I.out
#BSUB -R "rusage[mem=32] span[hosts=1]"
#BSUB -q cpuqueue
#BSUB -J "[1-1]"

python pairwise_hamming_distance.py



