#BSUB -W 12:00
#BSUB -n 1
#BSUB -e /home/levinej4/tmp/bsub_logs/distance_to_self_%J_%I.err
#BSUB -o /home/levinej4/tmp/bsub_logs/distance_to_self_%J_%I.out
#BSUB -R "rusage[mem=32] span[hosts=1]"
#BSUB -q cpuqueue
#BSUB -J "[11-12]"

j=$LSB_JOBINDEX
key=$(cat distance_to_self.txt | awk -v ln=$j "NR==ln")


python distance_to_self.py $key



