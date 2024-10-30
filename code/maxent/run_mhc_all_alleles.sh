#BSUB -W 48:00
#BSUB -n 1
#BSUB -e /home/levinej4/tmp/bsub_logs/new_sample_mhc_all_alleles_%J_%I.err
#BSUB -o /home/levinej4/tmp/bsub_logs/new_sample_mhc_all_alleles_%J_%I.out
#BSUB -R "rusage[mem=248] span[hosts=1]"
#BSUB -q gpuqueue
#BSUB -J "[1-9]"


mhcfile="/data/greenbaum/users/levinej4/peptidome/code/maxent/data/mhc_alleles_chunk_0${LSB_JOBINDEX}.txt"

csv_string=$(paste -sd, $mhcfile)

model="independent"

python run_mhc.py $model $csv_string

# models=("independent" "ncov" "nskew" "nskewfcov" "train")

# for m in "${models[@]}"
# do
#   echo $m;
#   echo $csv_string;
#   python run_mhc.py $m $csv_string
# done

