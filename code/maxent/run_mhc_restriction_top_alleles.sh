#BSUB -W 48:00
#BSUB -n 1
#BSUB -e /home/levinej4/tmp/bsub_logs/%J_%I.err
#BSUB -o /home/levinej4/tmp/bsub_logs/%J_%I.out
#BSUB -R "rusage[mem=64] span[hosts=1]"
#BSUB -q gpuqueue
#BSUB -J "mhcNSF[1-3]"


# mhcfile="mhc_top_alleles_${LSB_JOBINDEX}.txt"

mhcfile="mhc_top_500_haplotypes_alleles_${LSB_JOBINDEX}.txt"

csv_string=$(paste -sd, $mhcfile)

model="Humanviruses_nskewfcov_1e7_60"

python run_mhc.py $model $csv_string

# models=("independent" "ncov" "nskew" "nskewfcov" "train")

# for m in "${models[@]}"
# do
#   echo $m;
#   echo $csv_string;
#   python run_mhc.py $m $csv_string
# done

