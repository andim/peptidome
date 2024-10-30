#BSUB -W 8:00
#BSUB -n 1
#BSUB -e /home/levinej4/tmp/bsub_logs/%J_%I.err
#BSUB -o /home/levinej4/tmp/bsub_logs/%J_%I.out
#BSUB -R "rusage[mem=64] span[hosts=1]"
#BSUB -q cpuqueue
#BSUB -J "mhcVSamp[1-500]"

j=$LSB_JOBINDEX
i=$(cat /data/lareauc/levinej/pep/peptidome/code/maxent/data/top_500_allowed_haplotypes.txt | awk -v ln=$j "NR==ln")
python mhc_sample_restriction.py independent_1e7_60 Humanviruses $i
python mhc_sample_restriction.py ncov_1e7_60 Humanviruses $i
python mhc_sample_restriction.py nskew_1e7_60 Humanviruses $i
python mhc_sample_restriction.py nskewfcov_1e7_60 Humanviruses $i
# python mhc_uniform.py $i