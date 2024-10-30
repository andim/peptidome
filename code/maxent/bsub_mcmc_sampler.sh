#BSUB -W 160:00
#BSUB -n 1
#BSUB -e /home/levinej4/bsub_logs/9_23_thermo_%J_%I.err
#BSUB -o /home/levinej4/bsub_logs/9_23_thermo_%J_%I.out
#BSUB -R rusage[mem=64]
#BSUB -q cpuqueue
#BSUB -J "vThin60[3-4]"

j=$LSB_JOBINDEX
i=$(cat ./mcmc_param_sets.txt | awk -v ln=$j "NR==ln")


# python run_mcmc_thermo.py $i
# python run_mcmc_thermo.py Human_nskewfcov_1e7_100
# python run_mcmc_thermo.py Humanviruses_nskewfcov_1e7_60
# python run_mcmc_thermo.py Humanviruses_nskew_1e7_60
python run_mcmc_thermo.py Humanviruses_ncov_1e7_60