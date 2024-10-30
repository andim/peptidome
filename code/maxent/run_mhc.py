# helper function to run part of coincidence_prob.ipynb asynchronously
# data per model must already by in data/net_mhc_input_{model}.txt

import sys
import os

repopath = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

sys.path.append(f"{repopath}/code/lib")
from netmhcrunutils import run_netMHC


# models = ["independent", "ncov", "nskew", "nskewfcov", "train"]

model = sys.argv[1]
hla_genes = sys.argv[2].split(",")

for hla in hla_genes:

    # print(f'filtering for MHC-pan binders for model "{model}" for allele {hla}')
    # run_netMHC_pan(
    #     f"{repopath}/code/maxent/data/net_mhc_input_{model}.txt",
    #     f"{repopath}/code/maxent/data/net_mhc_output_{model}",
    #     hla,
    #     binder_only=True,
    #     overwrite=True,
    # )

    # hla = hla.replace(":", "")

    print(f'filtering for MHC binders for model "{model}" for allele {hla}')

    # run_netMHC(
    #     f"{repopath}/code/maxent/data/netMHC_input/net_mhc_input_{model}_2x.txt",
    #     f"{repopath}/code/maxent/data/netMHC_output/net_mhc_output_{model}_2x",
    #     hla,
    #     binder_only=True,
    #     overwrite=True,
    # )

    # run_netMHC(
    #     f"{repopath}/code/maxent/data/netMHC_input/net_mhc_input_{model}.txt",
    #     f"{repopath}/code/maxent/data/netMHC_output/net_mhc_output_{model}",
    #     hla,
    #     binder_only=True,
    #     overwrite=True,
    # )

    # run_netMHC(
    #     f"{repopath}/code/maxent/data/netMHC_input/net_mhc_input_Humanviruses_{model}.txt",
    #     f"{repopath}/code/maxent/data/netMHC_output/net_mhc_output_Humanviruses_{model}",
    #     hla,
    #     binder_only=True,
    #     overwrite=True,
    # )
    # run_netMHC(
    #     f"{repopath}/code/maxent/data/netMHC_input/net_mhc_input_Malaria_{model}.txt",
    #     f"{repopath}/code/maxent/data/netMHC_output/net_mhc_output_Malaria_{model}",
    #     hla,
    #     binder_only=True,
    #     overwrite=True,
    # )

    # run_netMHC(
    #     f"{repopath}/code/maxent/data/netMHC_input/net_mhc_input_thermo_{model}.txt",
    #     f"{repopath}/code/maxent/data/netMHC_output/net_mhc_output_thermo_{model}",
    #     hla,
    #     binder_only=True,
    #     overwrite=True,
    # )

    # run_netMHC(
    #     f"{repopath}/code/maxent/data/netMHC_input/net_mhc_input_thermo_virus_{model}.txt",
    #     f"{repopath}/code/maxent/data/netMHC_output/net_mhc_output_thermo_virus_{model}",
    #     hla,
    #     binder_only=True,
    #     overwrite=True,
    # )

    # run_netMHC(
    #     f"{repopath}/code/maxent/data/netMHC_input/net_mhc_input_thermo_{model}.txt",
    #     f"{repopath}/code/maxent/data/netMHC_output/full_output/net_mhc_output_{model}",
    #     hla,
    #     binder_only=True,
    #     overwrite=True,
    # )
    # run_netMHC(
    #     f"{repopath}/code/maxent/data/netMHC_input/net_mhc_input_thermo_virus_{model}.txt",
    #     f"{repopath}/code/maxent/data/netMHC_output/full_output/net_mhc_output_thermo_virus_{model}",
    #     hla,
    #     binder_only=False,
    #     overwrite=True,
    # )

    run_netMHC(
        f"{repopath}/code/maxent/data/netMHC_input/net_mhc_input_{model}.txt",
        f"{repopath}/code/maxent/data/netMHC_output/net_mhc_output_{model}",
        hla,
        binder_only=True,
        overwrite=False,
    )
    # run_netMHC(
    #     f"{repopath}/code/maxent/data/netMHC_input/net_mhc_input_{model}.txt",
    #     f"{repopath}/code/maxent/data/netMHC_output/full_output/net_mhc_output_{model}",
    #     hla,
    #     binder_only=False,
    #     overwrite=True,
    # )
