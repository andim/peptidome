import sys
import os

repopath = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
# sys.path.append(f"{repopath}/code/lib")
sys.path.append(f"{repopath}/code/")
from lib import *
from lib.maxent import *

outdirectory = "data/mhc_sampling_results"


haplotype = sys.argv[1]


def compute_pc(arr):
    N = arr.shape[0]
    _, counts = np.unique(arr, return_counts=True, axis=0)
    return np.sum(counts * (counts - 1)) / (N * (N - 1))


def get_mhc_df(allele):
    try:
        return pd.read_csv(
            f"data/netMHC_output/net_mhc_output_uniform_replicate-{allele}.csv"
        )

    except Exception as e:
        print("PD ERROR")
        print(allele)
        return None


def get_combined_binders(alleles):
    kmers = set()
    for allele in alleles:
        mhc_df = get_mhc_df(allele)
        if mhc_df is not None:
            new_set = set(mhc_df["Peptide"])
            kmers = kmers.union(new_set)
    return kmers


print(f"filtering for MHC binders for sample from uniform distribution")

[a1, a2, b1, b2, c1, c2] = haplotype.split(",")
prefix = f"uniform_"
hap_string = "|".join([a1, a2, b1, b2, c1, c2])
outfile = os.path.join(outdirectory, f"{prefix}{hap_string}.json")
k = 9

result = {}


def get_uniform_matrix():
    uniform_kmers = []
    with open("data/netMHC_input/net_mhc_input_uniform_replicate.txt") as f:
        for k in f.readlines():
            uniform_kmers.append(k.strip())
    return kmers_to_matrix(uniform_kmers)


if os.path.exists(outfile):
    print("File exists...skipping")
else:
    binders = get_combined_binders([a1, a2, b1, b2, c1, c2])
    matrix_all = get_uniform_matrix()

    filtered_kmers = [km for km in matrix_to_kmers(matrix_all) if km in binders]
    matrix_mhc = kmers_to_matrix(filtered_kmers)

    result["model"] = "uniform"
    result["haplotype"] = haplotype
    result["N"] = matrix_all.shape[0]
    result["pc"] = compute_pc(matrix_all)

    result["S"] = np.log((20**k))

    result["N_mhc"] = matrix_mhc.shape[0]

    result["b"] = matrix_mhc.shape[0] / matrix_all.shape[0]
    result["pc_mhc"] = compute_pc(matrix_mhc)
    result["S_mhc_raw"] = np.log((20**k))
    result["S_mhc_with_logb"] = result["S_mhc_raw"] + np.log(result["b"])

    with open(outfile, "w") as fp:
        json.dump(result, fp)
