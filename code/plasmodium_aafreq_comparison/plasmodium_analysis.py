import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from Bio.SeqIO import parse
from Bio.Alphabet.IUPAC import protein
plt.style.use('../peptidome.mplstyle')

import sys
sys.path.append('..')
from lib import *

PLASMODIUM_REGEX = re.compile(r".*((Plasmodium.*)|(Malaria).*)\.fasta")


def get_plasmodium_paths():
    """
    Return a list of file paths to fasta files which are of the genus Plasmodium
    """
    paths = []
    for file in (os.listdir("../../data")):
        match = PLASMODIUM_REGEX.match(file)
        if match:
            paths.append((file, match.groups(1)[0]))
    return paths


def get_plasmodium_data(plasmodium_paths:list):
    """
    param: plasmodium_paths -> list of string paths to plasmodium fasta files
    """
    data = {p[1]:[0 for l in protein.letters] for p in plasmodium_paths}
    cols = [l for l in protein.letters]

    for i, path in enumerate(plasmodium_paths):
        with open(os.path.join("../../data", plasmodium_paths[i][0])) as fasta_file:
            for row in parse(fasta_file, "fasta"):
                for letter in protein.letters:
                    data[path[1]][cols.index(letter)] += row.seq.count(letter)
    df = pd.DataFrame.from_dict(data, orient="index", columns=cols)
    df["total"] = df.sum(axis=1)
    df = df.div(df["total"], axis=0)
    df = df.drop("total", axis=1)
    return df


def get_plasmodium_distributions(plasmodium_df:pd.DataFrame):
    """
    return mean, standard deviation of amino acid frequencies across the 
    genus Plasmodium
    """
    plasmodium_df_t = plasmodium_df.transpose()

    std = plasmodium_df_t.std(axis=1)
    means = plasmodium_df_t.mean(axis=1)
    diff_from_mean = np.abs(plasmodium_df_t.subtract(means, axis=0))
    diff_from_mean.rename(columns=lambda x: x+"_diff_from_mean", inplace=True)

    print(diff_from_mean.max(axis=1))
    data_summary = diff_from_mean
    data_summary["std"] = std
    data_summary["mean"] = means

    return data_summary


def plot_aa_freqs(plasmodium_df:pd.DataFrame):
    plasmodium_df_t = plasmodium_df.transpose()

    fig, ax = plt.subplots()
    plasmodium_df_t.plot(kind="bar", ax=ax)
    ax.tick_params(axis='both', which='major', labelsize=4)
    plt.xlabel('Amino Acid', fontsize=6)
    plt.ylabel('Frequency', fontsize=6)
    fig.tight_layout()

    plt.legend(loc=1, prop={'size': 4})
    plt.savefig("plasmodium_comparison.png")
    plt.show()

    
        
if __name__ == "__main__":
    df = get_plasmodium_data(get_plasmodium_paths())
    plot_aa_freqs(df)