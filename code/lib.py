import random
from itertools import groupby
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

aminoacids = 'ACDEFGHIKLMNPQRSTVWY'

datadir = '/home/amayer/data/proteomes/'
human = datadir+'uniprot-homosapiens-up000005640.fasta'
mouse = datadir+'uniprot-musmusculus-up000000589.fasta'
malaria = datadir+'uniprot-PlasmodiumFalciparum-up000001450.fasta'
influenzaB = datadir+'uniprot-influenzaB-UP000127412.fasta'
cmv = datadir+'uniprot-cmv-UP000008991.fasta'
hcv = datadir+'uniprot-HCV-UP000000518.fasta'
denv = datadir+'uniprot-DENV-UP000002500.fasta'
tuberculosis = datadir+'uniprot-mycobacteriumtuberculosis-UP000001584.fasta'
listeria = datadir+'uniprot-listeriamonocytogenes-UP000000817.fasta'
hiv = datadir+'uniprot-HIV1-UP000002241.fasta'
ebv = datadir+'uniprot-EBV-UP000153037.fasta'
pseudoburk = datadir+'uniprot-burkholderiapseudomallei-UP000000605.fasta'


pathogenfilepaths = [malaria, influenzaB, cmv, hcv, denv, tuberculosis, listeria, hiv, ebv, pseudoburk]
pathogennames = ['Malaria', 'Influenza B', 'CMV', 'HCV', 'Dengue', 'Tuberculosis', 'Listeria', 'HIV', 'Epstein-Barr virus', 'Burkholderia pseudomallei']
pathogens = dict(zip(pathogennames, pathogenfilepaths))


def fasta_iter(fasta_name, returnheader=True):
    """
    Given a fasta file return a iterator over tuples of header, complete sequence.
    """
    f = open(fasta_name)
    faiter = (x[1] for x in groupby(f, lambda line: line[0] == ">"))
    for header in faiter:
        # drop the ">"
        header = next(header)[1:].strip()
        # join all sequence lines together
        seq = "".join(s.strip() for s in next(faiter))
        if returnheader:
            yield header, seq
        else:
            yield seq

def unique_amino_acids(proteome):
    "returns an array of all unique amino acids used within a proteome"
    return np.unique(list(''.join([seq for h, seq in proteome])))

def strcolumn_to_charcolumns(df, column):
    k = len(df[column][0]) 
    for i in range(1, k+1):
        df['aa'+str(i)] = [s[i-1] for s in df[column]]
    return df


def scrambled(iterable):
    for s in iterable:
        l = list(s)
        random.shuffle(l)
        shuffled = ''.join(l)
        yield shuffled

def count_kmers_proteome(proteome, k, **kwargs):
    return count_kmers_iterable(fasta_iter(proteome, returnheader=False), k, **kwargs)

def count_kmers_iterable(iterable, k, **kwargs):
    """
    Count number of kmers in all strings of an iterable
    """
    counter = defaultdict(int)
    for seq in iterable:
        count_kmers(seq, k, counter=counter, **kwargs)
    return counter

def count_kmers(string, k, counter=None, gap=0):
    """
    Count number of kmers in a given string.
    """
    if counter is None:
        counter = defaultdict(int)
    for i in range(len(string)-k-gap+1):
        if gap:
            counter[string[i]+string[i+gap+1:i+k+gap]] += 1
        else:
            counter[string[i:i+k]] += 1
    return counter

def normalize(counter):
    arr = np.array(list(counter.values()), dtype=np.float)
    arr /= np.sum(arr)
    return arr
 
def counter_to_df(counter, norm=True):
    if norm:
        return pd.DataFrame(dict(seq=list(counter.keys()), freq=normalize(counter)))
    arr = np.array(list(counter.values()), dtype=np.float)
    return pd.DataFrame(dict(seq=list(counter.keys()), count=arr))

def loglikelihood_independent(string, charprobdict, k=None):
    if k and (len(string) != k):
        return np.nan
    logp = 0.0
    for c in string:
        try:
            logp += charprobdict[c]
        except KeyError:
            logp = np.nan
    return logp

def loglikelihood_mc(string, charprobdict, doubletprobdict, k=None):
    if k and (len(string) != k):
        return np.nan
    logp = 0.0
    cold = None
    for c in string:
        try:
            if not cold:
                logp += charprobdict[c]
            else:
                logp += doubletprobdict[cold][c]
        except KeyError:
            logp = np.nan
        cold = c
    return logp

def loglikelihood_triplet(string, charprobdict, doubletprobdict, tripletprobdict, k=None):
    if k and (len(string) != k):
        return np.nan
    logp = 0.0
    cm1, cm2 = None, None
    for c in string:
        try:
            if (not cm1) and (not cm2):
                logp += charprobdict[c]
            elif not cm2:
                logp += doubletprobdict[cm1][c]
            else:
                logp += tripletprobdict[cm2+cm1][c]
        except KeyError:
            logp = np.nan
        cm2 = cm1
        cm1 = c
    return logp

def plot_histograms(valuess, labels, nbins=40, ax=None, xmin=None, xmax=None):
    if not ax:
        ax = plt.gca()
    if (xmin is None) or (xmax is None):
        mean = np.mean([np.mean(values) for values in valuess])
        std = np.mean([np.std(values) for values in valuess])
    if xmin is None:
        xmin = round(mean-5*std)
    if xmax is None:
        xmax  = round(mean+5*std)
    bins = np.linspace(xmin, xmax, nbins)
    for values, label in zip(valuess, labels):
        counts, bins = np.histogram(values, bins=bins)
        ax.plot(0.5*(bins[:-1]+bins[1:]), counts/len(values), label=label)
    ax.legend()
    return ax

def mcmcsampler(x0, energy, jump, nsteps, nburnin=0, nsample=1):
    nsteps, nburnin, nsample = int(nsteps), int(nburnin), int(nsample)
    x = x0
    Ex = energy(x)
    states = []
    for i in range(nsteps):
        xp = jump(x)
        Exp = energy(xp)
        if np.random.rand() < np.exp(-Exp+Ex):
            x = xp
            Ex = Exp
        if (i > nburnin) and (i % nsample == 0):
            states.append(''.join(x))
    return np.array(states)

def energy_ising(s, h, Jk):
    "energy of a translation invariant ising model"
    energy = sum(h[c] for c in s)
    for k, J in enumerate(Jk):
        for i in range(len(s)-1-k):
            energy += J[s[i]][s[i+k+1]]
    return -energy
