import random, re
import gzip
from mimetypes import guess_type
from functools import partial
from collections import defaultdict
import os.path
from itertools import groupby
import numpy as np
import pandas as pd
import scipy.special
import scipy.stats
import matplotlib.pyplot as plt
from Bio import SeqIO

from . import nsb


aminoacids = 'ACDEFGHIKLMNPQRSTVWY'
aminoacids_set = set(aminoacids)
naminoacids = len(aminoacids)

# Define path variables
repopath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
datadir = os.path.join(repopath, 'data/')
plotsdir = os.path.join(repopath,  'plots/')

def make_path(row):
    "Return path based on a row from the proteome file"
    path = row['proteomeid'] + row['shortname'] + '.fasta'
    path += '.gz' if row['speciesid'] else ''
    return path

def load_proteomes(only_pathogens=False):
    """
    Load metadata of proteomes.
    """
    proteomes = pd.read_csv(datadir + 'proteomes.csv', dtype=str, na_filter=False)
    proteomes['path'] = proteomes.apply(make_path, axis=1)
    proteomes.set_index('shortname', inplace=True)
    if only_pathogens:
        mask = proteomes['type'].isin(['bacterium', 'virus', 'parasite'])
        proteomes = proteomes[mask] 
    return proteomes

def proteome_path(name):
    proteomes = load_proteomes()
    return datadir + proteomes.loc[name]['path']

def fasta_iter(fasta_name, returnheader=True, returndescription=False):
    """
    Given a fasta file return a iterator over tuples of header, complete sequence.
    """
    if returnheader and returndescription:
        raise Exception('one of returnheader/returndescription needs to be False')
    if guess_type(fasta_name)[1] =='gzip':
        _open = partial(gzip.open, mode='rt')
    else:
        _open = open
    with _open(fasta_name) as f:
        fasta_sequences = SeqIO.parse(f, 'fasta')
        for fasta in fasta_sequences:
            if returndescription:
                yield fasta.description, str(fasta.seq)
            elif returnheader:
                yield fasta.id, str(fasta.seq)
            else:
                yield str(fasta.seq)

# Alternative code that does not rely on Biopython
#def fasta_iter(fasta_name, returnheader=True):
#    """
#    Given a fasta file return a iterator over tuples of header, complete sequence.
#    """
#    f = open(fasta_name)
#    faiter = (x[1] for x in groupby(f, lambda line: line[0] == ">"))
#    for header in faiter:
#        # drop the ">"
#        header = next(header)[1:].strip()
#        # join all sequence lines together
#        seq = "".join(s.strip() for s in next(faiter))
#        if returnheader:
#            yield header, seq
#        else:
#            yield seq

def load_proteome_as_df(name):
    "Return proteome as dataframe given its name"
    headers, seqs = list(zip(*[(h, seq) for h, seq in fasta_iter(proteome_path(name),                                                             returndescription=True, returnheader=False)]))
    genes = []
    for h in headers:
        m = re.search('(?<=GN\=)[^\s]+', h)
        if m:
             genes.append(m.group(0))
        else:
             genes.append('')
    accessions = [h.split('|')[1] for h in headers] 
    df = pd.DataFrame(dict(Gene=genes, Accession=accessions, Sequence=seqs))
    return df

def unique_amino_acids(proteome):
    "returns an array of all unique amino acids used within a proteome"
    return np.unique(list(''.join([seq for h, seq in proteome])))


human = proteome_path('Human')


def entropy_nsb(n, base=None):
    """"
    Estimate the entropy of a discrete distribution from counts per category

    n: array of counts 
    base: base in which to measure the entropy (default: nats)
    """
    N = np.sum(n)
    K = len(n)
    nxkx = nsb.make_nxkx(n, K)
    entropy = nsb.S(nxkx, N, K)
    if base:
        entropy /= np.log(base)
    return entropy

def entropy_grassberger(n, base=None):
    """"
    Estimate the entropy of a discrete distribution from counts per category

    n: array of counts 
    base: base in which to measure the entropy (default: nats)
    """
    N = np.sum(n)
    entropy = np.log(N) - np.sum(n*scipy.special.digamma(n+1e-20))/N
    if base:
        entropy /= np.log(base)
    return entropy

def calc_jsd(p, q, base=np.e):
    "Calculate Jensen Shannon Divergence between p and q"
    p, q = np.array(p, copy=True), np.array(q, copy=True)
    p /= np.sum(p)
    q /= np.sum(q)
    m = 0.5*(p + q)
    return 0.5*(scipy.stats.entropy(p, m, base=base)
                + scipy.stats.entropy(q, m, base=base))

def calc_mi(df2_or_seqs, gap=0):
    """Calculate the mutual information between
       residues from a count of pairs of amino acids.
       Uses the Treves-Panzeri correction for finite size
    """
    try:
        df2 = df2_or_seqs
        strcolumn_to_charcolumns(df2, 'seq')
    except:
        df2 = Counter(df2_or_seqs, k=2, gap=gap).to_df(norm=False, clean=True)
        strcolumn_to_charcolumns(df2, 'seq')
    df11 = df2.groupby('aa1').agg(np.sum)['count']
    df11 /= np.sum(df11)
    df12 = df2.groupby('aa2').agg(np.sum)['count']
    df12 /= np.sum(df12)
    df2['theory'] = [float(df11.loc[s[0]] * df12.loc[s[1]]) for s in df2['seq']]
    df2['freq'] = df2['count']/np.sum(df2['count'])
    mi = np.sum(df2['freq']*np.log2(df2['freq']/df2['theory']))
    micorr = mi - (len(aminoacids)-1)**2/(2*np.log(2)*np.sum(df2['count']))
    return micorr

def calc_mi_std(seqs, gap):
    mis = []
    for i in range(30):
        df2 = Counter(random.sample(seqs, int(len(seqs)/2)), k=2, gap=gap).to_df(norm=False, clean=True)
        mis.append(calc_mi(df2))
    return np.mean(mis), np.std(mis, ddof=1)/2**.5

def strcolumn_to_charcolumns(df, column, prefix='aa'):
    """Build columns of chars from a column of strings of fixed length."""
    k = len(df[column][0]) 
    for i in range(1, k+1):
        df[prefix+str(i)] = [s[i-1] for s in df[column]]
    return df

def scrambled(iterable):
    for s in iterable:
        l = list(s)
        random.shuffle(l)
        shuffled = ''.join(l)
        yield shuffled

class Counter(defaultdict):

    def __init__(self, iterable, k, gap=0, **kwargs):
        """
        Counter class

        iterable: sequences or proteome filename
        k: int, kmer length
        gap: int, gap between first and subsequent letters
        """
        super(Counter, self).__init__(int)
        self.k = k
        self.gap = gap
        if isinstance(iterable, str):
            iterable = fasta_iter(iterable, returnheader=False)
        self.count(iterable, **kwargs)

    def count(self, iterable, **kwargs):
        for seq in iterable:
            count_kmers(seq, self.k, gap=self.gap, counter=self, **kwargs)

    def clean(self):
        "keep only kmers composed of standard amino acids"
        keys = list(self.keys())
        for key in keys:
            if not isvalidaa(key):
                del self[key]

    def to_df(self, norm=True, clean=True):
        """Convert a (kmer, count) dict to a pandas DataFrame
        
        clean: only accept counts responding to valid amino acid letters 
        """
        if clean:
            self.clean()
        if norm:
            df = pd.DataFrame(dict(seq=list(self.keys()), freq=normalize(self)))
        else:
            arr = np.array(list(self.values()), dtype=np.float)
            df = pd.DataFrame(dict(seq=list(self.keys()), count=arr))
        df.sort_values('seq', inplace=True)
        return df

def count_kmers_proteome(proteome, k, **kwargs):
    return count_kmers_iterable(fasta_iter(proteome, returnheader=False), k, **kwargs)

def count_kmers_iterable(iterable, k, clean=False, **kwargs):
    """
    Count number of kmers in all strings of an iterable
    """
    counter = defaultdict(int)
    for seq in iterable:
        count_kmers(seq, k, counter=counter, **kwargs)
    if clean:
        counter = {k:counter[k] for k in counter.keys() if isvalidaa(k)}
    return counter

def calc_tripletmodelparams(proteome):
    df = Counter(proteome, 1).to_df(norm=True)
#    df = counter_to_df(count_kmers_proteome(proteome, 1), norm=True)
    df = df.set_index('seq')
    charlogp = np.log10(df['freq']).to_dict()

    #df1 = counter_to_df(count_kmers_proteome(proteome, 2), norm=False)
    df1 = Counter(proteome, 2).to_df(norm=False)
    strcolumn_to_charcolumns(df1, 'seq')
    count = df1.pivot(columns='aa1', index='aa2')['count']
    count /= np.sum(count, axis=0)
    count[count.isna()] = 1e-10
    doubletlogp = np.log10(count).to_dict()

    df2 = Counter(proteome, 3).to_df(norm=False)
    df2['aa12'] = [s[:2] for s in df2['seq']]
    df2['aa3'] = [s[2] for s in df2['seq']]
    count = df2.pivot(columns='aa12', index='aa3')['count']
    count /= np.sum(count, axis=0)
    count[count.isna()] = 1e-10
    tripletlogp = np.log10(count).to_dict()

    modelparams = dict(charlogp=charlogp, doubletlogp=doubletlogp, tripletlogp=tripletlogp)
    return modelparams


def iscontained(string, strings):
    "Is one of the strings contained in string?"
    for s in strings:
        if s in string:
            return True
    return False

try:
    from .clib import count_kmers
except ImportError:
    print('clib not found')
    def count_kmers(string, k, counter=None, gap=0):
        """
        Count occurrence of kmers in a given string.
        """
        if counter is None:
            counter = defaultdict(int)
        for i in range(len(string)-k-gap+1):
            if gap:
                counter[string[i]+string[i+gap+1:i+k+gap]] += 1
            else:
                counter[string[i:i+k]] += 1
        return counter

def plot_sorted(data, ax=None, normalize=True, scalex=1.0, scaley=1.0, **kwargs):
    if ax is None:
        ax = plt.gca()
    sorted_data = np.sort(data)  # Or data.sort(), if data can be modified
    # Cumulative counts:
    if normalize:
        norm = sorted_data.size
    else:
        norm = 1
    #ax.step(sorted_data, np.arange(sorted_data.size)/norm)  # From 0 to the number of data points-1
    ax.set_xscale('log')
    ax.set_yscale('log')
    return ax.step(sorted_data[::-1]*scalex, scaley*np.arange(sorted_data.size)/norm, **kwargs)


def normalize(counter):
    "Given a (kmer, count) dict returns a normalized array of frequencies"
    arr = np.array(list(counter.values()), dtype=np.float)
    arr /= np.sum(arr)
    return arr

def isvalidaa(string):
    "returns true if string is composed only of characters from the standard amino acid alphabet"
    return all(c in aminoacids_set for c in string)

def counter_to_df(counter, norm=True, clean=True):
    """Convert a (kmer, count) dict to a pandas DataFrame
    
    clean: only accept counts responding to valid amino acid letters 
    """
    if clean:
        counter = {k:counter[k] for k in counter.keys() if isvalidaa(k)}
    if norm:
        return pd.DataFrame(dict(seq=list(counter.keys()), freq=normalize(counter)))
    arr = np.array(list(counter.values()), dtype=np.float)
    return pd.DataFrame(dict(seq=list(counter.keys()), count=arr))

def loglikelihood_independent(string, charlogp=None, k=None, **kwargs):
    if k and (len(string) != k):
        return np.nan
    logp = 0.0
    for c in string:
        try:
            logp += charlogp[c]
        except KeyError:
            logp = np.nan
    return logp

def loglikelihood_mc(string, charlogp=None, doubletlogp=None, k=None, **kwargs):
    if k and (len(string) != k):
        return np.nan
    logp = 0.0
    cold = None
    for c in string:
        try:
            if not cold:
                logp += charlogp[c]
            else:
                logp += doubletlogp[cold][c]
        except KeyError:
            logp = np.nan
        cold = c
    return logp

def loglikelihood_triplet(string, charlogp=None, doubletlogp=None, tripletlogp=None, k=None):
    """ Calculate the loglikelihood of a given string given a triplet model.

    charlogp: log probabilities of different characters log P(c)
    doubletlogp: conditional frequency of character given previous character log P(c_i | c_i-1)
    tripletlogp: conditional frequency of character given previous two characters log P(c_i | c_i-1, c_i-2)
    """

    if k and (len(string) != k):
        return np.nan
    logp = 0.0
    cm1, cm2 = None, None
    for c in string:
        try:
            if (not cm1) and (not cm2):
                logp += charlogp[c]
            elif not cm2:
                logp += doubletlogp[cm1][c]
            else:
                logp += tripletlogp[cm2+cm1][c]
        except KeyError:
            logp = np.nan
        cm2 = cm1
        cm1 = c
    return logp

def likelihoods_epitopes(epitopes, likelihood, k):
    epitopes = list(epitopes)
    likelihoods = np.array([likelihood(seq[i:i+k], k) for seq in epitopes for i in range(len(seq)-k+1)])
    weights = np.array([1.0/(len(seq)-k+1) for seq in epitopes for i in range(len(seq)-k+1)])
    return likelihoods, weights

def plot_histograms(valuess, labels, weights=None, nbins=40, ax=None,
                    xmin=None, xmax=None, **kwargs):
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
    for i, (values, label) in enumerate(zip(valuess, labels)):
        if weights:
            counts, bins = np.histogram(values, bins=bins, weights=weights[i])
            counts /= np.sum(weights[i])
        else:
            counts, bins = np.histogram(values, bins=bins)
            counts = counts/len(values)
        ax.plot(0.5*(bins[:-1]+bins[1:]), counts,
                label=label, **kwargs)
    ax.legend()
    return ax

def mcmcsampler(x0, energy, jump, nsteps, nburnin=0, nsample=1, prng=None):
    "Markov chain Monte carlo sampler"
    if prng is None:
        prng = np.random
    nsteps, nburnin, nsample = int(nsteps), int(nburnin), int(nsample)
    x = x0
    Ex = energy(x)
    states = []
    for i in range(nsteps):
        xp = jump(x)
        Exp = energy(xp)
        if prng.rand() < np.exp(-Exp+Ex):
            x = xp
            Ex = Exp
        if (i > nburnin) and (i % nsample == 0):
            states.append(x)
    return np.array(states)

def energy_ising(s, h, Jk):
    "energy of a translation invariant ising model"
    energy = sum(h[c] for c in s)
    for k, J in enumerate(Jk):
        for i in range(len(s)-1-k):
            energy += J[s[i]][s[i+k+1]]
    return -energy


_aatonumber = {c: i for i, c in enumerate(aminoacids)}
_numbertoaa = {i: c for i, c in enumerate(aminoacids)}

def map_aatonumber(seq):
    """
    Map sequence to array of number
    """
    seq = np.array(list(seq))
    return np.vectorize(_aatonumber.__getitem__)(seq)

def map_numbertoaa(seq):
    """
    Map integer to amino acid sequence
    """
    seq = list(seq)
    return np.vectorize(_numbertoaa.__getitem__)(seq)


def aatonumber(char):
    return _aatonumber[char]

# code modified from OpenVax/pepdata project
# see https://github.com/openvax/pepdata/blob/master/pepdata/iedb/tcell.py
def load_iedb_tcellepitopes(
        mhc_class=None,  # 1, 2, or None for neither
        hla=None,
        exclude_hla=None,
        human_only=False,
        positive_only=False,
        peptide_length=None,
        assay_method=None,
        assay_group=None,
        only_standard_amino_acids=True,
        reduced_alphabet=None,  # 20 letter AA strings -> simpler alphabet
        nrows=None):
    """
    Load IEDB T-cell data without aggregating multiple entries for same epitope
    Parameters
    ----------
    mhc_class: {None, 1, 2}
        Restrict to MHC Class I or Class II (or None for neither)
    hla: regex pattern, optional
        Restrict results to specific HLA type used in assay
    exclude_hla: regex pattern, optional
        Exclude certain HLA types
    human_only: bool
        Restrict to human samples (default False)
    positive_only: bool
        Restrict to epitopes with positive Assay Qualitative Measure
    peptide_length: int, optional
        Restrict epitopes to amino acid strings of given length
    assay_method string, optional
        Only collect results with assay methods containing the given string
    assay_group: string, optional
        Only collect results with assay groups containing the given string
    only_standard_amino_acids : bool, optional
        Drop sequences which use non-standard amino acids, anything outside
        the core 20, such as X or U (default = True)
    reduced_alphabet: dictionary, optional
        Remap amino acid letters to some other alphabet
    nrows: int, optional
        Don't load the full IEDB dataset but instead read only the first nrows
    """
    path = datadir + 'iedb-tcell.zip'
    df = pd.read_csv(
            path,
            header=[0, 1],
            skipinitialspace=True,
            nrows=nrows,
            low_memory=False,
            na_values=['unidentified'],
            error_bad_lines=False,
            encoding="latin-1")

    # Sometimes the IEDB seems to put in an extra comma in the
    # header line, which creates an unnamed column of NaNs.
    # To deal with this, drop any columns which are all NaN
    df = df.dropna(axis=1, how="all")

    n = len(df)
    epitope_column_key = ("Epitope", "Description")
    mhc_allele_column_key = ("MHC", "Allele Name")
    assay_group_column_key = ("Assay", "Assay Group")
    assay_method_column_key = ("Assay", "Method/Technique")

    epitopes = df[epitope_column_key].str.upper()

    null_epitope_seq = epitopes.isnull()
    mask = ~null_epitope_seq

    if only_standard_amino_acids:
        # drop the sequence if it contains unknown amino acids
        mask &= epitopes.apply(isvalidaa)

    if human_only:
        organism = df[('Host', 'Name')]
        mask &= organism.str.startswith('Homo sapiens', na=False).astype('bool')

    if positive_only:
        # there are several types of positive assays (Positive, Positive-High etc.)
        # we thus invert from the negative ones
        mask &= ~(df['Assay', 'Qualitative Measure'] == 'Negative')

    # Match known alleles such as "HLA-A*02:01",
    # broader groupings such as "HLA-A2"
    # and unknown alleles of the MHC-1 listed either as
    #  "HLA-Class I,allele undetermined"
    #  or
    #  "Class I,allele undetermined"
    mhc = df[mhc_allele_column_key]

    if mhc_class is not None:
        # since MHC classes can be specified as either strings ("I") or integers
        # standard them to be strings
        if mhc_class == 1:
            mhc_class = "I"
        elif mhc_class == 2:
            mhc_class = "II"
        if mhc_class not in {"I", "II"}:
            raise ValueError("Invalid MHC class: %s" % mhc_class)
        allele_dict = load_alleles_dict()
        mhc_class_mask = [False] * len(df)
        for i, allele_name in enumerate(mhc):
            allele_object = allele_dict.get(allele_name)
            if allele_object and allele_object.mhc_class == mhc_class:
                mhc_class_mask[i] = True
        mask &= np.array(mhc_class_mask)

    if hla:
        mask &= df[mhc_allele_column_key].str.contains(hla, na=False)

    if exclude_hla:
        mask &= ~(df[mhc_allele_column_key].str.contains(exclude_hla, na=False))

    if assay_group:
        mask &= df[assay_group_column_key].str.contains(assay_group)

    if assay_method:
        mask &= df[assay_method_column_key].str.contains(assay_method)

    if peptide_length:
        assert peptide_length > 0
        mask &= df[epitope_column_key].str.len() == peptide_length

    return df[mask]

def load_iedb_bcellepitopes(human_only=False, only_standard_amino_acids=True):
    """
    Load IEDB B-cell data 

    human_only: bool
        Restrict to human samples (default False)
    only_standard_amino_acids : bool, optional
        Drop sequences which use non-standard amino acids, anything outside
        the core 20, such as X or U (default = True)
    """
    path = datadir + 'iedb-bcell.zip'
    df = pd.read_csv(
            path,
            header=[0, 1],
            skipinitialspace=True,
            low_memory=False,
            error_bad_lines=False,
            encoding="latin-1")

    epitope_column_key = 'Epitope', 'Description'

    mask = df['Epitope', 'Object Type'] == 'Linear peptide'

    epitopes = df[epitope_column_key].str.upper()
    null_epitope_seq = epitopes.isnull()
    mask &= ~null_epitope_seq

    if only_standard_amino_acids:
        # drop the sequence if it contains unknown amino acids
        mask &= epitopes.apply(isvalidaa)

    if human_only:
        organism = df[('Host', 'Name')]
        mask &= organism.str.startswith('Homo sapiens', na=False).astype('bool')

    return df[mask]

def falling_factorial(x, n):
    "returns x (x-1) ... (x-n+1)"
    return scipy.special.factorial(x)/scipy.special.factorial(x-n+1)

codon_map = {"UUU":"F", "UUC":"F", "UUA":"L", "UUG":"L",
    "UCU":"S", "UCC":"S", "UCA":"S", "UCG":"S",
    "UAU":"Y", "UAC":"Y", "UAA":"STOP", "UAG":"STOP",
    "UGU":"C", "UGC":"C", "UGA":"STOP", "UGG":"W",
    "CUU":"L", "CUC":"L", "CUA":"L", "CUG":"L",
    "CCU":"P", "CCC":"P", "CCA":"P", "CCG":"P",
    "CAU":"H", "CAC":"H", "CAA":"Q", "CAG":"Q",
    "CGU":"R", "CGC":"R", "CGA":"R", "CGG":"R",
    "AUU":"I", "AUC":"I", "AUA":"I", "AUG":"M",
    "ACU":"T", "ACC":"T", "ACA":"T", "ACG":"T",
    "AAU":"N", "AAC":"N", "AAA":"K", "AAG":"K",
    "AGU":"S", "AGC":"S", "AGA":"R", "AGG":"R",
    "GUU":"V", "GUC":"V", "GUA":"V", "GUG":"V",
    "GCU":"A", "GCC":"A", "GCA":"A", "GCG":"A",
    "GAU":"D", "GAC":"D", "GAA":"E", "GAG":"E",
    "GGU":"G", "GGC":"G", "GGA":"G", "GGG":"G",}
nt_to_ind = {
    'A' : 0,
    'C' : 1,
    'G' : 2,
    'U' : 3
    }
def ntfreq_to_aafreq(ntfreq):
    frequencies = {aa:0 for aa in aminoacids}
    for nts, aa in codon_map.items():
        if not aa == 'STOP':
            frequencies[aa] += np.prod([ntfreq[nt_to_ind[nt]] for nt in nts])
    sum_ = sum(frequencies.values())
    for aa in aminoacids:
        frequencies[aa] /= sum_
    return frequencies

def dict_to_array(dict_):
    "return an array from a dictionary by sorting the keys"
    keys = sorted(dict_.keys())
    return np.array([dict_[key] for key in keys])
