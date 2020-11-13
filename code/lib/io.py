import re
import gzip
from mimetypes import guess_type
from functools import partial
import os.path
from itertools import groupby
import numpy as np
import pandas as pd
from Bio import SeqIO

from .main import *

# Define path variables
repopath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
datadir = os.path.join(repopath, 'data/')
figuredir = os.path.join(repopath,  'figures/raw/')

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
    if name == 'ufo':
        return datadir + 'ufos/ufo.fasta'
    if name == 'ext':
        return datadir + 'ufos/ext.fasta'
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

def count_kmers_proteome(proteome, k, **kwargs):
    return count_kmers_iterable(fasta_iter(proteome, returnheader=False), k, **kwargs)

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

def load_proteome_as_df_path(path):
    "Return proteome as dataframe given its name"
    headers, seqs = list(zip(*[(h, seq) for h, seq in fasta_iter(path,
        returndescription=True, returnheader=False)]))
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


def load_proteome_as_df(name):
    "Return proteome as dataframe given its name"
    return load_proteome_as_df_path(proteome_path(name))

def load_matrix(path):
    return np.array(pd.read_csv(path, sep=' ', header=None))

human = proteome_path('Human')


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

