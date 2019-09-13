import itertools, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import sys
sys.path.append('..')
from lib import *
import clib
from base import *

output = True
aas_arr = np.array(list(aminoacids))
N = 4
for name, row in proteomes.iterrows():
    if name == 'Human':
    #if not os.path.exists('data/%s.csv' % name):
        print(name)
        seed = 1234
        prng = np.random.RandomState(seed)

        proteome = datadir + row['path']
        seqs = [s for s in fasta_iter(proteome, returnheader=False)]
        #train, test = train_test_split(seqs, test_size=0.5)
        train, test = seqs, seqs

        # evaluate empirical observables for fitting
        #df0 = count(train, 1)
        df0 = pseudocount_f1(train)
        #df1 = count(train, 2, gap=0)
        df1 = pseudocount_f2(train, 2, 0, df0) 
        #dfgap1 = count(train, 2, gap=1)
        dfgap1 = pseudocount_f2(train, 2, 1, df0) 
        #dfgap2 = count(train, 2, gap=2)
        dfgap2 = pseudocount_f2(train, 2, 2, df0) 

        print('fit')
        h, Jk = fit_ising(df0, [df1, dfgap1, dfgap2], nmcmc=1e5, niter=20, epsilon=0.1, prng=prng, output=output, N=N)

        print('compare on 4mers')
        k = 4
        df4 = count(train, k)

        #df4_count = count(test, k)
        kmers = [''.join(s) for s in itertools.product(aminoacids, repeat=k)]
        df4_test = pd.DataFrame.from_dict(dict(seq=kmers, count=np.ones(len(kmers))))
        df4_test.set_index('seq', inplace=True)
        df4_count = counter_to_df(count_kmers_iterable(test, k), norm=False)
        df4_count.set_index('seq', inplace=True)
        df4_test = df4_test.add(df4_count, fill_value=0.0)
        df4_test['freq'] = df4_test['count'] / np.sum(df4_test['count'])

        m, jsd_test = calc_logfold(df4, df4_test)
        jsd_flat = calc_jsd(df4_test['freq'], np.ones_like(df4_test['freq']))

        tripletparams = calc_tripletmodelparams(proteome)
        kmers = df4_test.index
        df4_test['freq_ind'] = np.array([10**(loglikelihood_independent(s, **tripletparams)) for s in kmers])
        df4_test['freq_mc'] = np.array([10**(loglikelihood_mc(s, **tripletparams)) for s in kmers])
        df4_test['freq_tri'] = np.array([10**(loglikelihood_triplet(s, **tripletparams)) for s in kmers])
        jsd_ind = calc_jsd(df4_test['freq'], df4_test['freq_ind'])
        jsd_mc = calc_jsd(df4_test['freq'], df4_test['freq_mc'])
        jsd_tri = calc_jsd(df4_test['freq'], df4_test['freq_tri'])

        q = len(aminoacids)
        Z = np.exp(scipy.special.logsumexp([-clib.energy(np.array(s), h, Jk) for s in itertools.product(range(q), repeat=k)]))
        df4_test['freq_maxent'] = np.exp([-clib.energy(map_aatonumber(s), h, Jk) for s in kmers])/Z
        jsd_maxent = calc_jsd(df4_test['freq'], df4_test['freq_maxent'])
        #nmcmc = 1e7
        #prng = np.random
        #def jump(x):
        #    return prng.randint(q, size=N)
        #def energy(x):
        #    return clib.energy(x, h, Jk)
        #x0 = jump(None)
        #samples = mcmcsampler(x0, energy, jump, nmcmc)
        #samples = [''.join(aas_arr[s]) for s in samples]
        #df4_model = count(samples, 4)
        #m, jsd_model = calc_logfold(df4_test, df4_model)
        print('4mer', 'test', jsd_test, 'maxent', jsd_maxent,
              'flat', jsd_flat, 'ind', jsd_ind, 'mc', jsd_mc, 'tri', jsd_tri)

        df4_test.to_csv('data/%s.csv' % name)

        print(h, Jk)
        save(name, h, Jk)
