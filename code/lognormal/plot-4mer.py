
#k = 4
#df = counter_to_df(count_kmers_proteome(human, k), norm=False)
#df = df[~df['seq'].str.contains('U|B|X|Z')]
#df['freq'] = df['count'] / np.sum(df['count'])
#
#counts, bins = np.histogram(np.log10(df['freq']), bins=int(2*np.log(len(df['freq']))))
#ax.plot(0.5*(bins[:-1]+bins[1:]), counts/(np.sum(counts)*np.diff(bins)), label='observed')
#x = np.linspace(k*mu-5*(k*sigmasq)**.5, k*mu+5*(k*sigmasq)**.5)
#ax.plot(x, scipy.stats.norm.pdf(x, k*mu, (k*sigmasq)**.5), label='expected')
#ax.set_xlabel('$log_{10}$ frequency');
#ax.legend()
