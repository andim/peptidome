# 2018-11-07: Meeting with Ben Greenbaum

- Compare bird flu with human flu in terms of peptide distribution (also need to compare human and bird proteome to see what the self distribution looks like)

- General point: general evolutionary pressure on viruses to stay close to human proteome due to physical constraints. Therefore immunity should not create big holes around self. This contrasts with classic Perelson et al. view of shape space coverage.

# 2018-11-12: Meeting with Bill Bialek 

- Given that departures from independence are small, one could try and use the analytical results from perturbation theory to invert C <-> J relationship. How well does this do?
- The decay of the mutual information with distance is slower than the growth in the number of interactions at that distance -> the overall ensemble is constrained.
- physics language: log p propto energy, density of states = distribution of log p over all possible sequences
- It might be worthwhile to analytically understand the convergence of log p to a normal distribution using the convergence of the cumulants of the distribution.
- Using ideas from large deviation theory one might also obtain analytical results about the tails of the distribution.

# 2018-11-13: Meeting with Warren James

Interested in the project, previous programming experience in Python, --> send him some references/code to get started

# 2018-11-14: Interesting group: Michael Hecht @ Princeton chemistry

From http://chemlabs.princeton.edu/hecht/research/ :
"We have developed methods enabling the design and construction of vast combinatorial libraries of novel amino acid sequences. Most of these sequences fold into stable protein structures (see graphic at right), and many of them display biochemical activities. With the availability of millions of novel proteins, one can begin to go beyond “Protein Design” and consider of “Proteome Design.” This enables the exploration of new questions at the interface of chemistry and biology: Are natural proteins special? Or is it possible to produce new sequences that fold and function? (We think that latter is true). Since our collections of designed proteins are expressed from synthetic genes in living cells, we can now construct artificial “genomes” comprising sequences that never before existed in biology, but nonetheless provide functions necessary to sustain the growth of living cells."

# 2018-11-16: Meeting with Ben Greenbaum

- chop up viral quasispecies data into kmers: Is there any correlation between the degree of conservation vs. how close the kmer is to the closest self-peptide? Databases: Los Alamos HIV, GISAID flu

# 2019-01-11: Interesting groups: Bjoern Peters and Alessandro Sette @ La Jolla Institute

Both groups are involved in curating IEDB and they have also worked extensively on various bioinformatics analysis and predictions tools build on top of the database. Bjoern Peters is e.g. one of the authors of netMHC. See https://www.lji.org/faculty-research/labs/peters/ and https://www.lji.org/faculty-research/labs/sette

# 2019-01-11: Notes on data analysis

One can export all viruses with a human host from uniprot by searching for host:9606. The majority of these proteins are many duplicates of proteins of HIV1, which creates various issues with the down stream analysis. I have thus also generated a version of the viral proteome excluding HIV1.

# 2019-01-14: Email to Ben Greenbaum:

Among other things I've now also looked at B cell epitopes and interestingly for Malaria they are also more similar to the human proteome than random peptides.

As viruses have such short proteomes it is hard to get good statistics on a single virus. I've thus started analyzing also more global statistics across the human virome (by downloading all uniprot proteins with host = human). These analyses again show how remarkably universal the constraints on proteins are: again most of the statistics are very similar to the human proteome.

I was thinking that maybe soon it will be the time to switch from this exploratory phase to making up our mind on what we think the key takeaways are.

I am still a bit unsure about how to best think about what these analyses tell us and thus also about how to tell the story eventually. On the one hand the analyses show that proteins are very highly random and that the small deviations from randomness are mostly conserved across species. On the other hand the differences between peptides from different proteomes while small are often statistically significant, including the differences between random peptides from an organism and those in IEDB.

# 2019-01-15: Questions

- How should one quantify the degree to which constraints on the proteome statistics are universal vs. phyla specific? What should we expect physically?

- Which type of Max ent model is most appropriate? A model with translation invariant pairwise couplings? Constraining the number of cooccurrences of amino acids (which could account for the fact that the proteome distribution is a mixture of distributions for different classes of proteins)?  

- What does it all mean for immunology? Despite the statistical significance of differences between proteomes the effect sizes seem small.
