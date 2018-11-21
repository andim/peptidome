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

