---
layout: page
title: Notes
order: 3
---

# 2019-09-10: References to look at 

Unpublished paper Klatzmann group who looked at whether presentation of peptides in the thymus is biased towards viral-like peptides. Underlying papers

"- Derbinski, J., Schulte, A., Kyewski, B. & Klein, L. Promiscuous gene
expression in medullary thymic epithelial cells mirrors the peripheral
self. Nature Immunology 2, 1032–1039 (2001).

- Espinosa, G. et al. Peptides presented by HLA class I molecules in
the human thymus. Journal of Proteomics 94, 23–36 (2013).


I think the first ref is about that fact that the 'thymopeptidome' is
unique because thymic APCs express AIRE which controls what proteins
are expressed (to make sure that protein expressed only some specific
tissues are also expressed in the thymus). The second ref contains the
dataset itself from what I understand.
"

Also reread Misha ... Shugay paper where they reanalyze data from Diego Chowell



# 2019-09-05: Protein tissue distribution database

https://www.proteinatlas.org/ has data on the distribution of human proteins across tissues and cell types. This could be a useful resource to check how much the peptide statistics should vary across tissues. 

# 2019-08-22: Protein structure and family databases

- Pfam: Protein domains found in sequences of a given proteome. See e.g. [Human](http://pfam.xfam.org/proteome/9606#tabview=tab2), interestingly there are for example nearly 8000 occurrences of C2H2 zinc finger domains in line with the signal we have seen inthe mutual information analysis. A CSV file with where domains are located can be downloaded via [FTP](ftp://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/proteomes/) 
- [CathDB](http://www.cathdb.info): protein domains classified (at top level) into mainly alpha, mainly beta, alpha/beta, and few secondary structure

# 2019-07-31: Skype w/ Ben

Other flu strains to look at: H5N1 (bird flu), Flu B (only human), H3N2 (other seasonal flu strain).

Revisit the question of optimal crossreactivity in the light of the biases that we have seen.

Grouping proteins: Look at https://www.ncbi.nlm.nih.gov/geo/. A few groups to considerhousekeeping genes, innate immunity related genes, heat shock proteins, metal ion transport, olfactory receptor family, HLA family, mitochondrial.

# 2019-04-09: Reading on Chargaff's rule

This relates to the comment by Richard Neher about %A = %T and %G = %C in double-stranded DNA organisms. This is known as Chargaff's rule (https://en.wikipedia.org/wiki/Chargaff%27s_rules): 

"The first rule holds that a double-stranded DNA molecule globally has percentage base pair equality: %A = %T and %G = %C. The rigorous validation of the rule constitutes the basis of Watson-Crick pairs in the DNA double helix model.

The second rule holds that both %A = %T and %G = %C are valid for each of the two DNA strands. This describes only a global feature of the base composition in a single DNA strand."

The first rule needs to be true given the base-pairing that we now know about. The second rule is of a statistical nature and indeed as one might expect can be violated.

"In most bacterial genomes (which are generally 80-90% coding) genes are arranged in such a fashion that approximately 50% of the coding sequence lies on either strand. Wacław Szybalski, in the 1960s, showed that in bacteriophage coding sequences purines (A and G) exceed pyrimidines (C and T). This rule has since been confirmed in other organisms and should probably be now termed "Szybalski's rule". While Szybalski's rule generally holds, exceptions are known to exist. The biological basis for Szybalski's rule, like Chargaff's, is not yet known."

There are some ideas that the second rule might hold because of inversions and inverted transpositions (see G Albrecht Buehler, PNAS 2006). 


# 2019-04-01: Skype w/ Ben G

HIV/HCV database: search in the database for different subtypes and choose complete genome. Than download amino acid fasta files (or first reading frame fasta files).

For HIV/SIV compare HIV-1, HIV-2 and SIV for Chimpanzee and Sooty Mangabee

Flu database at https://www.gisaid.org/ -> Sign up for account (done: andim / LangesPasswort)


# Estimating entropy of discrete distributions

Overview of different methods on https://memming.wordpress.com/2014/02/09/a-guide-to-discrete-entropy-estimators/

# 2019-03-23: Interesting lab @ Princeton: Ileana Cristea

See https://scholar.princeton.edu/cristealab

# 2019-03-18: Meeting w/ Bill

Given that the distributions are so close, we can say that to a large extent the immune system cannot "generalize". The immune system instead relies on overfitting: It memorizes the precise nature of the finite number of self peptides to avoid recognizing.

We can get a lower bound on the entropy by looking at coincidence probabilities. We have `p_c = \sum_i p_i^2 = <p_i> = <e^{log p_i}> >= e^{<log p>}`. From there it follows `-log p_c <= - <log p> = S`. How do estimates of the entropy compare to this lower bound and the upper bound $k log 20$?

What sets the optimal level of crossreactivity? Maybe we can use a simple model for the crossreactivity of different receptors to get some insights. Idea: Additive model of interaction energy of position-weight-matrix type with tunable variance of entries. What fraction of receptors is self-reactive? How close is any receptor to being self-reactive?

# 2019-03-09: Meeting w/ Ben G

Look at HIV and flu databases to see how far proteomes of different strains spread out. Is HIV more spread out more because it is a retrovirus? What about other retroviruses?

Clean up distance calculations. Make shell theory argument more precise. Visualize degree of shared vs. private biases in different proteomes.

# 2019-03-05: Emails with Ben Greenbaum

Thanks for these great questions. I am also excited about these results - I did not expect that there would be so much interesting structure in the data. As a general caveat these are all quick and dirty analyses I mostly did over two long flights, so we will need to see how things come out once everything is cleaned up, but the final results will probably be globally similar.

> 1) The attached figure is interesting - for instance the chronic viruses seem closer to human that the acute ones - is that real?

CMV definitely is very close to human and much closer than the acute viruses. The other chronic viruses are close to each other, but it's less clear whether they are significantly closer to human than the acute ones.

> 2) Can you try even more pathogens - for instance I see HCMV but not HSV?

I curated the list of pathogens by looking at the pathogens with the highest number of IEDB peptides, but we can easily add more. I just added HSV1, which does cluster with the other chronic viruses (HCV, EBV). This addition also gives some feel for the stability of the clustering (see attached): now CMV clusters outside of the other chronic viruses.

> 3) Do like things really cluster? I see influenza A and B together - would that work with multiple influenza strains? what about HIV? or HIV2 or SIV?

I would expect that different strains of the same virus will cluster. Given that even similar viruses such as influenza A and B cluster, the strains should be even more similar.

> 4) On slide 17 it creates the impression that viruses are closer to uniform than bacteria - is that an accurate reading of this visualization?

Yes that's an accurate reading. The attached figure shows the distance to uniform along the diagonal, which also shows that on average the viruses are closer to uniform than bacteria. There is a possible technical artifact that might affect this conclusion, which arises from the fact that the viral proteomes are smaller than the bacterial ones. I am using pseudocounts to regularize the inferred distributions, so when data is limiting the distributions will appear closer to uniform. I can try and check how much of an issue this is, but I would expect the general conclusion to be robust.

# 2019-02-14: Meeting w/ Bill Bialek

Big picture question to ask: What would a satisfactory answer look like?

How does discrimination work? Framing: Finiteness of self proteome or distributional differences?

To test how much the finiteness plays a role consider not just where epitopes fall into the distribution but how far they are from the discrete self-peptides

# 2019-02-12: Skype w/ Ben Greenbaum

Group at Penn Oncology also looked at alignment of neoantigens with self/IEDB and also found evidence that this tells you something

Use deduplicated set of proteins for the mutual information calculation, talk with Sasha about how to do this.

Can we learn something about viruses by looking at how they deviate from the different kind of models that we are fitting?

Interaction of the peptidome distribution with population-level HLA distribution? How do things get filtered?


# 2019-02-06: Reading notes on technicalities of maxent fitting

Some interesting technical ideas can be found in a Nonnenmacher et al. Plos Comp Bio paper 2017 entitled "Signatures of criticality arise from random subsampling in simple population models". They are implemented in an open source Matlab package https://github.com/mackelab/CorBinian.


# 2019-01-15: Some of the open questions

- How should one quantify the degree to which constraints on the proteome statistics are universal vs. phyla specific? What should we expect physically?

- Which type of Max ent model is most appropriate? A model with translation invariant pairwise couplings? Constraining the number of cooccurrences of amino acids (which could account for the fact that the proteome distribution is a mixture of distributions for different classes of proteins)?  

- What does it all mean for immunology? Despite the statistical significance of differences between proteomes the effect sizes seem small.

# 2019-01-14: Email to Ben Greenbaum:

Among other things I've now also looked at B cell epitopes and interestingly for Malaria they are also more similar to the human proteome than random peptides.

As viruses have such short proteomes it is hard to get good statistics on a single virus. I've thus started analyzing also more global statistics across the human virome (by downloading all uniprot proteins with host = human). These analyses again show how remarkably universal the constraints on proteins are: again most of the statistics are very similar to the human proteome.

I was thinking that maybe soon it will be the time to switch from this exploratory phase to making up our mind on what we think the key takeaways are.

I am still a bit unsure about how to best think about what these analyses tell us and thus also about how to tell the story eventually. On the one hand the analyses show that proteins are very highly random and that the small deviations from randomness are mostly conserved across species. On the other hand the differences between peptides from different proteomes while small are often statistically significant, including the differences between random peptides from an organism and those in IEDB.

# 2019-01-11: Interesting groups: Bjoern Peters and Alessandro Sette @ La Jolla Institute

Both groups are involved in curating IEDB and they have also worked extensively on various bioinformatics analysis and predictions tools build on top of the database. Bjoern Peters is e.g. one of the authors of netMHC. See https://www.lji.org/faculty-research/labs/peters/ and https://www.lji.org/faculty-research/labs/sette

# 2019-01-11: Notes on data analysis

One can export all viruses with a human host from uniprot by searching for host:9606. The majority of these proteins are many duplicates of proteins of HIV1, which creates various issues with the down stream analysis. I have thus also generated a version of the viral proteome excluding HIV1.

# 2018-11-16: Meeting with Ben Greenbaum

- chop up viral quasispecies data into kmers: Is there any correlation between the degree of conservation vs. how close the kmer is to the closest self-peptide? Databases: Los Alamos HIV, GISAID flu

# 2018-11-14: Interesting group: Michael Hecht @ Princeton chemistry

From http://chemlabs.princeton.edu/hecht/research/ :
"We have developed methods enabling the design and construction of vast combinatorial libraries of novel amino acid sequences. Most of these sequences fold into stable protein structures (see graphic at right), and many of them display biochemical activities. With the availability of millions of novel proteins, one can begin to go beyond “Protein Design” and consider of “Proteome Design.” This enables the exploration of new questions at the interface of chemistry and biology: Are natural proteins special? Or is it possible to produce new sequences that fold and function? (We think that latter is true). Since our collections of designed proteins are expressed from synthetic genes in living cells, we can now construct artificial “genomes” comprising sequences that never before existed in biology, but nonetheless provide functions necessary to sustain the growth of living cells."

# 2018-11-13: Meeting with Warren James

Interested in the project, previous programming experience in Python, --> send him some references/code to get started

# 2018-11-12: Meeting with Bill Bialek 

- Given that departures from independence are small, one could try and use the analytical results from perturbation theory to invert C <-> J relationship. How well does this do?
- The decay of the mutual information with distance is slower than the growth in the number of interactions at that distance -> the overall ensemble is constrained.
- physics language: log p propto energy, density of states = distribution of log p over all possible sequences
- It might be worthwhile to analytically understand the convergence of log p to a normal distribution using the convergence of the cumulants of the distribution.
- Using ideas from large deviation theory one might also obtain analytical results about the tails of the distribution.

# 2018-11-07: Meeting with Ben Greenbaum

- Compare bird flu with human flu in terms of peptide distribution (also need to compare human and bird proteome to see what the self distribution looks like)

- General point: general evolutionary pressure on viruses to stay close to human proteome due to physical constraints. Therefore immunity should not create big holes around self. This contrasts with classic Perelson et al. view of shape space coverage.

# 2018-10-01: Question/Ideas from Quentin

- Small overlap in proteome reported in Bremel & Homan 2015: shouldn't bacteria be selected for ressemblance to escape the immune system?
- Is there a great difference in proteome between ubiquitous vs specialized genes (e.g in human)? Similarly is there a link between protein concentration or number of cells expressing a given protein and its proteome?  How do the two compare with pathogenic proteomes?
- How sensitive to HLA alleles set is the "presented" proteome? (isn't there a software somewhere to predict binding of (peptide,MHC) duet? = ability to present a given peptide). Check HLA subtypes linked with diseases (e.g HLA B27 for diabetes, + others for arthritis, guts auto immune diseases,...) vs GEM affinty for MHC.
- What fraction of the proteome can be presented?
- How many pentamers (or any size mer) are needed  to correctly differentiate between two proteomes? can we compute some channel capacity of the MHC presentation system? PCA? tSNE?
- Can common motifs be explained by physical constraints (secondary structure, charge, etc)?
- is the length of presented peptides optimized to prevent secondary structure formation?
- Are protein portions forming free loops in protein structures more diverse? Statistically different from globular parts? Membrane parts?
- HIV bnAbs reproduce physical properties of HIV's CD4 binding site, can this be mapped to T cells? (Kayla's talk in Arup's lab)
- Bremel & Homan only focus on bacteria, what about viruses? How do their results hold versus very diverse viruses such as HIV/flu synthetically
- *Is the motif preference also found in homologous proteins? (I don't remember what I meant there, maybe linked to something I have read in Bremel and Homan)*
- Do known human SNPs respect the generative distribution of pentamers? Can we detect an evolutionnary pressure linked with MHC presentation (i.e could it be that the adaptive immune system is a constraint on genome evolution)?
- is avoidance of "human like" peptides more or less pronounced for pathogens frequently infecting human vs the ones not infecting?
- is avoidance of "human like" peptides more pronounced for secreted peptides (e.g toxines) since they are more likely to be taken up?
- how does human proteome compare with the ones of others jawed vertebrates?
- mechanism of cellular graft reject? how close too humans proteomes are ?
- *Treg activation via presentation of Ig peptides?? could this be a mechanism for induced auto immune diseases? (again I don't remember where this comes from but most certainly from a comment in Bremel and Homan)*
- link with immunotherapy: can we find tumor mutations that corresponds to very low probability peptides in the generative model and use such peptides to train T cells?


## Ideas
- Bremel & Homan use a ridiculous fraction or Ig, would it change a lot to generate them synthetically?
- Construct a MaxEnt model for bacterial genome (cf Alice's thesis, Christophe's temporal RBM for a spatially restricted max ent?)
- a generative model for HLA? (check in large cohorts of patients if HLA set completly random or correlated associations)


## Litterature to look at
### General
- Kardhar & Chakraborty pMHC/TCR recognition using extreme value statistics
- TCRdist
- Marta Lukza cancer stuff


### T cell activation
- Paul françois PNAS 2013
- Altan bonnet 2005 plos bio
- rendall and sontag Royal soc ope science 2017


### Auto immune diseases
- Misha papers on minimal residual diseases
- acute articular rhumatism (after acute streptococcus infection) (check if T cell mediated)
- Guillain Barré syndrom often ~ month after small infection or vaccination (check if T cell mediated)
- Generally look for T cells mediated auto-immune diseases.
