---
layout: page
title: Notes
order: 3
---

# Work 21/08/10

A consequence of the modeled distributions is that viral peptides are more likely to coincide with self-peptides or to be very similar to them. I worked on a better illustration of this phenomenon and on an assessment of whether these overlaps are further increased empirically due to mimicry not captured in the marginal statistics on which the models are based. To do so I calculate the distance to the nearest self peptide for randomly chosen peptides from the human virome. I compared the distribution of these distances with those for peptides drawn from different models. Interestingly, the full model largely reproduces the observed increases in (near-)coincidences.

I also reworked the analysis of how the likelihood of positive T cell responses depends on distance to the nearest self. I found that different assays differ markedly in the average fraction of positive responses. For example, tetramer assays are always almost positive in IEDB. To make things more comparable I focused the analysis only on those assays within IEDB that used the ELISPOT assay (which is the most common). This significantly lowers the fraction of positive assays overall, but most specifically it lowers the value at distance zero to close to zero and thus more in line with theoretical expectations.


# Skype w/ Ben G 21/05/05

Take home messages
- method development: ensemble statistics for protein primary sequences
- ensemble statistics predict that recognized antigens are closer to self than random peptides
- higher order statistics make coincidences and near-coincidences much more likely between host and pathogen peptides
- recast the role of positive selection as active training beyond HLA restriction
- prediction of immunodominance at ~ 10 epitopes 

# Paper from Arup's group

Gao et al. Learning from HIV-1 to predict the immunogenicity of T cell epitopes in SARS-COV-2
https://github.com/andy90/immunogenicity_predictor

Use approach inspired by Luksza et al. to predict immunogenicity of T cell epitopes.

# Software package for probabilistic graphical models

Jerome Tubiana has a general package for building various kinds of probabilistic graphical models (code documentation quality varies, but abstractions seem useful):
https://github.com/jertubiana/PGM

# New immunogenicity prediction paper - Gfeller et al. Cell Reports Medicine 2021

Curated dataset with immunogenicity results from various papers:
https://www.cell.com/cms/10.1016/j.xcrm.2021.100194/attachment/48c7a879-c9e3-42fc-9a47-53971d5a5ebf/mmc2

# Reading on statistics

A few references:
- Cover, Thomas: Chapter 11
- https://en.wikipedia.org/wiki/Error_exponents_in_hypothesis_testing
- https://en.wikipedia.org/wiki/Neyman%E2%80%93Pearson_lemma


# Progress on open questions Jan 28

- Calculate coincidence probability from models -- it is interesting that small changes in entropy mask big differences in coincidence probability
-> while the entropy of the distributions decreases only little there are order of magnitude differences in the coincidence probability due to the tail of more probable kmers

- The DKL analysis shows there is little generic discriminability. How well one can discriminate when not looking at the average peptide but at the subset of peptides that is most discriminable?
-> here again the higher order models make a larger difference, because they change the tails

Both results together provide an illustration of what is gained by considering models going beyond the independent site description


# Call with Bill December 22

Further questions to be explored:
- The DKL analysis shows there is little generic discriminability. Is there a subset of peptides which allows discrimination?
- Calculate coincidence probability from models -- it is interesting that small changes in entropy mask big differences in coincidence probability
- What is the distribution of holes? Distribution of substitutions of sequences distance n away?
- What does this imply for immune strategies? very sophisticated: adjust receptor 'receptive fields' to be broader in regions that are less dense. More realistically: optimize distribution given a constraint on discrimination power (bounded change in response of T cell for a single amino acid change)
- plot viruses on top of distance plot


# Notes on bioinformatics

To normalize taxon ids to the species level we can download taxonomy information from ncbi and then manipulate resulting json file to find species taxon id.
"efetch -db taxonomy -id "11697"  -format native -mode xml -json"

We can cluster proteome data similarly to the uniref 90 clustering in the following manner. See https://github.com/soedinglab/MMseqs2
`mmseqs easy-cluster ../UP000005640Human.fasta.gz UP000005640Human tmp --min-seq-id 0.9 -c 0.8 --cov-mode 0`
cov-mode 0: bidirectional overlap (relative to max of query and target sequence)
c 0.8: at least 80% overlap (documentation mentions that this is a good threshold to perserve multi domain structure of proteins)

# Take home messages (skype with Ben) October 29

- non-self closer to self than what you might think
- little room for a generic classifier, so you have to overfit
- need about 10 peptides for collective decision making
- training on self as a way to focus on generally important regions

# Immunogenicity paper (Wells et al. Cell 2020)

Assayed which parameters determine immunogenicity of tumor neoantigens. 

# Skype w/ Ben

How many peptides would you need to make a reliable distinction between self and non-self? Relation to the order 10 peptides that are typically immunodominant?

# Constructing datasets?

NCBI taxonomy: https://www.ncbi.nlm.nih.gov/taxonomy

Woolhouse et al. Human viruses: discovery and emergence, Phil. Trans. R. Soc. B (2012): Propose a list of 219 virus species known to infect humans 

Follow up dataset: https://www.nature.com/articles/sdata201817#Sec8 list of 214 known human RNA viruses

https://talk.ictvonline.org/ - International Committee on Taxonomy of Viruses: Provide taxonomic classification of viruses on multiple levels

Virgin et al. Redefining Chronic Viral Infection Cell 2009: list of viruses causing chronic viral infections in humans

Virgin The Virome in Mammalian Physiology and Disease Cell 2014: argues for the importance of viral interactions with host physiology

PATRIC: BACTERIAL BIOINFORMATICS RESOURCE CENTER - https://patricbrc.org/ - unified set of bacterial genomes and annotations

Ecohealth alliance list:
https://zenodo.org/record/807517

New NCBI virus portal:
https://www.ncbi.nlm.nih.gov/labs/virus/vssi/#/

# Mapping between MHC types, length preference and T cell types

Here is a small table for future reference

| MHC type | T cell | Antigen origin           | Peptide length | HLA types        |
| -------- | ------ | ------------------------ | -------------- | ---------------- |
| I        | CD8    | intracellular (viruses)  | 9              | HLA-A/B/C/E/F/G/ | 
| II       | CD4    | extracellular (bacteria) | 15             | HLA-D(M/O/P/Q/R) |


#  Old literature on evolution of adaptive immune recognition

Percus, Percus, Perelson PNAS 1993:

"Evolution has shaped the repertoires of immunoglobulin and TCR genes found in vertebrates. Here we suppose that frequently encountered pathogens provide an evolutionary driving force for recognition of epitopes that are common to large classes of microorganisms, for example, polysaccharides found in bacterial cell walls. Evidence that at least some of these antigens are treated specifically by the immune system is found in the fact that polysaccharides stimulate B cells in the absence of T-cell help. At the same time, some epitopes may be expected to be found on self-molecules and on foreign molecules. Although deletion or inactivation of self-reactive clones is known to occur during B-cell and T-cell development, it still seems reasonable that over evolutionary time genes coding for the recognition of common self- antigens would tend to be eliminated and mechanisms would be discovered to minimize interactions with common self- antigens. Thus, we suggest that repertoires are shaped to recognize common features of pathogens that are not simul- taneously present on self-molecules. Work by Chalufour et al. (6) and Claverie et al. (7), in which the sequences ofknown epitopes were compared with sequences of self-proteins, indicates that epitopes are sequences that are unusually "rare" in known proteins. Similar results have been obtained by Ohno (8). Since only a small fraction ofproteins have been sequenced so far, this result is still prelimninary but is supportive of our general hypothesis."


# 2020-07-20: Maxent code on the web

A couple of python maxent codes on the web:

- https://github.com/debbiemarkslab/EVcouplings/
- https://github.com/KIT-MBS/pydca
- https://github.com/eltrompetero/coniii/

Matlab plm code:
- https://github.com/magnusekeberg/plmDCA

# 2020-07-10: Literature reading

Weidmann ... Lipas: Where Natural Protein Sequences Stand out From Randomness. Compare protein sequences to simple models based on the distribution of pairwise distances and find that local compositional biases can account for most of the observed structure. Unclear quantification though in terms of residuals of the distance distribution.

# 2020-07-08: Skype w/ Ben

A number of thoughts that we have talked about before but that we might want to pick up again in more detail:

- Error model for how sloppy viral replication would generate new peptides. 
- Virus of ~10000 kmers, immune system seems to base its decision on a subset of maybe 10 peptides (immunodominance). What is the immune system picking?
- Extreme value distribution for outliers?

# 2020-05-12: Email to Ivan M and Ben G

I finally found some time to further look into the statistics of the Ufos. I am now using the updated list of ufos that Ben send and have also performed the same analysis for the extension peptides.

As regards the points we've discussed last time I have done the following:

- I checked that I get similar statistics for peptides when excluding the start region of the ufos (first 10 amino acids, red line in the attached likelihood plot).

- As I mentioned last time the primary determinant of the likelihood distributions are the amino acid usage statistics. As expected from the difference in the shape of the distribution the amino acid frequencies within the ufos are only weekly correlated with the amino acid frequencies within human peptides (see aafreqsUfo.png in the attachment). The correlations are much weaker than for many other sources of peptides, including for peptides drawn at random with equal nucleotide usage. This raises a potentially interesting question of how nucleotide statistics within UTRs differ to cause these differences in amino acids seen within the UFOs.

I have also taken a look at the amino acid diversity within ufos as a function of distance from the 5' end. Interestingly it seems that there is a restricted diversity of the amino acids seen close to the 5' end beyond what is seen in human proteins. This seems to suggest the existence of a mechanism during either the cap snatching or expression that has some sequence specificity beyond simply the requirement of having an AUG start codon.

# 2020-03-26: Discussion with Ivan Marazzi / Ben G

- Only 2-3 amino acids from human, than ~10-15 aa from UTR, than canonical or overprinted (different reading frame) protein
- Order 1% of all peptides made from UFOs. 
- Do UTRs from other viruses that do not produce UFOs also show weird statistics? 
- What about statistics of pseudogenes? These are no longer functional so should no longer have the same constraints in terms of evolution. But some of them are still sometimes expressed. Are they ignored as self?
- Different statistics as a function of position along the UFO.

# 2020-03-23: Discussion with Ben

I asked elodie if there were any recent dna cross species viruses in human and she said its rarer (with all sampling caveats) and this was the only one she could recall: https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?lvl=0&id=743300

Look at overall differences in statistics between dna viruses and rna viruses. It seems like dna viruses (CMV, HSV1, HCMV) are closer to human statistics than RNA viruses (Flu A, Flu B, SARS-Cov-2). SARS has about 3-fold to 10-fold lower error rate than flu.

# 2020-03-23: SARS-CoV 2 protein sequence

Download fasta from https://www.ncbi.nlm.nih.gov/protein?term=txid2697049[All%20Fields]%20AND%20refseq[filter]&format=fasta

Could programmatically download using eutils:
https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=protein&term=txid2697049+AND+refseq[filter]
https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=protein&id=1820616061&rettype=fasta&retmode=text

# 2019-12-18: Sum up and next steps

We are trying to answer the question of how protein statistics might have shaped the organization of adaptive immune defenses. To do so we are characterizing the statistical structure of small peptides drawn from host or pathogen proteomes. Based on our analyses so far there seems to be a more limited amount of statistical structure for short peptides than one might have naively thought (1). Furthermore a large fraction of this structure is universal, i.e. shared between pathogens and their hosts. This seems to indicate that the immune system cannot as a matter of principle distinguish self and non-self in a purely statistical manner (2). Instead the solution it employs is to "overfit", i.e. it learns to ignore the sparse set of self peptides. The primary evolutionary pressure shaping the immune repertoire then is to cover the universally more dense regions of peptide space more, and any preference for regions enriched in non-self peptides might be secondary. We have found some evidence that indeed the immune system does recognize peptides that are close to self at least as likely (if not slightly at higher rates) than peptides that are far from any self peptides (3).

(1) Which further checks should we do to convince ourselves that we capture most of the relevant statistics?

Three-point correlations? Maybe as a more stringent test we could also look at how well our maxent models reproduce statistics on the distribution of pairwise hamming distances?

(2) This statement is true on average and for a single peptide. We can relax both assumptions and ask if still it remains true.

Are there any strong outliers that distinguish pathogens from their host? (While the average peptide might look similar, there might be one universal kmer that is shared across a lot of pathogens but not found in host.) Could the immune system discriminate statistically by using collective decision making and averaging over multiple peptides? Both are possibilities, which we might want to explore and/or rule out.

(3) Can we find further evidence for or against this idea?

In particular, we might want to take another look at cancer neoantigen immunogenicity as analyzed previously in Martha's paper. What other predictions could we test?


# 2019-10-14: Adding up mutual information?

For a while now I have been wondering how mutual information adds up. In general simply adding it up does not work, but intuitively it seems like this might work in the limit of weak correlations. It seems that this is indeed the case in Ising models, see Schneidman et al. Nature 2006: "For weak correlations, we can solve the Ising model in perturbation theory to show that the multi-information IN is the sum of mutual information terms between all pairs of cells."

TODO: refresh intuition by reading up on perturbation theory in Ising models

Talked with Bill: this is indeed what you would expect. Has previously done this calculation for Ising models. Intuitively this should also hold for Potts models.

A quick calculation: Scaling of summed mutual information with length: $I_0 \int_1^N x^{-1/2} \sim 2 I_0 N^{1/2}$ with $I_0 approx 0.015$ we get 0.3 bit. When do we get as much information from context as from one-point frequencies (~0.15 bit)? -> at N=25. At N=9 (peptide on MHC length) we only get a reduction of 0.09 bit, i.e. correlations at this length are less important than the one point frequencies.

# 2019-10-7 Meeting with Quentin
(heavily skewed towards Q's thinking)
## About pathogens epitopes being close to self
- can it be explained by the need of proliferation signals given by the self in order to keep a clone alive until infection? Only TCRs recognizing at least mildy self can remain alive
- if epitopes need to be close to self to elicit an immune response, is there a self concentration/ neighborhood avoidance effect? Practically: if we look at the number of times a kmer is found in the human/mice genome is it more than what is expected by chance using the peptidome model ("concentration" effect)? Conversely if we look at the number of kmers in the human/mice genome with distance 1 from a self kmer, are they underrepresented compared to what would be expected from peptidome statistics?

## About differences in peptidome statistics
- are physical constraints on peptidome solely contained in 1-point aa frequencies?
- are cancer mutation stirring peptidome towards uniform aa distribution (such as frameshifts, less for SNPs) immunogenic?
- link Arup's quorum sensing idea with peptidome (they actually do a simulation with a bacterian genome in the paper)

## About T cell regulation
- T effectors and T regs use different MHC class, thus interacting T reg and T effectors recognise different kmers while "targeting" the same pathogen. Could this be the mechanism for activation specificity? (joint probability of encountering 2 kmers close to self but not self)


## Miscellaneous
- where do auto-immune peptides lie in the peptide space (PCA image, + look at Misha's ALICE paper)
- Role of memory repertoire: T cells for specific epitopes are selected, this goes "against" the quorum sensing idea with numerous not so specific T cells eliciting a response
- What's the fraction of activated T cells "by error" (meaning without infection) during basal life?
- how many epitopes are needed for a T cell vaccine? (like Flu)
- Kinetic proofreading at the cellular scale? (with a hierarchy of T cell types)
- could it be that in the end all the hardest work of activation decision is mostly done by the innate immune system?
- about tissue specific peptides: I remember a talk from people of Adrien Six's group talking about repertoire sequencing in different organs     

# 2019-10-06: similar sounding Shugay abstract

See [here](http://mccmb.belozersky.msu.ru/2019/thesis/MCCMB2019/abstracts/179.pdf)

# 2019-09-26: Protein evolution angle: How do weak correlations add up?

TODO: Correlated population state despite weak interactions in proteins? To start think about mutual information decay from a Markov chain model?

# 2019-09-17: Meeting with Bill

Now that we know what the relevant problem is (distinguish a peptide from a set of learned self-peptides drawn both from approximately the same distribution) we should revisit the question of how one might construct a system to achieve such discrimination.

Maybe things become easier as dimensionality becomes large. Pointers for how to proceed could be gleaned from Shannon's argument about achieving channel capacity with random codes.

# 2019-09-10: References about immunopeptidomics (thymic presentation)

Unpublished paper Klatzmann group who looked at whether presentation of peptides in the thymus is biased towards viral-like peptides. Underlying papers

- Derbinski, J., Schulte, A., Kyewski, B. & Klein, L. Promiscuous gene
expression in medullary thymic epithelial cells mirrors the peripheral
self. Nature Immunology 2, 1032–1039 (2001).

- Espinosa, G. et al. Peptides presented by HLA class I molecules in
the human thymus. Journal of Proteomics 94, 23–36 (2013).

I think the first ref is about that fact that the 'thymopeptidome' is
unique because thymic APCs express AIRE which controls what proteins
are expressed (to make sure that protein expressed only some specific
tissues are also expressed in the thymus). The second ref contains the
dataset itself from what I understand.


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
