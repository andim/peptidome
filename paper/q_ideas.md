# Questions
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

-
# Ideas
- Bremel & Homan use a ridiculous fraction or Ig, would it change a lot to generate them synthetically?
- Construct a MaxEnt model for bacterial genome (cf Alice's thesis, Christophe's temporal RBM for a spatially restricted max ent?)
- a generative model for HLA? (check in large cohorts of patients if HLA set completly random or correlated associations)


# Litterature to look at
## General
- Kardhar & Chakraborty pMHC/TCR recognition using extreme value statistics
- TCRdist
- Marta Lukza cancer stuff


## T cell activation
- Paul françois PNAS 2013
- Altan bonnet 2005 plos bio
- rendall and sontag Royal soc ope science 2017


## Auto immune diseases
- Misha papers on minimal residual diseases
- acute articular rhumatism (after acute streptococcus infection) (check if T cell mediated)
- Guillain Barré syndrom often ~ month after small infection or vaccination (check if T cell mediated)
- Generally look for T cells mediated auto-immune diseases.
