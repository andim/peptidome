# How different are self and nonself? 

Can we understand self/nonself discrimination as a statistical classification problem?

This was the question we started pondering in the summer of 2018. After long years of analysis and discussion, we finally arrived at some conceptual clarity and to the conclusion that self/nonself discrimination operates in a different regime.

We are sharing our full set of analyses here following the open science ethos. Note though that this code relates to research in progress. This means:
- It contains analyses not currently part of our manuscript, but that informed our thinking.
- It is likely to not be user-friendly.  

Feedback very welcome!

As in our previous work (see e.g. [Gaimann et al](https://github.com/andim/paper-tcellimprint)) we have stand-alone code located at [our paper's repo](https://github.com/jonlevi/self_not_self), allowing direct reproduction of figures from the manuscript. 

All rights reserved until publication of the corresponding manuscript.


## Structure of this repo
### code
This contains the majority of the code used to explore the topics at hand. Though there are many subdirectories each with slightly different purposes, there are a few main ones to focus on
1) maxent - this directory contains the code for inferring and the notebooks for analyzing the maximum entropy peptidome models. 
2) nndist - this directory contains the code for running distance distribution analyses, including distance to nearest self-peptide for pathogen peptides
3) lib - misc. code for modeling, plotting, and handling input data
4) netmhc - code for running and analyzing predictions made by netMHC for peptide presentation
5) maxent/data - contains a lot of the simulated data used in the paper. 

### data
contains some of the input data used in the paper. Also see [dropbox](https://www.dropbox.com/scl/fo/h63nuakv9r1nkop3g1vnk/AG2xJv8Sh1pkQ7zj7lfMKNo?rlkey=nucxeb7h7yh72yjhjpbm33lnd&e=3&st=w2ssb90r&dl=0)
