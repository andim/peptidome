# How does neighbor density relate to likelihoods ?

How does the likelihood of a sequence relate to the probability that any of its neighboring sequences (here defined as differing by a single amino acid) relate to each other?

It turns out that for an independent site model (to which the amino acid distribution is close to empirically) the total probability of all neighboring sites is completely determined by the likelihood of the sequence itself as long as the variance is small. We can even get an analytical formula for the scaling, which turns out to be somewhat sublinear (with an exponent of approximately ~ 1-1/k). In the figure below this is shown for an independent site model for 9mers based on the amino acid frequencies of the human proteome.

Practically this finding helps us relate our findings in terms of likelihoods to more traditional bioinformatic approaches using nearest neighbor densities.
