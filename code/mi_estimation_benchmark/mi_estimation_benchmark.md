# Benchmarking mutual information estimators

A comparison of different ways of estimating mutual information.

Here we compare estimation of mutual information using three different techniques and investigate their bias by comparing subsampling human proteins to 1000 or 10000 samples. The techniques we consider are a direct calculation, a direct calculation with the Treves-Panzeri correction for finite sampling, and a calculation by substracting the entropy of the joint entropy from the sum of the entropies of the single sites (both entropies calculated using the Grassberger estimator). The direct calculation using the the Treves-Panzeri correction does have the lowest bias at moderate sample sizes and should thus be the preferred technique.
