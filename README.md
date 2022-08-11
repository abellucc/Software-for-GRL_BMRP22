# Software-for-GRL_BMRP22
Scripts for GRL paper Bellucci et al. 2022 (Intermittent behaviour in AMOC-AMV relationship)  

This script is a diagnostic tool used to perform the analyses presented in Bellucci et al.(2022; hereafter B22).
Specifically, the scripts implements a change point detection algorithm to identify changes in the statistical
properties of an AMOC-AMV moving correlation data sequence. The algorithm is based on the Pruned Exact Linear Time
(PELT) scheme (Killick et al., 2012) and requires the ruptures Python library, a package designed for the analysis
and segmentation of non-stationary signals (Truong et al., 2020). 
Additionally, this script does also implement continuous wavelet transform analysis (Terrence and Compo (1998) and 
other statistical analyses as presented in B22.

References

Bellucci, A., Mattei, D., Ruggieri, P. and Famooss Paolini, L. (2022), Intermittent behavior in the AMOC-AMV relationship,
Geophyiscal Research Letters, under review.

Killick, R., Fearnhead, P., and Eckley, I. A. (2012), Optimal Detection of Changepoints With a Linear Computational Cost, 
Journal of the American Statistical Association, 107:500, 1590-1598.

Torrence, C., and Compo, G. P. (1998), A Practical Guide to Wavelet Analysis. 
Bulletin of the American Meteorological Society, 79(1):61–78.

Truong, C., Oudre, L., and Vayatis, N. Selective review of offline change point detection methods. 
Signal Processing, 167:107299, 2020.  
