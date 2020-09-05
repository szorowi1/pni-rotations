Background
----------

Let us begin by considering the usual problem in fMRI regression analysis. We observe a set of measurements :math:`Y \in \mathbb{R}^{T,V}`, where :math:`T` is the number of time points (acquisitions) and :math:`V` is the number of voxels (where typically :math:`T \ll V`). We also have a design matrix, :math:`X \in \mathbb{R}^{T,K}`, comprised of :math:`K` timeseries including both task and nuisance regressors. The goal is to estimate a set of activation weights, :math:`W \in \mathbb{R}^{K,V}`, relating the hypothetical to observed signals. Mathematically we can express this as:

.. math::

    Y = XW + \epsilon
    
where :math:`\epsilon \in \mathbb{R}^{V,1}` is the residual error. This would be a simple problem except for the fact that fMRI data exhibits spatiotemporal autocorrelation. [2]_ Put another way, the BOLD signal at time :math:`t` for two adjacent tends to be correlated. Similarly, the BOLD signal within a voxel at times :math:`t` and :math:`t+1` also tend to be correlated. As such, we cannot assume the residuals are independent and identically distributed, i.e. :math:`\epsilon \sim \mathcal{N}(0,\sigma^2I)`, and ordinary least squares cannot be performed.

Rather than explicitly model the degree of spatiotemporal autocorrelation, however, conventional fMRI analysis handles both during preprocessing. Spatial autocorrelation is typically accounted for using a fixed-width Gaussian kernel as motivated by the matched filter theorem. [3]_ Temporal autocorrelation is typically accounted for using a prewhitening procedure, which ensures the :math:`iid` assumptions necessary for least-squares regression. [4]_ 

Importantly both methods for handling autocorrelation make non-trivial assumptions about the data. According to the matched filter theorem, the size of the filter kernel should match the spatial frequency of the desired signal. Because this is often unknown in advance, smoothing is applied using a best-guess of the optimal kernel width (typically 6mm or 12mm). This is not without risk. Suboptimally small kernels will preserve true signal amplitude but will inadequately suppress noise. In contrast, suboptimally large kernels will effectively reduce noise but also shrink true signal amplitude. As such, suboptimally small and large smoothing kernels increase the risk of false positives and false negatives, respectively. This problem is compounded by the fact that the spatial extent of the true signal, and thus the ideal kernel size, will likely vary across the brain due to regional differences in the vasculature. [5]_ Thus, the choice of kernel size represents a trade-off between estimation power (the ability to accurately resolve BOLD signal change) and detection power (the ability to distinguish true from spurious signal change). 

Assumptions in prewhitening are similarly non-trivial. Underestimating temporal autocorrelation will spuriously increase the estimated number of degrees of freedom and consequently increase the false positive rate in first level analyses and may bias results in second level analyses (especially for rapid event related designs). [6]_ This problem is similarly compounded by the fact that the degree of temporal autocorrelation varies spatially across the brain. [7]_ As such, it is unsurprising that SPM, which assumes a global AR(1) temporal autocorrelation, has worse estimation issues than the other major packages. [7,8]_

In summary, spatiotemporal autocorrelation is a significant challenge for fMRI regression analysis. The usual methods for handling this rely on statistical assumptions that run non-trivial risk for being violated. Ideally, we would use an analytic method that estimates the spatiotemporal autocorrelation from the data directly without need for smoothing or prewhitening. In the sections below, we introduce matrix-normal fMRI regression with the hopes of overcoming these challenges.

References
^^^^^^^^^^
.. [1] Shvartsman, M., Sundaram, N., Aoi, M. C., Charles, A., Wilke, T. C., & Cohen, J. D. (2017). Matrix-normal models for fMRI analysis. arXiv preprint arXiv:1711.03058.
.. [2] Bullmore, E., Brammer, M., Williams, S. C., Rabe‐Hesketh, S., Janot, N., David, A., ... & Sham, P. (1996). Statistical methods of estimation and inference for functional MR image analysis. Magnetic Resonance in Medicine, 35(2), 261-277.
.. [3] Worsley, K. J., Marrett, S., Neelin, P., & Evans, A. C. (1996). Searching scale space for activation in PET images. Human brain mapping, 4(1), 74-90.
.. [4] Woolrich, M. W., Ripley, B. D., Brady, M., & Smith, S. M. (2001). Temporal autocorrelation in univariate linear modeling of FMRI data. Neuroimage, 14(6), 1370-1386.
.. [5] Lindquist, M. A., Loh, J. M., & Yue, Y. R. (2010). Adaptive spatial smoothing of fMRI images. Statistics and its Interface, 3(1), 3-13.
.. [6] Olszowy, W., Aston, J., Rua, C., & Williams, G. B. (2018). Accurate autocorrelation modeling substantially improves fMRI reliability. bioRxiv, 323154.
.. [7] Worsley, K. J., Liao, C. H., Aston, J., Petre, V., Duncan, G. H., Morales, F., & Evans, A. C. (2002). A general statistical analysis for fMRI data. Neuroimage, 15(1), 1-15.
.. [8] Eklund, A., Andersson, M., Josephson, C., Johannesson, M., & Knutsson, H. (2012). Does parametric fMRI analysis with SPM yield valid results?—An empirical study of 1484 rest datasets. NeuroImage, 61(3), 565-578.
.. [9] Jin, X., Carlin, B. P., & Banerjee, S. (2005). Generalized hierarchical multivariate CAR models for areal data. Biometrics, 61(4), 950-961.