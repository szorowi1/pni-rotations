Regression
----------

This section reviews the mathematics underlying matrix-normal regression. Much of this section was inspired by Shvartsman et al. (2017). [1]_
   
.. contents:: :local:


Matrix-normal distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^

The `matrix-normal distribution <https://en.wikipedia.org/wiki/Matrix_normal_distribution>`_ is a probability distribution that is a generalization of the multivariate normal distribution to matrix-valued random variables. The matrix-normal distribution is parameterized as:

.. math::

    Y \sim \mathcal{MN}(M, \Sigma_t, \Sigma_v)
    
where :math:`Y` are the observations described above; :math:`M \in \mathbb{R}^{T,V}` is the mean of the distribution; and :math:`\Sigma_t \in \mathbb{R}^{T,T}` and :math:`\Sigma_v \in \mathbb{R}^{V,V}` are the temporal and spatial covariance matrices, respectively. The matrix-normal distribution is related to the multivariate normal distribution in that:

.. math::

    \text{vec}(Y) \sim \mathcal{N}(\text{vec}(M), \Sigma_v \otimes \Sigma_t)
    
where :math:`\otimes` denotes the `Kronecker product <https://en.wikipedia.org/wiki/Kronecker_product>`_. We can reformulate the fMRI regression problem using matrix-normal notation:

.. math::

    Y \sim \mathcal{MN}(XW, \Sigma_t, \Sigma_v)

.. math::

    \text{vec}(Y) \sim \mathcal{N}(\text{vec}(XW), \Sigma_v \otimes \Sigma_t)

In this framework, ordinary least squares is equivalent to assuming an identity spatial covariance, :math:`\Sigma_v = I_v`, and an isotropic temporal covariance, :math:`\Sigma_t = \sigma^2I_t`. Similarly, the SPM analysis method can be achieved by assuming a global AR(1) temporal covariance. The question then is how to parameterize the matrix-normal distribution to account for spatiotemporal autocorrelation.


Graph laplacian
^^^^^^^^^^^^^^^
The graph laplacian matrix, :math:`L_{n,n}` is defined with respect to a simple graph with :math:`N` vertices such that:

.. math::

    L = D - A
    
where :math:`D` is the degree matrix and :math:`A` is the adjacency matrix of the graph. Thus, :math:`L_{n,n}` is comprised of integers where:

.. math::

    L_{i,j}:=
    \begin{cases}
    \deg(v_i) & \mbox{if}\ i = j \\
    -1 & \mbox{if}\ i \neq j\ \mbox{and}\ v_i \mbox{ is adjacent to } v_j \\
    0 & \mbox{otherwise}
    \end{cases}

The graph laplacian matrix has three important properties for our purposes.

1. As a covariance matrix, the inverse graph laplacian matrix penalizes differences between adjacent voxels. In other words, it acts as a smoothing matrix.
2. Because it is defined with respect to an adjacency matrix, the graph laplacian matrix can easily be constructed for both volume- and surface-based fMRI data.
3. The graph laplacian matrix is sparse with most of its elements being 0. It is therefore an efficient matrix to perform computations over.

Estimating regression weights (W)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The graph laplacian matrix is therefore well-suited to model the expected spatial autocorrelation of activation weights, :math:`W`. We therefore model the activations weight as matrix-normal distributed:

.. math::

    W \sim \mathcal{MN}(0, \Sigma_k, \Sigma_{vw})
    
where :math:`\Sigma_k` is the weights covariance and :math:`\Sigma_vw` is the weights spatial covariance. We will assume that the regressors are uncorrelated and the spatial covariance is distributed as the inverse graph laplacian:

.. math::

    W \sim \mathcal{MN}(0, I_k, \sigma^2 L^{-1})
    
.. math::

    \text{vec}(W) \sim \mathcal{N}(\text{vec}(0), \sigma^2 L^{-1} \otimes I_k)
    
Given the Gaussian marginalization theorem, we can fold the prior on :math:`W` into the likelihood:

.. math::

    \text{vec}(Y) \sim \mathcal{N}(\text{vec}(0), \Sigma_n + X\Sigma_wX^T)

where :math:`\Sigma_n = \Sigma_v \otimes \Sigma_t` and :math:`\Sigma_w = \sigma^2 L^{-1} \otimes I_k`. The trouble with this formulation is that the inverse of the graph laplacian, :math:`L^{-1}`, is dense. We want a formulation of the above where the graph laplacian is expressed as its precision. To do so, we make use of the `matrix inversion lemma <https://en.wikipedia.org/wiki/Woodbury_matrix_identity>`_:

.. math::

    (A + CBC^T)^{-1} = A^{-1} - A^{-1}C(B^{-1} + C^TA^{-1}C)^{-1}C^TA^{-1}
    
where :math:`A = \Sigma_{n}`, :math:`B = \Sigma_{w}`, and :math:`C = X`. Thus, the likelihood can be rewritten as:

.. math::

    \text{vec}(Y) \sim \mathcal{N}(\text{vec}(0), \Sigma_n^{-1} + \Sigma_n^{-1} X \left(\Sigma_w^{-1} + X^T \Sigma_n^{-1} X \right) X^T \Sigma_n^{-1})

where :math:`\Sigma_n^{-1} = \Sigma_v^{-1} \otimes \Sigma_t^{-1}` and :math:`\Sigma_w^{-1} = \sigma^{-2}L \otimes I_k^{-1}`. In this formulation, the spatial activation covariance is in its precision form and thus sparse and computationally efficient. See the appendix for derivations of the log-likelihood, including a transformation of the log-detemrinant using the matrix determinant lemma.

Using numeric optimization approaches and the log-likelihood, we can solve for the hyperparameters associated with each covariance matrix. After, we can solve for the regression weights:

.. math::

    \hat{W} = \left(\hat{\Sigma}_w^{-1} + X^T \hat{\Sigma}_n^{-1} X \right)^{-1} X^T \hat{\Sigma}_n^{-1} Y


Estimating autocorrelation coefficients (R)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

What about the temporal autocorrelation? Once we have :math:`\hat{W}`, we can get an estimate of the autocorrelated residual error:

.. math::

    U = Y - X\hat{W}
    
where :math:`U \in \mathbb{R}^{T,V}`. Following the convention of autoregression, we can then model the residuals as an AR(n) process such that: 

.. math::

    \text{vec}(U) = ZR + \epsilon
    
where :math:`U` has been vectorized in row-major order; :math:`Z \in \mathbb{R}^{TV,NV}` is the lagged residual matrix (described below); :math:`R \in \mathbb{R}^{NV,1}` is the autocorrelation coefficients, where :math:`\rho_{nv}` denotes the :math:`n` order autocorrelation for voxel :math:`v`; and :math:`\epsilon` is a vector of uncorrelated residuals. To be concrete, we will write this out for a dataset with 3 voxels, 3 time points, and an AR(2) model:

.. math::

    \begin{bmatrix}
    u_{1,1}   \\
    u_{1,2}   \\
    u_{1,3}   \\
    u_{2,1}   \\
    u_{2,2}   \\
    u_{2,3}   \\
    u_{3,1}   \\
    u_{3,2}   \\
    u_{3,3}   \\
    \end{bmatrix} = \begin{bmatrix}
    0 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 0 \\
    u_{1,1} & 0 & 0 & 0 & 0 & 0 \\
    0 & u_{1,2} & 0 & 0 & 0 & 0 \\
    0 & 0 & u_{1,3} & 0 & 0 & 0 \\
    u_{2,1} & 0 & 0 & u_{1,1} & 0 & 0 \\
    0 & u_{2,2} & 0 & 0 & u_{1,2} & 0 \\
    0 & 0 & u_{2,3} & 0 & 0 & u_{1,3}
    \end{bmatrix} \begin{bmatrix}
    \rho_{1,1}   \\
    \rho_{1,2}   \\
    \rho_{1,3}   \\
    \rho_{2,1}   \\
    \rho_{2,2}   \\
    \rho_{2,3}   \\
    \end{bmatrix} + \begin{bmatrix}
    \epsilon_{1,1}   \\
    \epsilon_{1,2}   \\
    \epsilon_{1,3}   \\
    \epsilon_{2,1}   \\
    \epsilon_{2,2}   \\
    \epsilon_{2,3}   \\
    \epsilon_{3,1}   \\
    \epsilon_{3,2}   \\
    \epsilon_{3,3}   \\
    \end{bmatrix}
    
As can be observed, :math:`Z` is a predominantly sparse matrix where the non-zero entries are time-lagged copies of :math:`U`, and there are many columns as voxels times autoregressive coefficients. Solving for :math:`R` allows us to estimate the autocorrelation present in our data. Unfortunately, this structure does not allow for us to use the matrix-normal distribution insofar that we are breaking the Kronecker structure. Fortunately though, we can still put a prior on the autoregressive coefficients:

.. math::

    R \sim \mathcal{N}(0, \Sigma_r)
    
To account for the spatial autocorrelation of the autoregressive coefficients, we can again use the graph laplacian matrix. This time, however, we explicitly paramerize it in a block diagonal matrix:

.. math::

    \Sigma_r = \sigma^2 L^{-1} \otimes I_n = \begin{bmatrix}
    \sigma^2 L^{-1} & 0 & \dots & 0 \\
    0 & \sigma^2 L^{-1} & \dots & 0 \\
    \vdots & \vdots & \ddots & 0 \\
    0 & 0 & \dots & \sigma^2 L^{-1} \\
    \end{bmatrix}
    
Again using the Gaussian marginalization theorem, we have:

.. math::
    
    \text{vec}(U) = \mathcal{N}(0, \Sigma_n + Z \Sigma_r Z^T)
    
where :math:`\Sigma_n` is the noise covariance capturing the uncorrelated residuals. We will assume it is an isotropic covariance matrix, :math:`\Sigma_n = \sigma^2 I_{tv}`. Again we will use the matrix inversion lemma to obtain a likelihood function with a sparse graph laplacian matrix:

.. math::

    \text{vec}(U) = \mathcal{N} \left( 0, \Sigma_n^{-1} + \Sigma_n^{-1}Z\left(\Sigma_r^{-1} + Z^T\Sigma_n^{-1}Z \right) Z^T \Sigma_n^{-1} \right) 

See the appendix for derivations of the log-likelihood.

Using numeric optimization approaches and the log-likelihood, we can solve for the hyperparameters associated with each covariance matrix. After, we can solve for the autocorrelation coefficients:

.. math::

    \hat{R} = \left(\hat{\Sigma}_r^{-1} + Z^T \hat{\Sigma}_n^{-1} Z \right)^{-1} Z^T \hat{\Sigma}_n^{-1} \vec{U}


Two-step algorithm
^^^^^^^^^^^^^^^^^^
    
With closed-form estimates of both the regression weights, :math:`W`, and autocorrelation coefficients, :math:`R`, we can perform an iterative two-step algorithm for optimizing the hyperparameters controlling both sets of estimates

1. Regression step
    a. Optimize with respect to hyperparameters in :math:`\Sigma_w`, holding :math:`\Sigma_n` constant.
    b. Solve for :math:`\hat{W}` and compute :math:`\hat{U}`.
2. Autocorrelation step
    a. Optimize with respect to hyperparameters in :math:`\Sigma_r` and :math:`\Sigma_n`.
    b. Solve for :math:`\hat{R}` and compute :math:`\Sigma_n`.
    
Appendix
^^^^^^^^

Derivation of activation log-likelihood
=======================================

Beginning with the likelihood function defined above:

.. math::

    \text{vec}(Y) \sim \mathcal{N}(\text{vec}(0), \Sigma_n^{-1} + \Sigma_n^{-1} X \left(\Sigma_w^{-1} + X^T \Sigma_n^{-1} X \right) X^T \Sigma_n^{-1})

The posterior covariance is equal to:

.. math::

    \Sigma = \Sigma_n + X \Sigma_w X^T

.. math::

    \Sigma^{-1} = \Sigma_n^{-1} + \Sigma_n^{-1} X \left(\Sigma_w^{-1} + X^T \Sigma_n^{-1} X \right) X^T \Sigma_n^{-1}

Therefore the log-likelihood can be expressed as:

.. math::

    \ell = -0.5 \left( \log | \Sigma | + Y^T \Sigma^{-1} Y \right)
    
The quadratic portion can be written as:
    
.. math::

    Y^T \Sigma^{-1} Y  = Y^T \left( \Sigma_n^{-1} + \Sigma_n^{-1} X \left(\Sigma_w^{-1} + X^T \Sigma_n^{-1} X \right) X^T \Sigma_n^{-1} \right) Y
    
.. math::

    = Y^T \Sigma_n^{-1} Y +  Y^T\left(\Sigma_n^{-1} X \left(\Sigma_w^{-1} + X^T \Sigma_n^{-1} X \right) X^T \Sigma_n^{-1} \right) Y

    
where :math:`\Sigma_n^{-1} = \Sigma_v^{-1} \otimes \Sigma_t^{-1}` and :math:`\Sigma_w^{-1} = \Sigma_{wv}^{-1} \otimes \Sigma_k^{-1}`. Using the graph laplacian prior as above, we have:

.. math::

    = Y^T \Sigma_n^{-1} Y +  Y^T\left(\Sigma_n^{-1} X \left(\sigma^{-2}L \otimes I_k + X^T \Sigma_n^{-1} X \right) X^T \Sigma_n^{-1} \right) Y


The determinant portion can be evaluated using the `matrix determinant lemma <https://en.wikipedia.org/wiki/Matrix_determinant_lemma>`_:

.. math::

    \log| A + CBC^T | = \log| B^{-1} + C^TA^{-1}C | + \log| B | + \log| A |
    
where :math:`A = \Sigma_{n}`, :math:`B = \Sigma_{w}`, and :math:`C = X`. Thus, the log-determinant form of the log-likelihood term can be expressed as:

.. math::

    \log | \Sigma_n + X \Sigma_w X^T | = \log| \Sigma_w^{-1} + X^T \Sigma_n^{-1} X | + \log| \Sigma_w | + \log| \Sigma_n |
    
Given the determinant property of the `Kronecker product <https://en.wikipedia.org/wiki/Kronecker_product#Properties>`_ we can reexpress the log-determinant as:

.. math::

    = \log| \Sigma_w^{-1} + X^T \Sigma_n^{-1} X | + K \log| \Sigma_{wv} | + V \log| \Sigma_k | + T \log| \Sigma_v | + V \log| \Sigma_t |

where :math:`T, V, K` are the number of time points, voxels, and regressors, respectively.

Using the graph laplacian prior as above, we have:

.. math::

    = \log| \sigma^{-2}L \otimes I_k + X^T \Sigma_n^{-1} X | + K \log| \sigma^2L^{-1} | + T \log| \Sigma_v | + V \log| \Sigma_t |
    
The log-determinant of the graph laplacian matrix can be further broken down:

.. math:: 

    \log| \sigma^2L^{-1} | = -\log| \sigma^{-2}L | = -\log \left( (\sigma^{-2})^{V-1} \right) - \log | L | = 2(V - 1) \log \sigma
    
As has been demonstrated elsewhere [9]_, the log-determinant of the graph laplacian is 0. Thus, the final log-determinant is:
    
.. math::
    
    = \log| \sigma^{-2}L \otimes I_k + X^T \Sigma_n^{-1} X | + 2K(V - 1) \log \sigma + T \log| \Sigma_v | + V \log| \Sigma_t |

    
Derivation of noise log-likelihood
==================================