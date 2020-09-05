API Reference
-------------

Covariances
^^^^^^^^^^^

Data covariance objects (wrapper around :code:`scipy.sparse.csc_matrix`).

.. currentmodule:: matnormal.covs

.. autosummary::
   :template: class.rst
   :toctree: _autosummary
   
   CovIdentity
   CovIsotropic
   CovGraphLaplacian
    
Likelihood
^^^^^^^^^^

Matrix-normal likelihood functions.

.. currentmodule:: matnormal.likelihood

.. autosummary::
   :template: function.rst
   :toctree: _autosummary
   
   matnormal_lpdf
   matnormal_inverse

Estimators
^^^^^^^^^^

Model estimation classes.

.. currentmodule:: matnormal.covs

.. autosummary::
   :template: function.rst
   :toctree: _autosummary

   matnormal_regression

Utilities
^^^^^^^^^

.. currentmodule:: matnormal.utilities

.. autosummary::
   :template: function.rst
   :toctree: _autosummary
   
   surf_to_adj
   vol_to_adj
   spm_hrf