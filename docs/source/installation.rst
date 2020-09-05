Installation
------------

To install matnormal from Github, you can use `pip <https://pip.pypa.io/en/stable/>`_ in a terminal:

.. code-block:: bash

    pip install git+https://github.com/szorowi1/matrix-normal.git

If not already installed, you will also need to install `SuiteSparse <https://github.com/jluttine/suitesparse>`_:

.. code-block:: bash
    
    conda install -c conda-forge suitesparse 
    
Dependencies
^^^^^^^^^^^^
The minimum required dependencies to run matnormal are:

- NumPy
- SciPy
- scikit-sparse
- mne (optional, used in building surface-based adjacency matrices)