Installation
============

Requirements
------------

The RFI Pipeline requires Python 3.8 or later and has the following dependencies:

* numpy >= 1.20.0
* pandas >= 1.3.0
* scipy >= 1.7.0
* h5py >= 3.0.0
* hdf5plugin >= 3.0.0

Optional dependencies for development:

* pytest >= 6.0
* pytest-cov
* black
* flake8
* mypy
* jupyter

Install from PyPI
-----------------

.. note::
   This package is not yet published on PyPI. Use the development installation method below.

Once published, you will be able to install the package using pip::

    pip install rfi-pipeline

Development Installation
------------------------

To install the latest development version from the source repository:

1. Clone the repository::

    git clone https://github.com/breakthrough-listen/rfi-pipeline.git
    cd rfi-pipeline

2. Install in development mode::

    pip install -e .

3. Or install with development dependencies::

    pip install -e ".[dev]"

4. For documentation building, install with docs dependencies::

    pip install -e ".[docs]"

Verification
------------

To verify that the installation was successful, you can run::

    rfi-pipeline --help

This should display the command-line help for the RFI pipeline tool.

You can also import the package in Python::

    import rfi_pipeline
    print(rfi_pipeline.__version__)

System Requirements
-------------------

**Memory Requirements**

The RFI Pipeline can be memory intensive when processing large datasets. We recommend:

* At least 32 GB RAM for typical processing
* Memory allocation should be at least 32 GB × number of processes
* Use the ``--max-rss-gb`` parameter to control memory usage

**Storage Requirements**

* Input data: HDF5 files containing astronomical observation data
* Output storage: Depends on the amount of RFI detected, typically much smaller than input
* Temporary storage: Batch processing requires minimal temporary storage

**CPU Requirements**

* Multi-core processor recommended for parallel processing
* Use ``--num-processes`` to control the number of parallel workers
