RFI Pipeline Documentation
==========================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   api
   examples

Overview
--------

``rfi_pipeline`` is a Python package for batching and parallelizing file processing tasks.

The package is designed as a simplified framework where users can implement their own file processing functions. Features include:

* Simplified process-based parallelization.
* Python :mod:`logging` compatibility to monitor processing.
* Progress tracking (see :ref:`progress-monitoring`)

``rfi_pipeline.example`` implements a simple algorithm for finding RFI in data from the Green Bank Telescope. This script can be executed from the command line with ``python -m rfi_pipeline`` or just ``rfi-pipeline``.

Key Features
~~~~~~~~~~~~

* **Flexible Architecture**: Framework supports custom file processing functions
* **Multi-processing support**: Parallel processing of data batches for improved performance
* **Statistical filtering**: Two-stage filtering process using configurable significance thresholds
* **Batch processing**: Divides large datasets into manageable batches for reliability
* **Comprehensive logging**: Detailed logging with configurable verbosity levels
* **Memory management**: Configurable memory limits to prevent system overload
* **Flexible configuration**: Command-line interface with extensive parameter customization
* **Example implementation**: Includes a complete example processor for RFI detection

Quick Start
-----------

Installation::

    pip install rfi-pipeline

Basic usage:

.. code-block:: python

    import numpy as np
    import pandas as pd

    import logging

    from h5py import File

    from rfi_pipeline import RunManager

    def some_processing(file_path):
        """
        Define the processing you're interested in here. Could be an RFI
        survey, a SETI project, gathering statistics to train an ML model...
        """
        logger = logging.getLogger('process_worker')
        logger.info(f'Reading in {file_path}')

        f = File(file_path)
        results = []
        num_time_slices = f['data'].shape[0]
        
        for i in range(num_time_slices):
            results.append({
                'time_index': i,
                'mean': np.mean(f['data'][i]),
                'median': np.median(f['data'][i]),
                'std': np.std(f['data'][i])
            })
        logger.info(f'Done!')
        return results
    
    files = [
        '/datag/pipeline/AGBT24B_999_21/blc04_blp04/blc04_guppi_60652_09060_DYSON5_0039.rawspec.0000.h5'
        '/datag/pipeline/AGBT20A_999_30/blc70_blp30/blc70_guppi_58984_35843_TIC20182165_0095.rawspec.0000.h5'
    ]
    manager = RunManager(
        file_job=some_processing,
        files=files,
        outdir='./example-run'
    )

For more detailed usage instructions, see the :doc:`usage`.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
