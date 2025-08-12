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

The RFI Pipeline is a Python package for detecting and analyzing Radio Frequency Interference (RFI) in astronomical observation data. It provides efficient multi-processing capabilities for analyzing large volumes of astronomical data stored in HDF5 format.

The package is designed as a flexible framework where users can implement their own file processing functions. An example implementation is provided in the ``rfi_pipeline.example`` module to demonstrate how to create custom processors.

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

Basic usage::

    rfi-pipeline file_list.txt output_directory --num-processes 4

For more detailed usage instructions, see the :doc:`usage` guide.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
