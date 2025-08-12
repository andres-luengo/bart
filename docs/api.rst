API Reference
=============

This section contains the full API documentation for the RFI Pipeline package.

.. currentmodule:: rfi_pipeline

Main Classes
------------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   RunManager
   BatchJob
   FileJob

Core Modules
------------

Manager Module
~~~~~~~~~~~~~~

.. automodule:: rfi_pipeline.manager
   :members:
   :undoc-members:
   :show-inheritance:

FileJob Module
~~~~~~~~~~~~~~

.. automodule:: rfi_pipeline.filejob
   :members:
   :undoc-members:
   :show-inheritance:

BatchJob Module
~~~~~~~~~~~~~~~

.. automodule:: rfi_pipeline.batchjob
   :members:
   :undoc-members:
   :show-inheritance:

Command Line Tools
------------------

Check Progress Module
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: rfi_pipeline.check_progress
   :members:
   :undoc-members:
   :show-inheritance:

Merge Module
~~~~~~~~~~~~

.. automodule:: rfi_pipeline.merge
   :members:
   :undoc-members:
   :show-inheritance:

Utility Functions
-----------------

The package provides several utility functions that can be used independently:

Statistical Functions
~~~~~~~~~~~~~~~~~~~~~

Functions for statistical analysis and filtering used in the RFI detection algorithm.

.. note::
   These functions are primarily used internally by the main classes but can be
   accessed for custom processing workflows.

Data I/O Functions
~~~~~~~~~~~~~~~~~~

Functions for reading and writing HDF5 files and CSV output data.

Constants and Configuration
---------------------------

.. autodata:: rfi_pipeline.__version__
   :annotation: = "0.1.0"

.. autodata:: rfi_pipeline.__author__
   :annotation: = "Breakthrough Listen"

.. autodata:: rfi_pipeline.__description__
   :annotation: = "Radio Frequency Interference Detection Pipeline"
