Usage Guide
===========

The RFI Pipeline provides both command-line tools and a Python API for detecting radio frequency interference in astronomical data.

Command Line Tools
------------------

Main Pipeline Tool
~~~~~~~~~~~~~~~~~~

The primary tool for running RFI detection::

    rfi-pipeline input_files.txt output_directory [options]

**Basic Example**::

    rfi-pipeline file_list.txt /path/to/output --num-processes 4 --max-rss-gb 64

**Input/Output Options:**

* ``infile``: Path to file containing paths to .h5 files (one per line)
* ``outdir``: Output directory for results
* ``-f, --force``: Overwrite existing output directory
* ``-c, --resume``: Resume processing from last completed file

**Processing Options:**

* ``--frequency-block-size``: Size of frequency blocks (default: 1024)
* ``--warm-significance``: Initial filtering threshold (default: 4.0)
* ``--hot-significance``: Secondary filtering threshold (default: 8.0)
* ``--hotter-significance``: SNR threshold for final filtering (default: 7.0)
* ``--sigma-clip``: Sigma clipping value (default: 3.0)
* ``--min-freq``: Minimum frequency to analyze (default: -inf)
* ``--max-freq``: Maximum frequency to analyze (default: inf)

**Resource Management:**

* ``-n, --num-processes``: Number of parallel processes (default: 1)
* ``-m, --max-rss-gb``: Maximum memory usage in GB (default: 32.0)
* ``--num-batches``: Number of processing batches (default: 100)

**Logging Options:**

* ``-v, --verbose``: Increase verbosity (can be repeated)
* ``-q, --quiet``: Decrease verbosity (can be repeated)

Progress Monitoring
~~~~~~~~~~~~~~~~~~~

Monitor the progress of a running RFI pipeline::

    rfi-pipeline-progress rundir [options]

**Examples:**

Check progress once and exit::

    rfi-pipeline-progress /path/to/pipeline/output

Monitor continuously with updates every 5 seconds::

    rfi-pipeline-progress /path/to/pipeline/output --update-interval 5

Monitor without screen clearing (useful for SSH)::

    rfi-pipeline-progress /path/to/pipeline/output -n 3 --no-clear

**Options:**

* ``rundir``: Path to the RFI pipeline output directory
* ``-n, --update-interval SECONDS``: Update interval for continuous monitoring
* ``--once``: Run once and exit
* ``--no-clear``: Disable screen clearing between updates

The progress monitor displays:

* Number of active workers
* Overall progress with percentage and progress bar
* Estimated time remaining
* Time since last file completion
* Individual worker progress and statistics

Merge Tool
~~~~~~~~~~

Merge all batch results into a single file::

    rfi-pipeline-merge rundir [outdir] [options]

**Examples:**

Merge batches to the same directory::

    rfi-pipeline-merge /path/to/pipeline/output

Merge to custom output directory::

    rfi-pipeline-merge /path/to/pipeline/output /path/to/merged/results

Output with compression::

    rfi-pipeline-merge /path/to/pipeline/output --compress --verbose

**Options:**

* ``rundir``: Path to RFI pipeline output directory
* ``outdir``: Output directory for merged data (optional)
* ``-f, --force``: Overwrite existing output files
* ``-v, --verbose``: Increase verbosity level
* ``--format``: Output format (csv, parquet, hdf5)
* ``--sort-by``: Column to sort by (default: frequency)
* ``--compress``: Compress output file

Python API
----------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    from rfi_pipeline import RunManager
    from rfi_pipeline.example import FileJob
    from pathlib import Path

    # Set up processing parameters
    process_params = {
        'freq_window': 1024,
        'warm_significance': 4.0,
        'hot_significance': 8.0,
        'hotter_significance': 7.0,
        'sigma_clip': 3.0,
        'min_freq': float('-inf'),
        'max_freq': float('inf')
    }

    # Initialize manager with the default file processor
    # (you can also provide a custom file processing function)
    files = [Path("data1.h5"), Path("data2.h5")]
    manager = RunManager(
        file_job=FileJob.run_func,
        process_params=process_params,
        num_batches=10,
        num_processes=4,
        files=tuple(files),
        outdir=Path("output"),
        max_rss=32 * 1024**3  # 32 GB in bytes
    )

    # Run processing
    manager.run()

Custom File Processors
~~~~~~~~~~~~~~~~~~~~~~

The RFI Pipeline is designed to work with custom file processing functions.
The built-in processor is provided as an example in ``rfi_pipeline.example.filejob``.
You can create your own file processor by implementing a function that takes
a file path and processing parameters and returns a pandas DataFrame:

.. code-block:: python

    from rfi_pipeline.example.filejob import FileJob
    
    # Or create your own custom processor
    def custom_file_processor(file_path, process_params):
        # Your custom processing logic here
        # Must return a pandas DataFrame with the required columns
        pass
    
    manager = RunManager(
        file_job=my_file_processor,  # or FileJob.run_func
        process_params=process_params,
        # ... other parameters
    )

Merge API
~~~~~~~~~

.. code-block:: python

    from pathlib import Path
    from rfi_pipeline.merge import merge_rfi_run

    # Merge batch files programmatically
    output_path = merge_rfi_run(
        rundir=Path("/path/to/pipeline/output"),
        outdir=Path("/path/to/merged/results"),
        format_type='csv',
        compress=True,
        sort_by='frequency',
        force=True
    )
    print(f"Merged data saved to: {output_path}")

Algorithm Details
-----------------

The RFI detection algorithm operates in multiple stages:

1. **Data Loading**: HDF5 files are loaded and divided into frequency blocks
2. **Warm Filtering**: Initial filtering using sigma-based thresholds
3. **Hot Filtering**: Secondary filtering using median absolute deviation
4. **Hotter Filtering**: Final SNR-based filtering with sigma clipping
5. **Feature Extraction**: Computation of frequency and kurtosis statistics

**Statistical Methods:**

* **Warm Filter**: Identifies blocks where max value exceeds median by ``warm_significance`` standard deviations
* **Hot Filter**: Further filters using ``hot_significance`` median absolute deviations
* **Hotter Filter**: SNR-based filtering using sigma-clipped noise estimation

**Output Data:**

Each detection includes:

* ``frequency``: Central frequency of the detection
* ``kurtosis``: Statistical kurtosis of the normalized signal
* ``source_file``: Path to the source data file
* ``batch_number``: Batch number (when merged)

Output Structure
----------------

The pipeline creates the following output structure::

    output_directory/
    ├── batches/
    │   ├── batch_000.csv
    │   ├── batch_001.csv
    │   └── ...
    ├── logs/
    │   ├── all_logs.log
    │   └── error_logs.log
    ├── files.csv
    ├── meta.json
    ├── progress-data.json
    └── target-list.txt

**File Descriptions:**

* ``batches/``: Individual CSV files for each processing batch
* ``logs/``: Comprehensive and error-specific log files
* ``files.csv``: List of processed files with metadata
* ``meta.json``: Processing metadata and parameters
* ``progress-data.json``: Real-time progress tracking data
* ``target-list.txt``: Copy of input file list

Performance Considerations
--------------------------

**Memory Usage:**

* Each process loads data files incrementally to manage memory
* Use ``--max-rss-gb`` to limit total memory consumption
* Recommended: 32 GB × number of processes

**Processing Speed:**

* Processing time depends on data volume and RFI density
* More processes generally improve performance but increase memory usage
* Optimal number of processes depends on available CPU cores and memory

**Batch Size:**

* Larger batch counts provide better fault tolerance
* Smaller batches allow for more granular progress monitoring
* Default of 100 batches works well for most use cases
