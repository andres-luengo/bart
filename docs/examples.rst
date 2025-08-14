Examples
========

This section provides practical examples of using the RFI Pipeline for various use cases.

Basic RFI Detection
--------------------

Preparing Input Files
~~~~~~~~~~~~~~~~~~~~~~

First, create a text file listing your HDF5 observation files::

    # file_list.txt
    /data/observations/obs_001.h5
    /data/observations/obs_002.h5
    /data/observations/obs_003.h5

Running Basic Detection
~~~~~~~~~~~~~~~~~~~~~~~

Run the pipeline with default settings::

    rfi-pipeline file_list.txt /output/basic_run

Run with custom thresholds::

    rfi-pipeline file_list.txt /output/custom_run \
        --warm-significance 5.0 \
        --hot-significance 10.0 \
        --frequency-block-size 2048

Advanced Processing
-------------------

High-Performance Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For large datasets with multiple CPU cores::

    rfi-pipeline file_list.txt /output/parallel_run \
        --num-processes 8 \
        --max-rss-gb 256 \
        --num-batches 200

Frequency Range Filtering
~~~~~~~~~~~~~~~~~~~~~~~~~

Process only specific frequency ranges::

    rfi-pipeline file_list.txt /output/filtered_run \
        --min-freq 1000.0 \
        --max-freq 2000.0 \
        --frequency-block-size 512

Resuming Interrupted Jobs
~~~~~~~~~~~~~~~~~~~~~~~~~

If a job is interrupted, resume from the last processed file::

    rfi-pipeline file_list.txt /output/resumed_run --resume

Monitoring and Analysis
-----------------------

Real-time Progress Monitoring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Monitor a running job::

    # Check once and exit
    rfi-pipeline-progress /output/my_run

    # Monitor continuously with 10-second updates
    rfi-pipeline-progress /output/my_run --update-interval 10

    # Monitor in SSH session (no screen clearing)
    rfi-pipeline-progress /output/my_run --update-interval 5 --no-clear

Merging Results
~~~~~~~~~~~~~~~

After processing completes, merge all batch results::

    # Merge to same directory
    rfi-pipeline-merge /output/my_run

    # Merge to custom location with compression
    rfi-pipeline-merge /output/my_run /results/merged \
        --format parquet \
        --compress \
        --sort-by frequency

Python API Examples
-------------------

Programmatic Processing
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from rfi_pipeline import RunManager
    from rfi_pipeline.example import FileJob
    from pathlib import Path
    import logging

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # If your processor needs configuration, pre-bind with functools.partial
    from functools import partial

    # Get list of files
    file_list = Path("file_list.txt")
    with file_list.open() as f:
        files = [Path(line.strip()) for line in f.readlines()]

    # Create and run manager
    job = FileJob({
            'freq_window': 1024,
            'warm_significance': 4.0,
            'hot_significance': 8.0,
            'hotter_significance': 7.0,
            'sigma_clip': 3.0,
            'min_freq': 1000.0,
            'max_freq': 2000.0
    })
    manager = RunManager(
        file_job=job,
        num_batches=50,
        num_processes=4,
        files=tuple(files),
        outdir=Path("output_api"),
        max_rss=64 * 1024**3  # 64 GB
    )

    manager.run()

Memory-Efficient Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For very large datasets that don't fit in memory::

    # Process in smaller batches with limited memory
    rfi-pipeline large_file_list.txt /output/memory_efficient \
        --num-processes 2 \
        --max-rss-gb 16 \
        --num-batches 500 \
        --frequency-block-size 512

Fault-Tolerant Processing
~~~~~~~~~~~~~~~~~~~~~~~~~

Set up processing that can handle file errors gracefully::

    # Use many small batches for better fault tolerance
    rfi-pipeline unreliable_files.txt /output/fault_tolerant \
        --num-batches 1000 \
        --verbose

    # Monitor progress and resume if needed
    rfi-pipeline-progress /output/fault_tolerant --update-interval 30

Data Analysis Examples
----------------------

Statistical Analysis
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # Load merged results
    df = pd.read_csv("merged_results.csv")

    # Basic statistics
    print("RFI Detection Statistics:")
    print(f"Total detections: {len(df)}")
    print(f"Frequency range: {df['frequency'].min():.2f} - {df['frequency'].max():.2f} MHz")
    print(f"Mean kurtosis: {df['kurtosis'].mean():.2f} ± {df['kurtosis'].std():.2f}")

    # Frequency distribution
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(df['frequency'], bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Number of Detections')
    plt.title('RFI Frequency Distribution')

    # Kurtosis distribution
    plt.subplot(1, 2, 2)
    plt.hist(df['kurtosis'], bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Kurtosis')
    plt.ylabel('Number of Detections')
    plt.title('Kurtosis Distribution')
    
    plt.tight_layout()
    plt.savefig('rfi_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

Filtering and Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Load and filter results
    df = pd.read_csv("merged_results.csv")

    # Define RFI categories based on kurtosis
    def classify_rfi(kurtosis):
        if kurtosis > 10:
            return 'Strong RFI'
        elif kurtosis > 5:
            return 'Moderate RFI'
        else:
            return 'Weak RFI'

    df['rfi_category'] = df['kurtosis'].apply(classify_rfi)

    # Filter for specific frequency bands
    l_band = df[(df['frequency'] >= 1000) & (df['frequency'] <= 2000)]
    s_band = df[(df['frequency'] >= 2000) & (df['frequency'] <= 4000)]

    print("RFI by Band:")
    print(f"L-band detections: {len(l_band)}")
    print(f"S-band detections: {len(s_band)}")

    # Category summary
    print("\nRFI Categories:")
    print(df['rfi_category'].value_counts())

Performance Optimization
------------------------

Tuning Parameters
~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pathlib import Path
    from rfi_pipeline.example import FileJob

    # Test different parameter combinations
    parameter_sets = [
        {'warm_significance': 3.0, 'hot_significance': 6.0},
        {'warm_significance': 4.0, 'hot_significance': 8.0},
        {'warm_significance': 5.0, 'hot_significance': 10.0},
    ]

    for i, params in enumerate(parameter_sets):
        process_params = {
            'freq_window': 1024,
            'hotter_significance': 7.0,
            'sigma_clip': 3.0,
            **params
        }
        # Run on test file
        result = FileJob(process_params).run(Path("test_observation.h5"))
        print(f"Parameter set {i+1}: {len(result)} detections")
        print(f"  Warm: {params['warm_significance']}, Hot: {params['hot_significance']}")

Profiling Performance
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import time
    import numpy as np
    from pathlib import Path
    from rfi_pipeline.example.filejob import FileJob

    def benchmark_processing(file_path, process_params, iterations=3):
        """Benchmark processing time for a single file."""
        times = []
        job = FileJob(process_params)
        
        for i in range(iterations):
            start_time = time.time()
            result = job.run(file_path)
            end_time = time.time()
            
            processing_time = end_time - start_time
            times.append(processing_time)
            
            print(f"Iteration {i+1}: {processing_time:.2f}s, {len(result)} detections")
        
        avg_time = sum(times) / len(times)
        print(f"Average time: {avg_time:.2f}s ± {np.std(times):.2f}s")
        
        return avg_time, result

    # Benchmark different block sizes
    block_sizes = [512, 1024, 2048, 4096]
    
    for block_size in block_sizes:
        print(f"\nTesting block size: {block_size}")
        params = {
            'freq_window': block_size,
            'warm_significance': 4.0,
            'hot_significance': 8.0,
            'hotter_significance': 7.0,
            'sigma_clip': 3.0
        }
        
        avg_time, _ = benchmark_processing(
            Path("test_file.h5"), 
            params, 
            iterations=3
        )
