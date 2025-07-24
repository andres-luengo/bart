"""
RFI Pipeline Merge Module

This module provides functionality to merge batch CSV files from an RFI pipeline run
into a single consolidated output file.
"""

__version__ = "0.1.0"

from pathlib import Path
from typing import Optional
import pandas as pd
import json
import logging


def merge_rfi_run(rundir: Path, outdir: Optional[Path] = None, 
                 format_type: str = 'csv', compress: bool = False,
                 sort_by: str = 'frequency', force: bool = False) -> Path:
    """
    Merge batch CSV files from an RFI pipeline run into a single output file.
    
    Args:
        rundir: Path to the RFI pipeline run directory (should contain batches/ subdirectory)
        outdir: Output directory. If None, uses rundir (same directory as the run)
        format_type: Output format ('csv', 'parquet', or 'hdf5')
        compress: Whether to compress the output
        sort_by: Column to sort by
        force: Whether to overwrite existing files
        
    Returns:
        Path to the merged output file
        
    Raises:
        FileNotFoundError: If rundir or batch files are not found
        ValueError: If invalid format_type is specified
    """
    from .__main__ import (
        find_batch_files, load_metadata, merge_batch_files, 
        save_merged_data, setup_logging
    )
    
    # Set up basic logging
    logger = setup_logging(1)  # INFO level
    
    # Validate format
    if format_type not in ['csv', 'parquet', 'hdf5']:
        raise ValueError(f"Invalid format_type: {format_type}. Must be 'csv', 'parquet', or 'hdf5'")
    
    # Set default output directory
    if outdir is None:
        outdir = rundir  # Output to the same directory as the run
    
    # Find and merge batch files
    batch_files = find_batch_files(rundir, logger)
    metadata = load_metadata(rundir, logger)
    merged_df = merge_batch_files(batch_files, logger)
    
    # Save merged data
    save_merged_data(
        merged_df, outdir, format_type, compress, 
        sort_by, force, metadata, logger
    )
    
    # Return path to output file
    if format_type == 'csv':
        filename = 'merged_data.csv.gz' if compress else 'merged_data.csv'
    elif format_type == 'parquet':
        filename = 'merged_data.parquet'
    else:  # hdf5
        filename = 'merged_data.h5'
    
    return outdir / filename