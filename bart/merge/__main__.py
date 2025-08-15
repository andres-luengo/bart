import argparse
import json
import logging
from pathlib import Path
from typing import Optional
import shutil

import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog='bart-merge',
        description='Merge batch CSV files from a BART run into a single output file.'
    )
    
    parser.add_argument(
        'rundir',
        type=Path,
        help='Path to the output directory of a BART run (should contain batches/ subdirectory)'
    )
    
    parser.add_argument(
        'outdir',
        type=Path,
        nargs='?',
        help='Path to output directory for merged data. If not specified, uses the run directory itself'
    )
    
    parser.add_argument(
        '-f', '--force',
        action='store_true',
        help='Overwrite existing output files without confirmation'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='count',
        default=0,
        help='Increase verbosity level. Use -v for INFO, -vv for DEBUG'
    )
    
    parser.add_argument(
        '--format',
        choices=['csv', 'parquet', 'hdf5'],
        default='csv',
        help='Output format for merged data (default: csv)'
    )
    
    parser.add_argument(
        '--sort-by',
        default='frequency',
        help='Column to sort merged data by (default: frequency)'
    )
    
    parser.add_argument(
        '--compress',
        action='store_true',
        help='Compress output file (format-dependent compression)'
    )
    
    parser.add_argument(
        '--read-only',
        action='store_true',
        help='Read-only mode: do not modify the original meta.json file. Safe for active runs.'
    )
    
    return parser.parse_args()


def setup_logging(verbosity: int) -> logging.Logger:
    """Set up logging based on verbosity level."""
    logger = logging.getLogger('bart-merge')
    
    # Set log level based on verbosity
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity >= 1:
        level = logging.INFO
    else:
        level = logging.WARNING
    
    # Configure logger
    logger.setLevel(level)
    
    # Create console handler
    handler = logging.StreamHandler()
    handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s: %(message)s'
    )
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    return logger


def find_batch_files(rundir: Path, logger: logging.Logger) -> list[Path]:
    """Find all batch CSV files in the run directory."""
    batches_dir = rundir / 'batches'
    
    if not batches_dir.exists():
        raise FileNotFoundError(f"Batches directory not found: {batches_dir}")
    
    if not batches_dir.is_dir():
        raise NotADirectoryError(f"Batches path is not a directory: {batches_dir}")
    
    # Find all batch_*.csv files
    batch_files = list(batches_dir.glob('batch_*.csv'))
    
    if not batch_files:
        raise FileNotFoundError(f"No batch CSV files found in: {batches_dir}")
    
    # Sort batch files by number
    batch_files.sort(key=lambda x: int(x.stem.split('_')[1]))
    
    logger.info(f"Found {len(batch_files)} batch files")
    logger.debug(f"Batch files: {[f.name for f in batch_files]}")
    
    return batch_files


def load_metadata(rundir: Path, logger: logging.Logger) -> Optional[dict]:
    """Load metadata from meta.json if it exists."""
    meta_file = rundir / 'meta.json'
    
    if meta_file.exists():
        try:
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Loaded metadata from {meta_file}")
            logger.debug(f"Metadata: {metadata}")
            return metadata
        except Exception as e:
            logger.warning(f"Failed to load metadata from {meta_file}: {e}")
    else:
        logger.info("No metadata file found")
    
    return None


def copy_files_csv(rundir: Path, outdir: Path, force: bool, logger: logging.Logger) -> None:
    """Copy files.csv meta table to output directory if it exists."""
    source_files_csv = rundir / 'files.csv'
    
    if not source_files_csv.exists():
        logger.info("No files.csv found in run directory")
        return
    
    dest_files_csv = outdir / 'files.csv'
    
    # Check if destination exists and handle overwrite
    if dest_files_csv.exists() and not force:
        response = input(f"files.csv already exists in {outdir}. Overwrite? (y/N): ")
        if response.lower() != 'y':
            logger.info("Skipping files.csv copy")
            return
    
    try:
        shutil.copy2(source_files_csv, dest_files_csv)
        logger.info(f"Copied files.csv from {source_files_csv} to {dest_files_csv}")
    except Exception as e:
        logger.warning(f"Failed to copy files.csv: {e}")


def merge_batch_files(batch_files: list[Path], logger: logging.Logger) -> pd.DataFrame:
    """Merge all batch CSV files into a single DataFrame."""
    logger.info("Starting to merge batch files...")
    
    dataframes = []
    total_rows = 0
    
    for i, batch_file in enumerate(batch_files):
        logger.debug(f"Processing {batch_file.name} ({i+1}/{len(batch_files)})")
        
        try:
            df = pd.read_csv(batch_file)
            rows_in_batch = len(df)
            total_rows += rows_in_batch
            
            # Add batch information
            df['batch_number'] = int(batch_file.stem.split('_')[1])
            
            dataframes.append(df)
            
            logger.debug(f"  Loaded {rows_in_batch} rows from {batch_file.name}")
            
        except Exception as e:
            logger.error(f"Failed to load {batch_file}: {e}")
            raise
    
    logger.info(f"Loaded {total_rows} total rows from {len(batch_files)} batch files")
    
    # Concatenate all dataframes
    logger.info("Concatenating dataframes...")
    merged_df = pd.concat(dataframes, ignore_index=True)
    
    logger.info(f"Merged data shape: {merged_df.shape}")
    
    return merged_df


def save_merged_data(df: pd.DataFrame, outdir: Path, format_type: str, 
                    compress: bool, sort_by: str, force: bool, 
                    metadata: Optional[dict], read_only: bool, logger: logging.Logger) -> None:
    """Save merged DataFrame to output file."""
        
    # Sort data if requested
    if sort_by and sort_by in df.columns:
        logger.info(f"Sorting data by '{sort_by}'...")
        df = df.sort_values(sort_by)
    elif sort_by:
        logger.warning(f"Sort column '{sort_by}' not found in data. Available columns: {list(df.columns)}")
    
    # Determine output filename
    if format_type == 'csv':
        filename = 'merged_data.csv.gz' if compress else 'merged_data.csv'
        output_path = outdir / filename
        
        # Check if file exists
        if output_path.exists() and not force:
            response = input(f"Output file {output_path} already exists. Overwrite? (y/N): ")
            if response.lower() != 'y':
                logger.info("Merge cancelled by user")
                return
        
        logger.info(f"Saving merged data to {output_path}")
        
        if compress:
            df.to_csv(output_path, index=False, compression='gzip')
        else:
            df.to_csv(output_path, index=False)
            
    elif format_type == 'parquet':
        filename = 'merged_data.parquet'
        output_path = outdir / filename
        
        # Check if file exists
        if output_path.exists() and not force:
            response = input(f"Output file {output_path} already exists. Overwrite? (y/N): ")
            if response.lower() != 'y':
                logger.info("Merge cancelled by user")
                return
        
        logger.info(f"Saving merged data to {output_path}")
        
        # Parquet has built-in compression
        compression = 'snappy' if compress else None
        df.to_parquet(output_path, index=False, compression=compression)
        
    elif format_type == 'hdf5':
        filename = 'merged_data.h5'
        output_path = outdir / filename
        
        # Check if file exists
        if output_path.exists() and not force:
            response = input(f"Output file {output_path} already exists. Overwrite? (y/N): ")
            if response.lower() != 'y':
                logger.info("Merge cancelled by user")
                return
        
        logger.info(f"Saving merged data to {output_path}")
        
        # Save with compression if requested
        complevel = 9 if compress else 0
        df.to_hdf(output_path, key='data', mode='w', complevel=complevel, complib='zlib')
    
    # Update existing metadata with merge information
    if metadata and not read_only:
        meta_file = outdir / 'meta.json'
        
        # Create merge information
        merge_info = {
            'merged_at': pd.Timestamp.now().isoformat(),
            'total_rows': len(df),
            'output_format': format_type,
            'compressed': compress,
            'sorted_by': sort_by if sort_by in df.columns else None,
            'merged_file': filename
        }
        
        # Add merge info to existing metadata
        updated_metadata = metadata.copy()
        updated_metadata['merge_info'] = merge_info
        
        # Save updated metadata back to meta.json
        with open(meta_file, 'w') as f:
            json.dump(updated_metadata, f, indent=2)
        
        logger.info(f"Updated metadata in {meta_file}")
    elif metadata and read_only:
        # In read-only mode, save metadata separately
        metadata_file = outdir / 'merge_metadata.json'
        
        # Create merge information
        merge_info = {
            'merged_at': pd.Timestamp.now().isoformat(),
            'total_rows': len(df),
            'output_format': format_type,
            'compressed': compress,
            'sorted_by': sort_by if sort_by in df.columns else None,
            'merged_file': filename
        }
        
        # Create separate metadata file with original + merge info
        combined_metadata = {
            'original_run_metadata': metadata,
            'merge_info': merge_info
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(combined_metadata, f, indent=2)
        
        logger.info(f"Saved merge metadata to {metadata_file} (read-only mode)")
    elif read_only:
        logger.info("Read-only mode: skipping metadata modification")
    
    logger.info(f"Successfully saved {len(df)} rows to {output_path}")


def main():
    """Main entry point for the merge tool."""
    args = parse_args()
    logger = setup_logging(args.verbose)
    
    try:
        logger.info(f"Starting bart merge for run directory: {args.rundir}")
        
        # Validate input directory
        if not args.rundir.exists():
            raise FileNotFoundError(f"Run directory not found: {args.rundir}")
        
        if not args.rundir.is_dir():
            raise NotADirectoryError(f"Run path is not a directory: {args.rundir}")
        
        # Set output directory
        if args.outdir:
            outdir = args.outdir
        else:
            outdir = args.rundir  # Output directly to the run directory
        
        logger.info(f"Output directory: {outdir}")  
        outdir.mkdir(parents=True, exist_ok=True)
        
        # Find batch files
        batch_files = find_batch_files(args.rundir, logger)
        
        # Load metadata
        metadata = load_metadata(args.rundir, logger)
        
        # Copy files.csv if merging to a different directory
        if outdir != args.rundir:
            copy_files_csv(args.rundir, outdir, args.force, logger)
        
        # Merge batch files
        merged_df = merge_batch_files(batch_files, logger)
        
        # Save merged data
        save_merged_data(
            merged_df, outdir, args.format, args.compress, 
            args.sort_by, args.force, metadata, args.read_only, logger
        )
        
        logger.info("Merge completed successfully!")
        
    except Exception as e:
        logger.error(f"Merge failed: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
