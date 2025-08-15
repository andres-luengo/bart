"""
Batch Job Module (BART)

This module provides the BatchJob class for processing batches of files
in parallel worker processes, handling progress tracking and result storage.
"""

import pandas as pd

from pathlib import Path

import logging

from typing import Any, Callable, Sequence, Iterable
from threading import Lock

import datetime as dt

from contextlib import contextmanager
import json
MAX_PROGRESS_LIST_LENGTH = 64

import os

import hdf5plugin  # ensure h5py import works
import h5py

from numpy import ndarray

PandasData = ndarray | Iterable | dict | pd.DataFrame
class BatchJob:
    """
    Processes a batch of files within a single worker process.
    
    The BatchJob class manages the processing of a subset of files assigned to
    a worker process. It handles individual file processing, progress tracking,
    result accumulation, and saving batch results to CSV files.
    
    Attributes:
        file_job: Function to process individual files
        batch: List of files in this batch
        batch_num: Numeric identifier for this batch
        save_path: Path where batch results will be saved
        files_csv_path: Path to the files metadata CSV
    """
    def __init__(
            self, *,
            file_job: Callable[[os.PathLike], PandasData],
            outdir: Path, 
            batch: Sequence[Path], 
            meta_lock: Lock,
            batch_num: int = -1,
            continue_on_exception = False
    ):
        self.file_job = file_job
        self._logger = logging.getLogger(f'{__name__} (batch {batch_num:>03})')

        self.batch = batch
        self.batch_num = batch_num

        self.save_path = outdir / 'batches' / f'batch_{batch_num:>03}.csv'
        self.files_csv_path = outdir / 'files.csv'

        self._meta_lock = meta_lock
        self._progress_data_path = outdir / 'progress-data.json'

        self._continue_on_exception = continue_on_exception

        with self.get_progress_data() as progress_data:
            batch_data = progress_data[self.batch_num]
            batch_data['worker pid'] = os.getpid()
            batch_data['batch size'] = len(self.batch)
            batch_data['num complete'] = 0
            batch_data['last file end time'] = dt.datetime.now(dt.timezone.utc).isoformat()
    
    def run(self):
        self._logger.info(f'Running on batch {self.batch_num}.')
        self._logger.debug(f'That is, {self.batch = }')
        keep_header = not self.save_path.is_file()
        for i, file in enumerate(self.batch):
            df = None
            file_info = None
            
            try:
                # Extract file header information
                file_info = self._extract_file_info(file)
                data = self.file_job(file)
                df = pd.DataFrame(data)
            except Exception as e:
                self._logger.error(f'Something went wrong on file {file}!', exc_info=True)
                df = None
                if not self._continue_on_exception: raise e
            else:
                if not df.empty:
                    df['source file'] = str(file)
                    df.to_csv(self.save_path, header=keep_header, mode='a', index=False)
                    if keep_header:
                        self._logger.info(f'Saved to {self.save_path}.')
                        keep_header = False

            # Update files.csv with file information
            self._update_files_csv(file, df, file_info)
            self._filejob_update_progress(i, df)
        
        with self.get_progress_data() as progress_data:
            del progress_data[self.batch_num]['worker pid']
            progress_data[self.batch_num]['num complete'] = len(self.batch)
    
    def _filejob_update_progress(self, i: int, df: pd.DataFrame | None):
        with self.get_progress_data() as progress_data:
            batch_progress = progress_data[self.batch_num]
            
            batch_progress['num complete'] = i + 1
            
            if 'times elapsed' not in batch_progress:
                batch_progress['times elapsed'] = []
            now = dt.datetime.now(dt.timezone.utc)
            last_job = dt.datetime.fromisoformat(batch_progress['last file end time'])
            this_job_time_elapsed = (now - last_job).total_seconds()
            times_elapsed = batch_progress['times elapsed']
            times_elapsed.append(this_job_time_elapsed)
            if len(times_elapsed) > MAX_PROGRESS_LIST_LENGTH:
                times_elapsed.pop(0)
    
            batch_progress['last file end time'] = now.isoformat()
            
            if 'hit counts' not in batch_progress: 
                batch_progress['hit counts'] = []
            hit_counts: list[int] = batch_progress['hit counts']
            if df is None:  # error
                hit_counts.append(-1)
            elif df.empty:  # no results
                hit_counts.append(0)
            else:  # normal results
                hit_counts.append(len(df))
            if len(hit_counts) > MAX_PROGRESS_LIST_LENGTH:
                hit_counts.pop(0)
    
    def _extract_file_info(self, file: Path) -> dict[str, Any]:
        """Extract header information from the file."""
        result = {
            'ra': None,
            'dec': None,
            'tstart': None,
            'nchans': None,
            'foff': None,
            'fch1': None,
            'flch': None
        }
        
        try:
            with h5py.File(file, 'r') as f:
                data = f['data']
                attrs = dict(data.attrs)
                
                # Extract values, handling various HDF5 attribute types
                # Map header keys to our result keys
                key_mapping = {
                    'src_raj': 'ra',
                    'src_dej': 'dec',
                    'tstart': 'tstart',
                    'nchans': 'nchans',
                    'foff': 'foff',
                    'fch1': 'fch1'
                }
                
                for header_key, result_key in key_mapping.items():
                    try:
                        value = attrs.get(header_key)
                        if value is not None:
                            # Convert to Python native type
                            if hasattr(value, 'item'):
                                result[result_key] = value.item()  # type: ignore
                            else:
                                result[result_key] = value  # type: ignore
                    except Exception:
                        result[result_key] = None
                
                # Calculate flch (fch1 + foff * nchans)
                try:
                    if all(result[k] is not None for k in ['fch1', 'foff', 'nchans']):
                        result['flch'] = result['fch1'] + result['foff'] * result['nchans']  # type: ignore
                except Exception:
                    result['flch'] = None
                        
        except Exception as e:
            self._logger.error(f'Failed to extract file info from {file}: {e}')
            
        return result
    
    def _update_files_csv(self, file: Path, df: pd.DataFrame | None, file_info: dict[str, Any] | None):
        """Update the files.csv with information about the processed file."""
        # Determine number of hits
        num_hits = -1  # Default for failed processing
        if df is not None:
            num_hits = 0 if df.empty else len(df)
        
        # Get current time in ISO format (UTC timezone)
        iso_time_completed = dt.datetime.now(dt.timezone.utc).isoformat()
        
        # Create row data
        row_data = {
            'file': str(file),
            'num_hits': num_hits,
            'time_completed': iso_time_completed
        }
        
        # Add header information if available
        if file_info is not None:
            row_data.update({
                'RA': file_info['ra'],
                'DEC': file_info['dec'],
                'tstart': file_info['tstart'],
                'nchans': file_info['nchans'],
                'foff': file_info['foff'],
                'fch1': file_info['fch1'],
                'flch': file_info['flch']
            })
        else:
            # Add None values for header columns
            row_data.update({
                'RA': None,
                'DEC': None,
                'tstart': None,
                'nchans': None,
                'foff': None,
                'fch1': None,
                'flch': None
            })
        
        # Write to files.csv (thread-safe)
        with self._meta_lock:
            files_csv_exists = self.files_csv_path.is_file()
            row_df = pd.DataFrame([row_data])
            row_df.to_csv(self.files_csv_path, header=not files_csv_exists, mode='a', index=False)
    
    @contextmanager
    def get_progress_data(self):
        # context manager mania
        with self._meta_lock:
            with self._progress_data_path.open('r') as f:
                progress_data: list[dict[str, Any]] = json.load(f)

            yield progress_data

            with self._progress_data_path.open('w') as f:
                json.dump(progress_data, f, indent=4)
