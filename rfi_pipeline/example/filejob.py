"""
Example File Job Module
~~~~~~~~~~~~~~~~~~~~~~~

This file provides an example implementation of a filter-based signal finding algorithm
for use with the RFI Pipeline framework. This is intended as a reference implementation
to demonstrate how to create custom file processing functions that work with :class:`rfi_pipeline.RunManager`.

When running this package as a script (i.e. ``python -m rfi_pipeline`` or ``rfi-pipeline``)
it creates a :class:`~rfi_pipeline.manager.RunManager` with :attr:`FileJob.run_func` as the file_job.

Users are encouraged to create their own file processing functions based on their
specific requirements and data analysis needs.
"""

import hdf5plugin # dumb
import h5py

import numpy as np
import pandas as pd

import scipy

from pathlib import Path

import logging

from typing import Any, Callable
from os import PathLike

import time

import re

import warnings

from numba import njit
logging.getLogger('numba').setLevel(logging.WARNING)


class FileJob:
    """
    Example implementation for processing individual HDF5 files for RFI detection.
    
    This class demonstrates how to implement a file processor that works with the
    RFI Pipeline framework. It handles loading astronomical observation data from HDF5 files
    and applies a multi-stage statistical filtering algorithm to detect radio
    frequency interference. 
    
    This is provided as an example - users should create their own file processors
    based on their specific analysis requirements.
        
    Processing Pipeline:
        1. Load data in frequency blocks
        2. Apply warm significance filtering (sigma-based)
        3. Apply hot significance filtering (MAD-based)  
        4. Apply hotter significance filtering (SNR-based with sigma clipping)
        5. Extract frequency and kurtosis features
        
    For RunManager, prefer :meth:`with_params`, which returns a callable(file_path) configured
    with your parameters. Use :meth:`run_func` for one-off processing of a single file.
    """
    def __init__(self, process_params: dict[str, Any]):
        """
        Initialize a FileJob with processing parameters.

        This stores configuration such as frequency windows and significance thresholds.
        Use run(file) to process individual HDF5 files.

        Parameters
        ----------
        process_params : dict[str, Any]
            Dictionary of processing parameters. Keys include:
            - max_freq (float): Maximum frequency for channel selection.
            - min_freq (float): Minimum frequency for channel selection.
            - freq_window (int): Size of the frequency window for block processing.
            - warm_significance (float): Threshold for warm significance detection.
            - hot_significance (float): Threshold for hot significance detection.
            - hotter_significance (float): Threshold for hotter significance detection.
            - sigma_clip (float): Sigma clipping value for data cleaning.

        Notes
        -----
        This class opens the HDF5 file inside run(file) and does not keep it open between calls.
        Prefer the convenience helpers FileJob.run_func(file, params) for one-offs or
        FileJob.with_params(params) when using RunManager.
        """
        self._max_freq = process_params['max_freq']
        self._min_freq = process_params['min_freq']

        self._frequency_window_size = process_params['freq_window']

        self._warm_significance = process_params['warm_significance']
        self._hot_significance = process_params['hot_significance']
        self._hotter_significance = process_params['hotter_significance']

        self._sigma_clip = process_params['sigma_clip']
    
    def _read_file_header(self, file: PathLike):
        file = Path(file)

        m = re.search(r'\/([^\/]+)$', str(file))
        if m:
            short_file = m.group(1)
        else:
            short_file = file
        self._logger = logging.getLogger(f'{__name__} ({short_file})')
        
        if not m:
            self._logger.warning(f'Got a weird file name: {file}')

        self._file = h5py.File(file)
        self._data: h5py.Dataset = self._file['data'] #type: ignore
        self._fch1: float = self._data.attrs['fch1'] #type: ignore
        self._foff: float = self._data.attrs['foff'] #type: ignore
        self._nchans: float = self._data.attrs['nchans'] #type: ignore
        self._nfpc: int | None = self._data.attrs.get('nfpc') #type: ignore
        if self._nfpc is None:
            self._nfpc: int = 1<<20
        else:
            self._logger.info(f'Got nfpc = {self._nfpc}')
        

        self._logger.info(f'Opened {file}')
        self._logger.debug(f'...with header {dict(self._data.attrs)}')

        self._min_channel = 0
        if np.isfinite(self._max_freq):
            self._min_channel = round((self._max_freq - self._fch1) / self._foff)
            self._min_channel = min(max(self._min_channel, 0), self._data.shape[2])

        self._max_channel = self._data.shape[2]
        if np.isfinite(self._min_freq):
            self._max_channel = round((self._min_freq - self._fch1) / self._foff)
            self._max_channel = max(min(self._max_channel, self._data.shape[2]), 0)

        self._logger.debug(f'Running on {self._max_channel - self._min_channel} channels, {(self._max_channel - self._min_channel) / self._data.shape[2]:.2%} of the data.')

        self._num_even_frequency_blocks = int(np.ceil((self._max_channel - self._min_channel) / self._frequency_window_size))
        # overlapping blocks; last even block doesn't get an odd block
        self._num_frequency_blocks = self._num_even_frequency_blocks * 2 - 1
        

    def run(self, file: PathLike) -> list[dict[str, Any]]:
        """
        Run an already initialized FileJob.
        """
        self._read_file_header(file)
        try:
            start_time = time.perf_counter()
            
            filtered_block_l_indices = self._filter_blocks()
            hits = self._get_hits(filtered_block_l_indices)
            
            end_time = time.perf_counter()
            self._logger.info(f'Finished file! Took {end_time - start_time:.3g}s')
            return hits
        finally:
            try:
                self.close()
            except Exception:
                pass
    
    def __call__(self, file: PathLike) -> list[dict[str, Any]]: 
        """Same as :meth:`run`."""
        return self.run(file)
    
    def _filter_blocks(self) -> np.ndarray:
        """Returns the lower index for every block that passes the warm and hot index filters"""
        start_time = time.perf_counter()

        test_strip = self._data[
            self._data.shape[0] // 2, # middle time bin
            0, # drop instrument id dimension
            self._min_channel:self._max_channel
        ]

        self._logger.debug(f'Starting with {self._num_frequency_blocks} blocks.')
        warm_indices = self._get_warm_indices(test_strip)
        self._logger.debug(f'Got {len(warm_indices)} warm blocks.')
        hot_indices = self._get_hot_indices(test_strip, warm_indices)
        self._logger.debug(f'Got {len(hot_indices)} hot blocks.')
        hotter_indices = self._get_hotter_indices(test_strip, hot_indices)

        end_time = time.perf_counter()
        self._logger.info(f'Done filtering blocks, took {end_time - start_time :.3g}s.')
        self._logger.info(f'Found {len(hotter_indices)} hotter blocks.')
        return hotter_indices

    def _get_warm_indices(self, test_strip: np.ndarray):
        """
        Returns data indices (i.e. don't index into test_strip directly with these)
        """
        warm_indices = []
        for i in range(self._num_frequency_blocks):
            l_index = i * self._frequency_window_size//2
            r_index = l_index + self._frequency_window_size

            if r_index > len(test_strip): break
            
            block_strip = test_strip[l_index:r_index]
            self._smooth_dc_spike(block_strip, l_idx=(l_index + self._min_channel))
            # probably makes it prefer bright narrowband signals but too soon to sigma clip
            others = np.delete(block_strip, np.argmax(block_strip))

            strip_significance = (np.max(block_strip) - np.median(others)) / np.std(others)
            
            if strip_significance > self._warm_significance:
                warm_indices.append(l_index + self._min_channel)
        
        return np.array(warm_indices)

    def _get_hot_indices(self, test_strip: np.ndarray, warm_indices: np.ndarray):
        """
        Returns data indices (i.e. don't index into test_strip directly with these)
        """
        hot_indices = []
        for warm_index in warm_indices:
            warm_test_index = warm_index - self._min_channel
            l_index = warm_test_index
            r_index = warm_test_index + self._frequency_window_size
            if r_index > len(test_strip): break

            strip = test_strip[l_index:r_index]
            self._smooth_dc_spike(strip, l_idx=(l_index + self._min_channel))

            strip_significance = (np.max(strip) - np.median(strip)) / scipy.stats.median_abs_deviation(strip)

            if (strip_significance > self._hot_significance):
                hot_indices.append(l_index + self._min_channel)
        return np.array(hot_indices)
    
    def _get_hotter_indices(self, test_strip: np.ndarray, hot_indices: np.ndarray) -> np.ndarray:
        hotter_indices = []
        for hot_index in hot_indices:
            hot_test_index = hot_index - self._min_channel
            l_index = hot_test_index
            r_index = l_index + self._frequency_window_size
            if r_index > len(test_strip): break

            strip = test_strip[l_index:r_index]
            self._smooth_dc_spike(strip, l_idx=(l_index + self._min_channel))

            clipped, _, _ = scipy.stats.sigmaclip(strip, self._sigma_clip, self._sigma_clip)
            noise = np.std(clipped)
            baseline = np.mean(clipped)
            
            signal = np.max(strip) - baseline

            snr = signal/noise

            if snr >= self._hotter_significance:
                hotter_indices.append(l_index + self._min_channel)
        
        return np.array(hotter_indices)    

    def _smooth_dc_spike(self, block: np.ndarray, l_idx: int, axis: int = -1):
        """
        If there is a DC spike in `block`, replaces it with the average of the values to the left and right of it
        along the specified frequency axis. Otherwise, does nothing. Modifies `block` in place.
        """
        r_idx = l_idx + block.shape[axis]

        r_to_spike = (r_idx + (self._nfpc // 2)) % self._nfpc

        if r_to_spike == 0 or r_to_spike > block.shape[axis]:
            return

        spike_idx = block.shape[axis] - r_to_spike 

        # Prepare slices for all axes
        slicer = [slice(None)] * block.ndim
        slicer_left = slicer.copy()
        slicer_right = slicer.copy()
        slicer_spike = slicer.copy()

        slicer_left[axis] = spike_idx - 1
        slicer_right[axis] = spike_idx + 1
        slicer_spike[axis] = spike_idx

        block[tuple(slicer_spike)] = (
            block[tuple(slicer_left)] + block[tuple(slicer_right)]
        ) / 2
    
    def _get_hits(self, block_l_indices: np.ndarray) -> list[dict[str, Any]]:
        start_time = time.perf_counter()

        # for <1000, generally fast enough to read in specific blocks
        data = self._data
        if len(block_l_indices) >= 2**10:
            # for some reason, using zeros_like or something like that loads in the entire dataset
            try:
                data = np.full(self._data.shape, np.nan)
                data[..., self._min_channel:self._max_channel] = self._data[..., self._min_channel:self._max_channel]
            except np._core._exceptions._ArrayMemoryError as e: # type: ignore
                self._logger.warning(
                    'Got a memory error while trying to load data all at once. '
                    'Falling back to loading by blocks...', 
                    exc_info=True)
                data = self._data

        rows = []
        for block_l_index in block_l_indices:
            left = block_l_index
            right = block_l_index + self._frequency_window_size

            flags = []

            freq_array = np.linspace(
                self._fch1 + left * self._foff,
                self._fch1 + right * self._foff,
                num=self._frequency_window_size
            )
            load_start_time = time.perf_counter()
            block = data[:, 0, left:right]
            load_end_time = time.perf_counter()

            self._logger.debug(f'Done loading block, took {load_end_time - load_start_time:.3g}s.')

            self._smooth_dc_spike(block, block_l_index)

            fit_start_time = time.perf_counter()  
            params = self._fit_frequency_thresholds(freq_array, block)
            fit_end_time = time.perf_counter()
            self._logger.debug(f'Done threshold-based estimation, took {fit_end_time - fit_start_time:.3g}s.')
            
            nan_mask = np.isnan(params).any(axis=1)
            num_nan_fits = np.sum(nan_mask)
            
            if num_nan_fits > 0:
                flags.append(f'nan fits: {num_nan_fits}')
            
            valid_mask = ~nan_mask
            if np.any(valid_mask):
                means = params[valid_mask, 0]
                widths = params[valid_mask, 1]  # width in frequency units (already converted)
                amps = params[valid_mask, 2]
                noise_baselines = params[valid_mask, 3]

                # For threshold method, width is already in appropriate units
                # Convert back to bins for consistency with original code
                width_bins = widths / np.abs(self._foff)

                # P_signal = amp * std * sqrt(2pi)
                # P_noise = C * 'signal width' = C * 2 * sqrt(2ln2) * std
                # should check with steve...
                # snrs = amps * np.sqrt(np.pi / np.log(2)) / (2 * noises) 
                noises_ = []
                for time_idx in np.arange(block.shape[0])[valid_mask]:
                    clipped, _, _ = scipy.stats.sigmaclip(block[time_idx], low=3, high=3)
                    noises_.append(np.std(clipped))
                noises = np.array(noises_)
                
                snrs = amps / noises

                # a bit more resilient for having a handful of bad time slices
                mean = np.median(means)
                width = np.mean(width_bins) # use threshold-based width directly (in bins)
                snr = np.mean(snrs)

                max_snr = np.max(snrs)
                others = np.delete(snrs, np.argmax(snrs))
                if (
                    # too many fits failed
                    valid_mask.sum() < block.shape[0] // 2
                    # max is way bigger than others
                    or max_snr - np.median(others) > 5 * np.std(others)
                ):
                    flags.append('blip')

            else:
                mean = snr = width = np.nan
                flags.append(f'no valid fits')
                self._logger.warning(f'Could not fit any Gaussians to block at {block_l_index}')
                # consider just dropping the block at this point tbh

            # normalize to stop kurtosis from exploding
            block_normalized = (block - np.mean(block)) / np.std(block)
            kurtosis = scipy.stats.kurtosis(block_normalized.flat)

            rows.append({
                'frequency_index': block_l_index,
                'frequency': mean,
                'kurtosis': kurtosis,
                'snr': snr,
                'width': width,
                'flags': '|'.join(flag.replace('|', '') for flag in flags)
            })
        
        end_time = time.perf_counter()
        self._logger.debug(f'Done getting hits, took {end_time - start_time:.3g}s')

        return rows

    def _fit_frequency_thresholds(self, freq_array: np.ndarray, block: np.ndarray):
        """
        Applies threshold-based width estimation to each frequency slice of the block.
        Returns parameters for each slice, with dimensions (num_slices, 4).
        Failed estimations result in NaN parameters for that slice.
        """
        all_params = []
        for i in range(block.shape[0]):
            slice_data = block[i, :]
            try:
                # Get threshold-based parameters
                mean_freq_idx, width, amplitude, noise_floor = _threshold_based_width_estimation(slice_data)
                
                if np.isnan(mean_freq_idx):
                    params = np.full((4,), np.nan)
                else:
                    # Convert frequency index to actual frequency
                    mean_freq = freq_array[int(mean_freq_idx)]
                    # Convert width from bins to frequency units (like stdev in Gaussian)
                    width_freq = width * np.abs(self._foff)
                    params = np.array([mean_freq, width_freq, amplitude, noise_floor])
            except:
                # Return NaN parameters for failed estimations
                params = np.full((4,), np.nan)
            all_params.append(params)
        return np.array(all_params)
    
    def _index_to_frequency(self, index: int):
        return self._fch1 + index * self._foff

    def close(self):
        """Close the file handler used by this FileJob."""
        self._file.close()

    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb):
        try:
            self._file.close()
        except Exception:
            pass

    @staticmethod
    def run_func(file: PathLike, process_params: dict[str, Any]) -> list[dict[str, Any]]:
        """Convenience function to process a single file with given parameters.

        Creates a temporary FileJob, runs it on the provided file, and ensures resources
        are released.
        """
        job = FileJob(process_params)
        try:
            return job.run(file)
        finally:
            try:
                job.close()
            except Exception:
                pass

    @staticmethod
    def with_params(process_params: dict[str, Any]) -> Callable[[PathLike], list[dict[str, Any]]]:
        """Return a callable(file) suitable for RunManager that applies these params.

        Each call constructs a fresh FileJob to avoid holding file handles across invocations.
        """
        def _fn(file: PathLike) -> list[dict[str, Any]]:
            return FileJob.run_func(file, process_params)
        return _fn

@njit
def _threshold_based_width_estimation(spectrum):
    """
    Threshold-based width estimation
    
    Find noise floor using sigma clipping, calculate middle point between noise and peak,
    then find width where signal drops below this threshold.
    """
    data = spectrum.copy()
    for _ in range(16): 
        mean_val = np.mean(data)
        std_val = np.std(data)
        lower_bound = mean_val - 2.0 * std_val
        upper_bound = mean_val + 2.0 * std_val
        
        mask = (data >= lower_bound) & (data <= upper_bound)
        if np.sum(mask) == 0: 
            break
        data = data[mask]

    noise_floor = np.median(data)
    
    # Find peak
    peak_value = np.max(spectrum)
    peak_idx = np.argmax(spectrum)
    
    # Calculate threshold (middle point between noise and peak)
    threshold = (noise_floor + peak_value) / 2
    
    # Find points above threshold around the peak
    above_threshold = spectrum > threshold
    
    if not np.any(above_threshold):
        return np.nan, np.nan, np.nan, np.nan
    
    # Find the contiguous region around the peak (brightest signal)
    # Start from the peak and expand left and right while above threshold
    left_idx = peak_idx
    right_idx = peak_idx
    
    # Expand left from peak
    while left_idx > 0 and above_threshold[left_idx - 1]:
        left_idx -= 1
    
    # Expand right from peak  
    while right_idx < len(spectrum) - 1 and above_threshold[right_idx + 1]:
        right_idx += 1
    
    # Calculate width of this contiguous region
    width = right_idx - left_idx + 1
    
    # Return parameters similar to what Gaussian fit would return
    # (mean, width, amplitude, noise_floor)
    amplitude = peak_value - noise_floor
    
    return float(peak_idx), float(width), float(amplitude), float(noise_floor)