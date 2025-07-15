import numpy as np
import pandas as pd

import scipy

from pathlib import Path

import hdf5plugin # dumb
import h5py

import logging

from typing import Any

import time

import re

import warnings

import scipy.optimize

# things to think about:
# DC spikes
# numba

class BatchJob:
    def __init__(
            self,
            process_params: dict[str, Any],
            outdir: Path, 
            batch: tuple[Path, ...], 
            batch_num: int = -1
    ):
        self._logger = logging.getLogger(f'{__name__} (batch {batch_num:>03})')

        self.process_params = process_params

        self.batch = batch
        self.batch_num = batch_num

        self.save_path = outdir / f'batch_{batch_num:>03}.csv'
    
    def run(self):
        self._logger.info(f'Running on batch {self.batch_num}.')
        self._logger.debug(f'That is, {self.batch = }')
        for i, file in enumerate(self.batch):
            df = FileJob(file, self.process_params).run()
            df['source file'] = str(file)
            if i == 0:
                keep_header = True
                self._logger.info(f'Saving to {self.save_path}')
            else:
                keep_header = False
            df.to_csv(self.save_path, header = keep_header, mode = 'a', index = False)

# these are run serially within each process, but the OOP makes things neat
class FileJob:
    def __init__(self, file: Path, process_params: dict[str, Any]):
        m = re.search(r'\/([^\/]+)$', str(file))
        if m:
            short_file = m.group(1)
        else:
            short_file = file
        self._logger = logging.getLogger(f'{__name__} ({short_file})')
        
        if not m:
            self._logger.warning(f'Got a weird file name: {file}')
        
        if 'spliced' in str(file):
            raise NotImplementedError('Spliced files are not supported.')

        self._file = h5py.File(file)
        self._data: h5py.Dataset = self._file['data'] #type: ignore
        self._fch1: float = self._data.attrs['fch1'] #type: ignore
        self._foff: float = self._data.attrs['foff'] #type: ignore
        self._nchans: float = self._data.attrs['nchans'] #type: ignore

        self._logger.info(f'Opened {file}')
        self._logger.debug(f'...with header {dict(self._data.attrs)}')
                
        self._frequency_window_size = process_params['freq_window']
        self._warm_significance = process_params['warm_significance']
        self._hot_significance = process_params['hot_significance']
        self._num_even_frequency_blocks = int(np.ceil(self._data.shape[2] / self._frequency_window_size))
        # overlapping blocks; last even block doesn't get an odd block
        self._num_frequency_blocks = self._num_even_frequency_blocks * 2 - 1
    
    def run(self):
        start_time = time.perf_counter()
        filtered_block_l_indices = self.filter_blocks()
        hits = self.get_hits(filtered_block_l_indices)
        df = pd.DataFrame(hits)
        end_time = time.perf_counter()
        self._logger.debug(f'Done! Took {end_time - start_time:.3g}s')
        return df
    
    def filter_blocks(self) -> np.ndarray:
        """Returns the lower index for every block that passes the warm and hot index filters"""
        test_strip = self._data[
            self._data.shape[0] // 2, # middle time bin
            0, # drop instrument id dimension
            : # all frequency bins
        ]

        warm_indices = self.get_warm_indices(test_strip)
        hot_indices = self.get_hot_indices(test_strip, warm_indices)

        return hot_indices

    def get_warm_indices(self, test_strip: np.ndarray):
        warm_indices = []
        for i in np.arange(0, self._num_even_frequency_blocks, 0.5):
            l_index = int(i * self._frequency_window_size)
            r_index = int((i + 1) * self._frequency_window_size)
            
            block_strip = test_strip[l_index:r_index]
            self.smooth_dc_spike(block_strip, l_idx=l_index)
            
            if (np.max(block_strip) - np.median(block_strip)) > (self._warm_significance * np.std(block_strip)):
                warm_indices.append(l_index)
        
        return np.array(warm_indices)

    def get_hot_indices(self, test_strip: np.ndarray, warm_indices: np.ndarray):
        hot_indices = []
        for warm_index in warm_indices:
            l_index = warm_index
            r_index = warm_index + self._frequency_window_size
            strip = test_strip[l_index:r_index]

            if (
                np.max(strip) - np.median(strip)
                > self._hot_significance * scipy.stats.median_abs_deviation(strip)
            ):
                hot_indices.append(l_index)
        return np.array(hot_indices)
    

    def smooth_dc_spike(self, block: np.ndarray, l_idx: int, axis: int = -1):
        """
        If there is a DC spike in `block`, replaces it with the average of the values to the left and right of it
        along the specified frequency axis. Otherwise, does nothing. Modifies `block` in place.
        """
        FINE_PER_COARSE = 1_048_576
        r_idx = l_idx + self._frequency_window_size

        r_to_spike = (r_idx + (FINE_PER_COARSE // 2)) % FINE_PER_COARSE

        if r_to_spike == 0 or r_to_spike > self._frequency_window_size:
            return

        spike_idx = (r_idx - r_to_spike) - l_idx

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
    
    def get_hits(self, block_l_indices: np.ndarray) -> list[dict[str, Any]]:
        rows = []
        for block_l_index in block_l_indices:
            left = block_l_index
            right = block_l_index + self._frequency_window_size

            flags = []

            freq_array = np.linspace(
                self._fch1 + left * self._foff,
                self._fch1 + right * self._foff,
                num = right - left
            )

            block = self._data[:, 0, left:right]
            self.smooth_dc_spike(block, block_l_index)
            
            params, _ = self.fit_frequency_gaussians(freq_array, block)
            
            # Count NaN fits
            nan_mask = np.isnan(params).any(axis=1)
            num_nan_fits = np.sum(nan_mask)
            
            if num_nan_fits > 0:
                flags.append(f'nan fits: {num_nan_fits}')
            
            # Calculate statistics using only non-NaN values
            valid_mask = ~nan_mask
            if np.any(valid_mask):
                means = params[valid_mask, 0]
                stds = params[valid_mask, 1]
                amps = params[valid_mask, 2]
                noises = params[valid_mask, 3]

                mean = np.mean(means)
                width = 2.355 * np.mean(stds) # stdev to FWHM
                snr = (amps / noises).mean()
            else:
                mean = snr = width = np.nan
                # consider just dropping the block at this point tbh

            # need it so kurtosis doesn't blow up
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
        return rows
    
    @staticmethod
    def signal_model(x, mean, stdev, amplitude, noise):
        exponent = -0.5 * ((x - mean) / stdev)**2
        return noise + amplitude * np.exp(exponent)

    def fit_frequency_gaussians(self, freq_array: np.ndarray, block: np.ndarray):
        """
        Fits a Gaussian to each frequency slice of the block.
        Returns parameters and covariance for each slice, with dimensions (num_slices, 4) and (num_slices, 4, 4).
        Failed fits result in NaN parameters for that slice."""
        all_params = []
        all_covs = []
        for i in range(block.shape[0]):
            slice_data = block[i, :]
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    params, cov = self.fit_gaussian_to_slice(freq_array, slice_data)
            except (RuntimeError, ValueError, scipy.optimize.OptimizeWarning):
                # Return NaN parameters for failed fits
                params = np.full((4,), np.nan)
                cov = np.full((4, 4), np.nan)
            all_params.append(params)
            all_covs.append(cov)
        params = np.array(all_params)
        covs = np.array(all_covs)
        return params, covs
    
    def fit_gaussian_to_slice(self, freq_array: np.ndarray, slice_data: np.ndarray):
        return scipy.optimize.curve_fit(
            self.signal_model, 
            freq_array, 
            slice_data, 
            p0=[
                freq_array[np.argmax(slice_data)], 
                np.abs(self._foff),
                np.max(slice_data) - np.median(slice_data),
                np.median(slice_data)
            ],
            # slow, screws up scale for some signals
            bounds=np.array([
                (freq_array[-1], freq_array[0]),
                (0, np.inf),
                (np.median(slice_data)/2, np.inf),
                (0, np.inf)
            ]).T,
            max_nfev=10_000
        )

    def index_to_frequency(self, index: int):
        return self._fch1 + index * self._foff

    def __del__(self):
        if hasattr(self, 'file'):
            self._file.close()