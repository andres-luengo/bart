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

from numba import njit
logging.getLogger('numba').setLevel(logging.WARNING)

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
        keep_header = True
        for i, file in enumerate(self.batch):
            try:
                df = FileJob(file, self.process_params).run()
            except Exception:
                self._logger.error(f'Something went wrong on file {file}!', exc_info=True)
            else:
                df['source file'] = str(file)
                df.to_csv(self.save_path, header=keep_header, mode='a', index=False)
                if keep_header:
                    self._logger.info(f'Saved to {self.save_path}.')
                    keep_header = False

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

        self._min_channel = 0
        if np.isfinite(process_params['max_freq']):
            self._min_channel = round((process_params['max_freq'] - self._fch1) / self._foff)
            self._min_channel = max(self._min_channel, 0)

        self._max_channel = self._data.shape[2]
        if np.isfinite(process_params['min_freq']):
            self._max_channel = round((process_params['min_freq'] - self._fch1) / self._foff)
            self._max_channel = min(self._max_channel, self._data.shape[2])

        self._logger.debug(f'Running on {self._max_channel - self._min_channel} channels, {(self._max_channel - self._min_channel) / self._data.shape[2]:.2%} of the data.')
                
        self._frequency_window_size = process_params['freq_window']
        self._warm_significance = process_params['warm_significance']
        self._hot_significance = process_params['hot_significance']
        self._num_even_frequency_blocks = int(np.ceil((self._max_channel - self._min_channel) / self._frequency_window_size))
        # overlapping blocks; last even block doesn't get an odd block
        self._num_frequency_blocks = self._num_even_frequency_blocks * 2 - 1
    
    def run(self):
        start_time = time.perf_counter()
        
        filtered_block_l_indices = self.filter_blocks()
        hits = self.get_hits(filtered_block_l_indices)
        df = pd.DataFrame(hits)
        
        end_time = time.perf_counter()
        self._logger.info(f'Finished file! Took {end_time - start_time:.3g}s')
        return df
    
    def filter_blocks(self) -> np.ndarray:
        """Returns the lower index for every block that passes the warm and hot index filters"""
        start_time = time.perf_counter()

        test_strip = self._data[
            self._data.shape[0] // 2, # middle time bin
            0, # drop instrument id dimension
            self._min_channel:self._max_channel
        ]

        warm_indices = self.get_warm_indices(test_strip)
        hot_indices = self.get_hot_indices(test_strip, warm_indices)

        end_time = time.perf_counter()
        self._logger.debug(f'Done filtering blocks, took {end_time - start_time :.3g}s.')
        self._logger.info(f'Found {len(hot_indices)} interesting blocks.')

        return hot_indices

    def get_warm_indices(self, test_strip: np.ndarray):
        """
        Returns data indices (i.e. don't index into test_strip directly with these)
        """
        warm_indices = []
        for i in range(self._num_frequency_blocks):
            l_index = i * self._frequency_window_size//2
            r_index = l_index + self._frequency_window_size

            if r_index > len(test_strip): break
            
            block_strip = test_strip[l_index:r_index]
            self.smooth_dc_spike(block_strip, l_idx=(l_index + self._min_channel))
            
            if (np.max(block_strip) - np.median(block_strip)) > (self._warm_significance * np.std(block_strip)):
                warm_indices.append(l_index + self._min_channel)
        
        return np.array(warm_indices)

    def get_hot_indices(self, test_strip: np.ndarray, warm_indices: np.ndarray):
        """
        Returns data indices (i.e. don't index into test_strip directly with these)
        """
        hot_indices = []
        for warm_index in warm_indices:
            warm_test_index = warm_index - self._min_channel
            l_index = warm_test_index
            r_index = warm_test_index + self._frequency_window_size
            strip = test_strip[l_index:r_index]
            self.smooth_dc_spike(strip, l_idx=(l_index + self._min_channel))

            if r_index > len(test_strip): break

            if (
                np.max(strip) - np.median(strip)
                > self._hot_significance * scipy.stats.median_abs_deviation(strip)
            ):
                hot_indices.append(l_index + self._min_channel)
        return np.array(hot_indices)
    

    def smooth_dc_spike(self, block: np.ndarray, l_idx: int, axis: int = -1):
        """
        If there is a DC spike in `block`, replaces it with the average of the values to the left and right of it
        along the specified frequency axis. Otherwise, does nothing. Modifies `block` in place.
        """
        FINE_PER_COARSE = 1_048_576
        r_idx = l_idx + block.shape[axis]

        r_to_spike = (r_idx + (FINE_PER_COARSE // 2)) % FINE_PER_COARSE

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
    
    def get_hits(self, block_l_indices: np.ndarray) -> list[dict[str, Any]]:
        start_time = time.perf_counter()

        # for <1000, generally fast enough to read in specific blocks
        if len(block_l_indices) < 2**10:
            data = self._data
        # otherwise, i think its faster to just load in the whole thing (sorry RAM)
        else:
            # for some reason, using zeros_like or something like that loads in the entire dataset
            data = np.full(self._data.shape, np.nan)
            data[..., self._min_channel:self._max_channel] = self._data[..., self._min_channel:self._max_channel]

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

            block = data[:, 0, left:right]
            self.smooth_dc_spike(block, block_l_index)
            
            params, _ = self.fit_frequency_gaussians(freq_array, block)
            
            nan_mask = np.isnan(params).any(axis=1)
            num_nan_fits = np.sum(nan_mask)
            
            if num_nan_fits > 0:
                flags.append(f'nan fits: {num_nan_fits}')
            
            valid_mask = ~nan_mask
            if np.any(valid_mask):
                means = params[valid_mask, 0]
                stds = params[valid_mask, 1]
                amps = params[valid_mask, 2]
                noises = params[valid_mask, 3]

                # P_signal = amp * std * sqrt(2pi)
                # P_noise = C * 'signal width' = C * 2 * sqrt(2ln2) * std
                # should check with steve...
                snrs = amps * np.sqrt(np.pi / np.log(2)) / (2 * noises)

                mean = np.mean(means)
                width = 2.355 * np.mean(stds) # stdev to FWHM
                snr = np.mean(snrs)

                if (
                    len(snrs) < 4
                    or (
                        np.max(snrs) - np.median(snrs) 
                        > 10 * scipy.stats.median_abs_deviation(snrs)
                    )
                ):
                    flags.append('blip')

            else:
                mean = snr = width = np.nan
                flags.append(f'no valid fits')
                self._logger.warning(f'Could not fit any Gaussians to block at {block_l_index}')
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
        
        end_time = time.perf_counter()
        self._logger.debug(f'Done getting hits, took {end_time - start_time:.3g}s')

        return rows

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
            signal_model, 
            freq_array, 
            slice_data, 
            p0=[
                freq_array[np.argmax(slice_data)], 
                np.abs(self._foff),
                np.max((np.max(slice_data) - np.median(slice_data), np.std(slice_data))),
                np.median(slice_data)
            ],
            # slow, screws up scale for some signals
            # can't really know if we got a fit that makes any sense otherwise...
            bounds=np.array([
                (freq_array[-1], freq_array[0]),
                (0, np.inf),
                (np.median(slice_data) * 0.25, np.inf),
                (0, np.inf)
            ]).T,
            max_nfev=10_000
        )

    def index_to_frequency(self, index: int):
        return self._fch1 + index * self._foff

    def __del__(self):
        if hasattr(self, '_file'):
            self._file.close()

@njit
def signal_model(x, mean, stdev, amplitude, noise):
    exponent = -0.5 * ((x - mean) / stdev)**2
    return noise + amplitude * np.exp(exponent)