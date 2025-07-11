import numpy as np
import pandas as pd

import scipy.stats

from pathlib import Path

import hdf5plugin # dumb
import h5py

import logging

from typing import Any

import time

HIT_COLUMNS = ('frequency', 'kurtosis',) # sure

# things to think about:
# (!!!) overlapping windows
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
        self._logger.info(f'Creating {self.save_path}')
        pd.DataFrame(columns = HIT_COLUMNS + ('source file',)).to_csv(self.save_path, index = False)
    
    def run(self):
        self._logger.info(f'Running on batch {self.batch_num}.')
        self._logger.debug(f'That is, {self.batch = }')
        for file in self.batch:
            df = FileJob(file, self.process_params).run()
            df['source file'] = str(file)
            df.to_csv(self.save_path, header = False, mode = 'a', index = False)

# these are run serially within each process, but the OOP makes things neat
class FileJob:
    def __init__(self, file: Path, process_params: dict[str, Any]):
        self._logger = logging.getLogger(f'{__name__} ({file})')

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
    
    def get_hits(self, block_l_indices: np.ndarray) -> list[dict[str, Any]]:
        rows = []
        for block_l_index in block_l_indices:
            left = block_l_index
            right = block_l_index + self._frequency_window_size

            block = self._data[:, 0, left:right]

            # otherwise 
            block_normalized = (block - np.mean(block)) / np.std(block)

            rows.append({
                'frequency': self.index_to_frequency((left + right) / 2),
                'kurtosis': scipy.stats.kurtosis(block_normalized.flat)
            })
        return rows
    
    def index_to_frequency(self, index: int):
        return self._fch1 + index * self._foff

    def __del__(self):
        if hasattr(self, 'file'):
            self._file.close()