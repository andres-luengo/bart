import numpy as np
import pandas as pd

from pathlib import Path
import hdf5plugin # dumb
import h5py

import logging

from typing import Any

from numba import njit

COLUMNS = ('file',)

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
        self._logger = logging.getLogger(__name__ + '.batch')

        self.process_params = process_params

        self.batch = batch
        self.batch_num = batch_num

        self.save_path = outdir / f'batch_{batch_num:>03}.csv'
        self._logger.info(f'Creating {self.save_path}')
        pd.DataFrame(columns = COLUMNS).to_csv(self.save_path)
    
    def run(self):
        self._logger.info(f'Running on batch {self.batch_num}.')
        self._logger.debug(f'That is, {self.batch = }')
        for file in self.batch:
            df = FileJob(file, self.process_params).run()
            df.to_csv(self.save_path, header = False, mode = 'a')

# these are run serially within each process, but the OOP makes things neat
class FileJob:
    def __init__(self, file: Path, process_params: dict[str, Any]):
        self._logger = logging.getLogger(__name__ + '.file')

        self.file = h5py.File(file)
        self.data: h5py.Dataset = self.file['data'] #type: ignore

        self._logger.info(f'Opened {file}')
        self._logger.debug(f'...with header {dict(self.data.attrs)}')
                
        self.frequency_window_size = process_params['freq_window']
        self._num_frequency_blocks = int(np.ceil(self.data.shape[2] / self.frequency_window_size))
    
    def run(self):
        warm_indices = self.get_warm_indices()
        print(warm_indices)
        hits = []
        df = pd.DataFrame(hits, columns=COLUMNS)
        return df

    def get_warm_indices(self):
        indices = []
        for i in range(self._num_frequency_blocks):
            l_index = i * self.frequency_window_size
            r_index = (i + 1) * self.frequency_window_size
            
            t_index = self.data.shape[0] // 2

            block = self.data[t_index, 0, l_index:r_index]
            
            if (np.max(block) - np.median(block)) > (5 * np.std(block)):
                indices.append(l_index)
        
        return np.array(indices)

    def get_hot_indices(self):
        pass

    def __del__(self):
        if hasattr(self, 'file'):
            self.file.close()