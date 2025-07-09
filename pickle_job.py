import numpy as np
import pandas as pd

import pathlib

import logging

class PickleJob:
    def __init__(
            self, 
            outdir: pathlib.Path, 
            batch: list[str], 
            batch_num: int = -1,
    ):
        self._logger = logging.getLogger(__name__)

        self.batch = batch
        self.batch_num = batch_num

        self.feature_table = pd.DataFrame(columns = [
            'frequency', 'file'
        ])

        # if you have more than 999 batches, you have bigger problems than the file names being out of order
        self.save_path = outdir / f'batch_{batch_num:>03}.csv'
        self._logger.info(f'Creating {self.save_path}')
        self.feature_table.to_csv(self.save_path, index = False)
    
    def run(self):
        return self.feature_table