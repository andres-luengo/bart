import numpy as np
import pandas as pd

from pathlib import Path

import logging

COLUMNS = ('file',)

class PickleJob:
    def __init__(
            self, 
            outdir: Path, 
            batch: tuple[Path, ...], 
            batch_num: int = -1,
    ):
        self._logger = logging.getLogger(__name__)

        self.batch = batch
        self.batch_num = batch_num

        self.save_path = outdir / f'batch_{batch_num:>03}.csv'
        self._logger.info(f'Creating {self.save_path}')
        pd.DataFrame(columns = COLUMNS).to_csv(self.save_path)
    
    def run(self):
        self._logger.info(f'Running on batch {self.batch_num}.')
        self._logger.debug(f'That is, {self.batch = }')
        self.process_file(self.batch[0])
    
    def process_file(self, file: Path):
        raise ValueError('explode :(')