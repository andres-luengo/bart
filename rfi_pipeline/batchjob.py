import pandas as pd

from pathlib import Path

import logging

from typing import Any
from threading import Lock

import datetime as dt

from contextlib import contextmanager
import json
MAX_PROGRESS_LIST_LENGTH = 16

import os

from .filejob import FileJob

class BatchJob:
    def __init__(
            self, *,
            process_params: dict[str, Any],
            outdir: Path, 
            batch: tuple[Path, ...], 
            progress_lock: Lock,
            batch_num: int = -1,
    ):
        self._logger = logging.getLogger(f'{__name__} (batch {batch_num:>03})')

        self.process_params = process_params

        self.batch = batch
        self.batch_num = batch_num

        self.save_path = outdir / 'batches' / f'batch_{batch_num:>03}.csv'

        self._progress_lock = progress_lock
        self._progress_data_path = outdir / 'progress-data.json'

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
            try:
                df = FileJob(file, self.process_params).run()
            except Exception:
                self._logger.error(f'Something went wrong on file {file}!', exc_info=True)
                df = None
            else:
                df['source file'] = str(file)
                df.to_csv(self.save_path, header=keep_header, mode='a', index=False)
                if keep_header:
                    self._logger.info(f'Saved to {self.save_path}.')
                    keep_header = False

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
            if df is None: # something went wrong
                hit_counts.append(-1)
            elif df.iloc[0]['flags'] == 'EMPTY FILE':
                hit_counts.append(0)
            else:
                hit_counts.append(len(df))
            if len(hit_counts) > MAX_PROGRESS_LIST_LENGTH:
                hit_counts.pop(0)
    
    @contextmanager
    def get_progress_data(self):
        # context manager mania
        with self._progress_lock:
            with self._progress_data_path.open('r') as f:
                progress_data: list[dict[str, Any]] = json.load(f)

            yield progress_data

            with self._progress_data_path.open('w') as f:
                json.dump(progress_data, f, indent=4)