"""
Run Manager Module
~~~~~~~~~~~~~~~~~~

This module provides the RunManager class for coordinating parallel RFI detection
across multiple files and batches.
"""

import pandas as pd

from .batchjob import BatchJob

from argparse import Namespace
from pathlib import Path

import multiprocessing as mp

import logging, logging.handlers

import json

from typing import Any, Callable, Sequence
from os import PathLike

import resource

import datetime as dt


class RunManager:
    """
    Manages the execution of RFI detection across multiple files and processes.
    
    The RunManager coordinates parallel processing by dividing input files into
    batches and distributing them across multiple worker processes. It handles
    resource management, progress tracking, and result consolidation.
    
    Attributes:
        file_job: Function to process individual files
        process_params: Parameters for the file processing function
        num_processes: Number of parallel worker processes
        batches: List of file batches for processing
        outdir: Output directory for results
        max_rss: Maximum memory usage in bytes
    """
    def __init__(
            self,
            file_job: Callable[[PathLike, dict[str, Any]], pd.DataFrame],
            process_params: dict[str, Any], 
            files: Sequence[PathLike],
            outdir: PathLike,
            num_batches: int = 1, 
            num_processes: int = 1,
            max_rss: int = 8,
            log_level: int = logging.WARNING,
            continue_on_exception=False
    ):
        """
        Initialize the RunManager.
        
        Args:
            file_job: Function to process individual files, should return DataFrame
            process_params: Dictionary of parameters for the file processing function
            num_batches: Number of batches to divide files into
            num_processes: Number of parallel worker processes
            files: Sequence of file paths to process
            outdir: Output directory for results and metadata
            max_rss: Maximum memory usage in bytes
        """
        self.file_job = file_job
        self.process_params = process_params
        
        self.num_processes = num_processes
        self.batches = self._make_batches(files, num_batches)
        
        self.outdir = Path(outdir)
        self.outdir.mkdir(exist_ok=True)
        (self.outdir / 'batches').mkdir(exist_ok=True)
        (self.outdir / 'logs').mkdir(exist_ok=True)
        self._save_targets_copy(files)

        self._logger = logging.getLogger(__name__)
        self._logging_setup(level=log_level)
        
        self.max_rss = max_rss
        self.continue_on_exception = continue_on_exception

        self._setup_meta()

        if (self.outdir / 'progress-data.json').is_file():
            self._clean_progress_file()
        else:
            self._setup_new_progress_file()

        self._completed_files = self._get_completed_files()
    
    def _save_targets_copy(self, targets: Sequence[PathLike]):
        with (self.outdir / 'target-list.txt').open('w') as f:
            f.writelines(str(path) for path in targets)

    def _logging_setup(self, level=logging.WARNING):
        (self.outdir / 'logs').mkdir(exist_ok=True)

        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s (%(process)d): %(message)s'
        )

        stderr_handler = logging.StreamHandler()
        stderr_handler.setLevel(level)
        stderr_handler.setFormatter(formatter)

        file_handler = logging.handlers.RotatingFileHandler(
            filename = self.outdir / 'logs' / 'all_logs.log',
            maxBytes = 2**20, # 1 MiB
            backupCount = 3, # so max of 4 MiB,
        )
        file_handler.setLevel(min(logging.INFO, level))
        file_handler.setFormatter(formatter)

        # if error files long enough that this is a problem, there are bigger ones
        err_file_handler = logging.FileHandler(
            filename = self.outdir / 'logs' / 'error_logs.log'
        )
        err_file_handler.setLevel(logging.ERROR)
        err_file_handler.setFormatter(formatter)

        self._logger.addHandler(stderr_handler)
        self._logger.addHandler(file_handler)
        self._logger.addHandler(err_file_handler)
        self._logger.setLevel(logging.DEBUG)

    def _make_batches(
            self, all_files: Sequence[PathLike], num_batches: int
    ) -> Sequence[Sequence[PathLike]]:
        batches = []
        for batch_idx in range(num_batches):
            batch = []
            for file in all_files[batch_idx::num_batches]:
                batch.append(Path(file))
            batches.append(tuple(batch))
        return tuple(batches)
        
    def _setup_meta(self):
        meta = {
            'start_time': dt.datetime.now(dt.timezone.utc).isoformat(),
            'outdir': str(self.outdir)
        } | self.process_params
        try:
            with open(self.outdir / 'meta.json', 'w') as f:
                json.dump(meta, f, indent=4)
        except TypeError as e:
            raise TypeError('Keys and values in process_params must be json-serializable.') from e
    
    def _setup_new_progress_file(self):
        progress_data = [
            {'batch size': len(batch)}
            for batch in self.batches
        ]
        with open(self.outdir / 'progress-data.json', 'w') as f:
            json.dump(progress_data, f, indent=4)
    
    def _clean_progress_file(self):
        data_path = self.outdir / 'progress-data.json'
        with data_path.open('r') as f:
            data = json.load(f)
        
        for batch_info in data:
            if 'worker pid' in batch_info:
                del batch_info['worker pid']
        
        with data_path.open('w') as f:
            json.dump(data, f)
    
    def _get_completed_files(self):
        completed_files: list[Path] = []
        for batch_csv_path in (self.outdir / 'batches').iterdir():
            df = pd.read_csv(batch_csv_path, usecols=['source file'])
            completed_files += list(map(Path, df['source file'].unique()))
        return set(completed_files)

    @classmethod
    def _from_namespace(cls, filejob: Callable[[PathLike, dict[str, Any]], pd.DataFrame], arg: Namespace, files: tuple[Path, ...]):
        return cls(
            filejob,
            # process_params = {
            #     'freq_window': arg.frequency_block_size,
            #     'warm_significance': arg.warm_significance,
            #     'hot_significance': arg.hot_significance,
            #     'hotter_significance': arg.hotter_significance,
            #     'sigma_clip': arg.sigma_clip,
            #     'min_freq': arg.min_freq,
            #     'max_freq': arg.max_freq
            # },
            process_params=vars(arg),
            num_batches=arg.num_batches,
            num_processes=arg.num_processes,
            files=files,
            outdir=arg.outdir,
            max_rss=arg.max_rss_gb * 1e9,
        )
    
    @staticmethod
    def _worker_init(log_queue: mp.Queue, max_memory: int):
        handler = logging.handlers.QueueHandler(log_queue)
        logger = logging.getLogger()
        logger.handlers.clear()
        logger.addHandler(handler)

        logger.info(f'Initializing worker with {max_memory/1e9:.1f}GB')
        resource.setrlimit(resource.RLIMIT_AS, (max_memory, max_memory))
    
    @staticmethod
    def _batch_job(kwargs: dict):
        try:
            job = BatchJob(**kwargs)
            return job.run()
        except Exception as e:
            id: int = kwargs['batch_num']
            logging.critical(
                f'EXCEPTION ON BATCH {id:<03}', 
                exc_info = True, stack_info = True
            )
            raise e
    
    def run(self):
        self._logger.info('Starting jobs...')

        with mp.Manager() as mp_manager:
            log_queue = mp_manager.Queue()
            meta_lock = mp_manager.Lock()

            batch_args = (
                {
                    'file_job': self.file_job,
                    'batch': tuple(file for file in batch if file not in self._completed_files),
                    'batch_num': i,
                    'process_params': self.process_params,
                    'outdir': self.outdir,
                    'meta_lock': meta_lock,
                    'continue_on_exception': self.continue_on_exception
                }
                for i, batch in enumerate(self.batches)
            )

            
            worker_listener = logging.handlers.QueueListener(
                log_queue,
                *logging.getLogger().handlers,
                respect_handler_level = True
            )
            worker_listener.start()

            with mp.Pool(
                processes=self.num_processes,
                initializer=self._worker_init,
                initargs=(
                    log_queue, 
                    int(self.max_rss // self.num_processes)
                )
            ) as p:
                # using this over map so that if anything raises it gets sent up ASAP
                results = p.imap_unordered(
                    self._batch_job, batch_args, chunksize=1
                )
                for _ in results: pass # consume lazy map
            
            worker_listener.stop()