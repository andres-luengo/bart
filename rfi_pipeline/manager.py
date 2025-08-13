"""
Run Manager Module
~~~~~~~~~~~~~~~~~~

Provides :class:`RunManager` to run a user ``file_job`` over files in parallel.

``file_job`` contract (must implement):

* Signature: ``file_job(path, process_params) -> pandas.DataFrame``
* One row per hit/result. Framework adds a ``source file`` column automatically.
* Use ``logging`` (that is, don't write directly to stdout with something like ``print``) so output is captured from workers.

Each returned DataFrame is appended to ``batches/batch_<NNN>.csv`` (header once).
To merge these batches, see :mod:`rfi_pipeline.merge`

Exceptions are logged; if ``continue_on_exception`` is False the run stops, else
the file is marked failed (``num_hits = -1``) and processing continues.

The max amount of memory per process is limited by the ``max_rss`` parameter.
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
    """Run a ``file_job`` over files in parallel batches.

    Summary:

    * Partitions files into ``num_batches``.
    * Spawns a pool of ``num_processes`` workers.
    * Streams each file's DataFrame output into a batch CSV (+ metadata files).
    * Centralises logging; use ``logging`` not ``print``.
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
            continue_on_exception: bool = False
    ):
        """Initialize the pipeline manager for parallel file processing.
            This constructor prepares batched work units, output / logging directories,
            metadata, and a progress-tracking file used to resume or monitor execution.
            
            Parameters
            ----------
            file_job : Callable[[os.PathLike, dict[str, Any]], pandas.DataFrame]
                A callable invoked for each input file. It must accept (path, process_params)
                and return a pandas DataFrame with that file's results.
            process_params : dict[str, Any]
                A dictionary of parameters passed unchanged to every invocation of `file_job`.
            files : Sequence[os.PathLike]
                Iterable of input file paths to process.
            outdir : os.PathLike
                Root directory where outputs, logs, batch manifests, and progress metadata
                will be written. Created if it does not exist.
            num_batches : int, default=1
                Number of batches to split `files` into. Each batch may be processed
                independently; useful for coarse progress tracking or distributing work.
            num_processes : int, default=1
                Number of worker processes for parallel execution. Values >1 enable
                multiprocessing; 1 forces serial execution (useful for debugging).
            max_rss : int, default=8
                Approximate total memory (in bytes) budget per worker. The value is
                divided among active workers to derive a per‑process cap (enforced elsewhere
                if applicable).
            log_level : int, default=logging.WARNING
                Logging verbosity level (e.g. logging.INFO, logging.DEBUG). Used to
                configure this manager's logger.
            continue_on_exception : bool, default=False
                If True, failures in processing individual files are logged and skipped;
                if False, the first exception aborts the run.
            
                
            Side Effects
            ------------
            Creates (if absent) the following within `outdir`:
            
            - batches
            - logs
            - a copy of target file list

            Initializes or cleans a progress JSON file (progress-data.json).
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
        """
        A file counts as completed iff it appears in ``files.csv`` with
        ``num_hits >= 0`` (zero‑hit or had results). Rows with ``num_hits == -1``
        (failure) are ignored so the file can be retried.
        """
        files_csv = self.outdir / 'files.csv'
        if not files_csv.is_file():
            return set()
        try:
            fdf = pd.read_csv(files_csv, usecols=['file', 'num_hits'])
            mask = fdf['num_hits'] >= 0
            return {Path(p) for p in fdf.loc[mask, 'file'].unique()}
        except Exception:
            return set()

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