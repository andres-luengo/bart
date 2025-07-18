from .process import BatchJob

from argparse import Namespace
from pathlib import Path

import multiprocessing as mp

import logging, logging.handlers

import json

from typing import Any

import resource

# splits up files into batches
# does multiprocessing stuff
# manages file nonsense
# manages resources
class Manager:
    def __init__(
            self, 
            process_params: dict[str, Any],
            num_batches: int, 
            num_processes: int, 
            files: tuple[Path, ...], 
            outdir: Path,
            max_rss: int
    ):
        self.process_params = process_params
        self.num_processes = num_processes
        self.batches: tuple[tuple[Path, ...], ...] = tuple(
            files[i::num_batches] for i in range(num_batches)
        )
        self.outdir = outdir
        self._logger = logging.getLogger(__name__)
        self.max_rss = max_rss

        with open(self.outdir / 'process-params.json', 'w') as f:
            json.dump(self.process_params, f)

    @classmethod
    def from_namespace(cls, arg: Namespace, files: tuple[Path, ...]):
        return cls(
            process_params = {
                'freq_window' : arg.frequency_block_size,
                'warm_significance' : arg.warm_significance,
                'hot_significance' : arg.hot_significance,
                'min_freq': arg.min_freq,
                'max_freq': arg.max_freq
            },
            num_batches=arg.num_batches,
            num_processes=arg.num_processes,
            files=files,
            outdir=arg.outdir,
            max_rss=arg.max_rss_gb * 1e9
        )
    
    @staticmethod
    def worker_init(log_queue: mp.Queue, max_memory: int):
        handler = logging.handlers.QueueHandler(log_queue)
        logger = logging.getLogger()
        logger.handlers.clear()
        logger.addHandler(handler)

        logger.info(f'Initializing worker with {max_memory/1e9 = }GB')
        resource.setrlimit(resource.RLIMIT_AS, (max_memory, max_memory))
    
    @staticmethod
    def batch_job(args: tuple[dict[str, Any], Path, tuple[Path, ...], int]):
        try:
            job = BatchJob(*args)
            return job.run()
        except Exception as e:
            id: int = args[-1]
            logging.critical(
                f'EXCEPTION ON BATCH {id:<03}', 
                exc_info = True, stack_info = True
            )
            raise e
    
    def run(self):
        self._logger.info('Starting jobs...')

        batch_args = (
            (
                self.process_params,
                self.outdir / 'batches',
                batch, 
                i
            )
            for i, batch in enumerate(self.batches)
        )

        log_queue = mp.Manager().Queue()
        worker_listener = logging.handlers.QueueListener(
            log_queue,
            *logging.getLogger().handlers,
            respect_handler_level = True
        )
        worker_listener.start()

        with mp.Pool(
            processes=self.num_processes,
            initializer=self.worker_init,
            initargs=(
                log_queue, 
                int(self.max_rss // self.num_processes)
            )
        ) as p:
            # using this over map so that if anything raises it gets sent up ASAP
            results = p.imap_unordered(
                self.batch_job, batch_args, chunksize=1
            )
            for _ in results: pass # consume lazy map
        
        worker_listener.stop()