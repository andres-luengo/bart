from process import BatchJob

from argparse import Namespace
from pathlib import Path

from multiprocessing import Pool, Manager, Queue

import logging, logging.handlers

from collections import deque

from typing import Any

# splits up files into batches
# does multiprocessing stuff
# manages file nonsense
# manages resources
class PicklesManager:
    def __init__(
            self, 
            process_params: dict[str, Any],
            num_batches: int, 
            num_processes: int, 
            files: tuple[Path, ...], 
            outdir: Path
    ):
        self.process_params = process_params
        self.num_processes = num_processes
        self.batches: tuple[tuple[Path, ...], ...] = tuple(
            files[i::num_batches] for i in range(num_batches)
        )
        self.outdir = outdir
        self._logger = logging.getLogger(__name__)

    @classmethod
    def from_namespace(cls, arg: Namespace, files: tuple[Path, ...]):
        return cls(
            process_params = {
                'freq_window' : arg.frequency_block_size,
                'warm_significance' : arg.warm_significance,
                'hot_significance' : arg.hot_significance
            },
            num_batches = arg.num_batches,
            num_processes = arg.num_processes,
            files = files,
            outdir = arg.outdir
        )
    
    @staticmethod
    def worker_init(log_queue: Queue):
        handler = logging.handlers.QueueHandler(log_queue)
        logger = logging.getLogger()
        logger.handlers.clear()
        logger.addHandler(handler)
    
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

        log_queue = Manager().Queue()
        worker_listener = logging.handlers.QueueListener(
            log_queue,
            *logging.getLogger().handlers,
            respect_handler_level = True # WHY IS THIS NOT THE DEFAULT*????????????????????
        )
        worker_listener.start()

        with Pool(
            processes=self.num_processes,
            initializer=self.worker_init,
            initargs=(log_queue,)
        ) as p:
            # using this over map so that if anything raises it gets sent up ASAP
            results = p.imap_unordered(
                self.batch_job, batch_args
            )
            deque(results, 0) # consume iterator
        
        worker_listener.stop()


# *like.
# def very_normal_function(normal_parameter: float, explode_orphanage: bool = True):
#     """
#     Totally normal function. Gives you the square of normal_parameter
    
#     Args:
#     - normal_parameter: parameter to square
#     - explode_orphanage: whether to explode an orphanage. defaults to true
#     """
#     if explode_orphanage:
#         nuclear_bomb.explode('orphanage')
#     return normal_parameter ** 2