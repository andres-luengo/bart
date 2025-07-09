from pickle_job import PickleJob

from argparse import Namespace
from io import TextIOWrapper

from multiprocessing import Pool, Manager, Queue

import logging, logging.handlers
import os

# splits up files into batches
# does multiprocessing stuff
# manages file nonsense
# manages resources
class PicklesManager:
    def __init__(self, num_batches: int, num_processes: int, files: list[str], outdir: TextIOWrapper):
        self._num_processes = num_processes
        self._batches = [files[i::num_batches] for i in range(num_batches)]
        self._outdir = outdir
        self._logger = logging.getLogger(__name__)
    
    @staticmethod
    def get_file_list(file: TextIOWrapper) -> list[str]:
        return file.read().strip().splitlines()

    @classmethod
    def from_namespace(cls, arg: Namespace):
        return cls(
            num_batches = arg.num_batches,
            num_processes = arg.num_processes,
            files = cls.get_file_list(arg.infile),
            outdir = arg.outdir
        )
    
    @staticmethod
    def worker_init(log_queue: Queue):
        handler = logging.handlers.QueueHandler(log_queue)
        logger = logging.getLogger()
        logger.handlers.clear()
        logger.addHandler(handler)
    
    @staticmethod
    def batch_job(batch: list[str]):
        job = PickleJob(batch)
        return job.run()
    
    def run(self):
        self._logger.info('Starting jobs...')
        log_queue = Manager().Queue()
        worker_listener = logging.handlers.QueueListener(
            log_queue,
            *logging.getLogger().handlers,
            respect_handler_level = True # WHY IS THIS NOT THE DEFAULT*????????????????????
        )
        worker_listener.start()

        with Pool(
            processes=self._num_processes,
            initializer=self.worker_init,
            initargs=(log_queue,)
        ) as p:
            # using this over map so that if anything raises it gets sent up ASAP
            results = p.imap_unordered(
                self.batch_job, self._batches
            )
            [*results]
        
        worker_listener.stop()


# *like.
# very_normal_ok_function(EXPLODE_ORPHANAGE = False) # otherwise orphanage explodes