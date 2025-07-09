from argparse import Namespace
from io import TextIOWrapper

# splits up files into batches
# does multiprocessing stuff
# manages file nonsense
# manages resources
class PicklesManager:
    def __init__(self, num_batches: int, num_processes: int, files: list[str], outdir: TextIOWrapper):
        self._num_processes = num_processes
        self._batches = [files[i::num_batches] for i in range(num_batches)]
        print(self._batches)
        self._outdir = outdir
    
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
    
    def run():
        pass

class ParallelLoggerWrapper: pass