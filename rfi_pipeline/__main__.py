import argparse
from pathlib import Path

from .manager import Manager

import logging, logging.handlers
import shutil

from io import TextIOWrapper
import sys

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog = 'rfi-pipeline'
    )
    io_group = parser.add_argument_group('Input and Output')
    io_group.add_argument(
        'infile',
        type = argparse.FileType('r'),
        help = 'Path to a file containing the paths to every .h5 file to be analyzed, separated by newlines.'
    )
    io_group.add_argument(
        'outdir',
        type = Path,
        help = 'Path to desired output folder'
    )
    io_group.add_argument(
        '-f', '--force',
        action = 'store_true',
        help = 'If present and outdir already exists, it will be overwritten.'
    )

    processing_group = parser.add_argument_group('Processing')
    processing_group.add_argument(
        '--min-freq',
        type=float,
        help='Frequency bins below this value are ignored. Defaults to -inf',
        default=float('-inf')
    )
    processing_group.add_argument(
        '--max-freq',
        type=float,
        help='Frequency bins above this value are ignored. Defaults to inf',
        default=float('inf')
    )
    processing_group.add_argument(
        '--frequency-block-size',
        type = int,
        help = 'Each scan is broken up into frequency blocks of this size before computing statistics.',
        default = 1024 # this is what caleb uses
    )
    processing_group.add_argument(
        '--warm-significance',
        type = float,
        help = (
            'We first filter for blocks that have data points WARM_SIGNIFICANCE '
            'sigma above the median along the middle time bins. Defaults to 4.0.'
        ),
        default = 4.0
    )
    processing_group.add_argument(
        '--hot-significance',
        type = float,
        help = (
            'A similar filter happens after the WARM_SIGNIFICANCE filter, '
            'leaving in blocks that have data points at least HOT_SIGNIFICANCE median '
            'absolute deviaions from the median. Defaults to 10.0'
        ),
        default = 10.0
    )
    processing_group.add_argument(
        '--hotter-significance',
        type = float,
        help = (
            'A similar filter happens after the HOT_SIGNIFICANCE filter, '
            'leaving in blocks whose max value has an SNR of at least this value. '
            'The noise is calculated by sigma clipping the data along the middle time slice '
            'with lower and upper clips of SIGMA_CLIP, and then calculating the standard deviation. '
            'Defaults to 10.0'
        ),
        default = 10.0
    )
    processing_group.add_argument(
        '--sigma-clip',
        type = float,
        help = (
            'The value by which to sigma clip data snippets in the HOTTER_SIGNIFICANCE filter. '
            'Defaults to 3.0.'
        ),
        default = 3.0
    )

    resource_management_group = parser.add_argument_group('Resource Management')
    resource_management_group.add_argument(
        '-n', '--num-processes',
        default = 1,
        type = int,
        help = 'Number of processes to use while processing in parallel.'
    )
    resource_management_group.add_argument(
        '-m', '--max-rss-gb',
        default = 32.0,
        type = float,
        help = (
            'Max amount of memory that the program is allowed to take, in GB. '
            'I recommend >= 32 * NUM_PROCESSES, but whatever floats your boat. '
            'Keep as small as possible to keep Matt (and everybody else...) happy.'
        )
    )

    resource_management_group.add_argument(
        '--num-batches',
        default = 100,
        type = int,
        help = (
            'This program breaks up processing into NUM_BATCHES batches. '
            'In case something bad happens or one batch output gets corrupted, the others should be fine. '
            'Defaults to 100.'
            )
    )

    logging_group = parser.add_argument_group('Logging')
    logging_group.add_argument(
        '-v', '--verbose',
        action = 'count',
        default = 0,
        help = (
            'Each repetition of this flag adds a level of logging below WARNING '
            'to print to stdout. For example, -vv adds INFO and DEBUG. '
            'Defaults to no repetitions, so just WARNING and above.'
            )
    )
    logging_group.add_argument(
        '-q', '--quiet',
        action = 'count',
        default = 0,
        help = (
            'Levels of logging at or above WARNING to mute. '
            'For example, -qq mutes WARNING, and ERROR, but leaves in CRITICAL. '
            'Defaults to no repetitions, so leaves WARNING and above.'
            )
    )

    return parser.parse_args()

def make_outdir(args: argparse.Namespace):
    try:
        args.outdir.mkdir()
    except FileNotFoundError as e:
        print(e)
        exit(1)
    except FileExistsError as e:
        if not args.force:
            print(e)
            exit(1)
        else:
            shutil.rmtree(args.outdir)
            args.outdir.mkdir()
    
    (args.outdir / 'logs').mkdir()
    (args.outdir / 'batches').mkdir()


def logging_setup(args: argparse.Namespace):
    logger = logging.getLogger()

    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s (%(process)d): %(message)s'
    )

    level = 30 + 10 * (args.quiet - args.verbose)
    stderr_handler = logging.StreamHandler()
    stderr_handler.setLevel(level)
    stderr_handler.setFormatter(formatter)

    file_handler = logging.handlers.RotatingFileHandler(
        filename = args.outdir / 'logs' / 'all_logs.log',
        maxBytes = 256 * 2**10, # 256 KiB
        backupCount = 3, # so max of 4 * 256 KiB = 1 MiB,
    )
    file_handler.setLevel(min(logging.INFO, level))
    file_handler.setFormatter(formatter)

    # if error files long enough that this is a problem, there are bigger ones
    err_file_handler = logging.FileHandler(
        filename = args.outdir / 'logs' / 'error_logs.log'
    )
    err_file_handler.setLevel(logging.ERROR)
    err_file_handler.setFormatter(formatter)

    logger.addHandler(stderr_handler)
    logger.addHandler(file_handler)
    logger.addHandler(err_file_handler)
    logger.setLevel(logging.DEBUG)

def get_file_names(file: TextIOWrapper) -> tuple[Path, ...]:
    lines = file.read().strip().splitlines()
    paths = map(Path, lines)
    return tuple(paths)

def main():
    args = parse_args()
    make_outdir(args)
    logging_setup(args)
    files = get_file_names(args.infile)
    manager = Manager.from_namespace(args, files)
    manager.run()

if __name__ == '__main__': main()