import argparse
import pathlib

from pickle_manager import PicklesManager

import logging, logging.handlers
import shutil

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog = 'pickles-redux',
        description = (
            'pickles 2 :)'
        )
    )
    parser.add_argument(
        'infile',
        type = argparse.FileType('r'),
        help = 'Path to a file containing the paths to every .h5 file to be analyzed, separated by newlines.'
    )
    parser.add_argument(
        'outdir',
        type = pathlib.Path,
        help = 'Path to desired output folder'
    )
    parser.add_argument(
        '-f', '--force',
        action = 'store_true',
        help = 'If present and outdir already exists, it will be overwritten.'
    )

    parser.add_argument(
        '-n', '--num-processes',
        default = 1,
        type = int,
        help = 'Number of processes to use while processing in parallel.'
    )
    parser.add_argument(
        '-m', '--max-rss-gb',
        default = 32.0,
        type = float,
        help = (
            'Max amount of memory that the program is allowed to take, in GB. '
            'I recommend >= 32 * NUM_PROCESSES, but whatever floats your boat. '
            'Keep as small as possible to keep Matt (and everybody else...) happy.'
        )
    )

    parser.add_argument(
        '--num-batches',
        default = 100,
        type = int,
        help = (
            'This program breaks up processing into NUM_BATCHES batches. '
            'In case something bad happens or one batch output gets corrupted, the others should be fine. '
            'Defaults to 100.'
            )
    )

    parser.add_argument(
        '-v', '--verbose',
        action = 'count',
        default = 0,
        help = (
            'Each repetition of this flag adds a level of logging below WARNING '
            'to print to stdout. For example, -vv adds INFO and DEBUG. '
            'Defaults to no repetitions, so just WARNING and above.'
            )
    )
    parser.add_argument(
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
    file_handler.setLevel(logging.DEBUG)
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

def main():
    args = parse_args()
    make_outdir(args)
    manager = PicklesManager.from_namespace(args)
    manager.run()

if __name__ == '__main__': main()