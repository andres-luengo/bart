import argparse
import pathlib

from pickle_manager import PicklesManager

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
        help = 'This program breaks up processing into NUM_BATCHES batches. In case something bad happens or one batch output gets corrupted, the others should be fine. Defaults to 100'
    )

    return parser.parse_args()


def main():
    args = parse_args()
    manager = PicklesManager.from_namespace(args)
    print(manager)

if __name__ == '__main__': main()