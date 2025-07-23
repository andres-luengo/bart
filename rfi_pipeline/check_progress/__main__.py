import argparse

import pathlib

import json

from typing import Any

import pandas as pd

from io import StringIO

import datetime

def _parse_args():
    global args

    parser = argparse.ArgumentParser('rfi-pipeline-progress')

    parser.add_argument(
        'rundir',
        help='Output directory of run whose progress to check.',
        type=pathlib.Path
    )

    args = parser.parse_args()

def _get_progress_data():
    with (args.rundir / 'progress-data.json').open('r') as f:
        data = json.load(f)
    return data

def _progress_bar(percentage: float) -> str:
    bar = '=' * min(int(percentage * 100), 100)
    if percentage % 1 >= 0.5:
        bar += '-'
    bar += ' ' * (100 - len(bar))
    return f'[{bar}]'

def _estimate_total_remaining_time(df: pd.DataFrame):
    pass

def format_progress_data(data: list[dict[str, Any]]) -> str:
    output = StringIO()

    df = pd.DataFrame(
        data, 
        columns=['worker pid', 'num complete', 'batch size', 'hit counts', 'times finished']
    )
    num_active_workers = (~df['worker pid'].isna()).sum()
    
    output.write(f'Number of active workers: {num_active_workers:n}\n')
    
    num_all_complete = int(df['num complete'].sum())
    num_all_files = int(max(df['batch size'].sum(), 1))
    percentage = num_all_complete / num_all_files
    
    output.write(f'Total progress: {num_all_complete:,d}/{num_all_files:,d} ({percentage:.2%})\n')
    output.write(f'{_progress_bar(percentage)}\n')


    output.write(f'\nALL WORKERS\n')
    for batch_idx, active_file in df[~df['worker pid'].isna()].iterrows():
        output.write(f'PID {int(active_file['worker pid'])}\n')
        output.write(f'batch {batch_idx}\n')
        
        num_complete = int(active_file['num complete'])
        batch_size = int(max(active_file['batch size'], 1))
        percentage = num_complete / batch_size
        output.write(f'Total progress: {num_complete:,d}/{batch_size:,d} ({percentage:.2%})\n')
        output.write(f'{_progress_bar(percentage)}\n')

    return output.getvalue().strip()
    
def main():
    _parse_args()
    data = _get_progress_data()
    print(format_progress_data(data))

if __name__ == '__main__': main()