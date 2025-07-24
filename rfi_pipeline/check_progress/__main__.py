import argparse

import pathlib

import json

from typing import Any

import numpy as np
import pandas as pd

from io import StringIO

import datetime as dt
from time import sleep

import sys

def _parse_args():
    global args

    parser = argparse.ArgumentParser('rfi-pipeline-progress')

    parser.add_argument(
        'rundir',
        help='Output directory of run whose progress to check.',
        type=pathlib.Path
    )
    
    parser.add_argument(
        '-n', '--update-interval',
        type=float,
        default=0,
        help='Update interval in seconds for continuous monitoring. If 0 or not specified, run once and exit. Default: 0'
    )
    
    parser.add_argument(
        '--once',
        action='store_true',
        help='Run once and exit (overrides --update-interval)'
    )

    args = parser.parse_args()

def _get_progress_data():
    with (args.rundir / 'progress-data.json').open('r') as f:
        data = json.load(f)
    return data

def _progress_bar(percentage: float, width: int = 50) -> str:
    bar = '=' * min(int(percentage * width), width)
    if (percentage * width) % 1 >= 0.5:
        bar += '-'
    bar += ' ' * (width - len(bar))
    return f'[{bar}]'

def _format_timedelta(td: dt.timedelta) -> str:
    """Format a timedelta to show only hours, minutes, and seconds without microseconds or '0 days'"""
    if td == dt.timedelta.max:
        return "∞"
    
    total_seconds = int(td.total_seconds())
    days = total_seconds // 86400
    hours = (total_seconds % 86400) // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    
    if days > 0:
        return f"{days} days {hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def _estimate_total_remaining_time(df: pd.DataFrame) -> float:
    num_complete_files = int(df['num complete'].sum())
    num_files = max(int(df['batch size'].sum()), 1)
    num_remaining_files = num_files - num_complete_files

    hit_count_lists = df['hit counts'].dropna()

    if len(hit_count_lists) == 0: return np.inf
    
    hit_counts = np.hstack(hit_count_lists) #type: ignore
    
    # seconds
    job_duration_lists = df['times elapsed'].dropna()
    job_durations = np.hstack(job_duration_lists) #type: ignore
    
    # want to ignore failed files in calculations
    valid_mask = (hit_counts >= 0)
    hit_counts = hit_counts[valid_mask]
    job_durations = job_durations[valid_mask]

    hits_remaining_est = np.mean(hit_counts) * num_remaining_files
    
    # hits/second
    hit_process_rate_est = np.mean(hit_counts / job_durations)

    # pretend there's only one worker if process is inactive for whatever reason
    num_workers = max((~df['worker pid'].isna()).sum(), 1)

    return np.divide(hits_remaining_est, hit_process_rate_est) / num_workers

def format_progress_data(data: list[dict[str, Any]]) -> str:
    output = StringIO()

    df = pd.DataFrame(
        data, 
        columns=['worker pid', 'num complete', 'batch size', 'hit counts', 
                 'times elapsed', 'last file end time']
    )
    now = dt.datetime.now(dt.timezone.utc)

    for batch_idx, active_file in df[~df['worker pid'].isna()].iterrows():
        output.write(f'PID {int(active_file['worker pid'])}\n')
        output.write(f'batch {batch_idx}\n')
        
        num_complete = int(active_file['num complete'])
        batch_size = int(max(active_file['batch size'], 1))
        percentage = num_complete / batch_size
        output.write(f'Total progress: {num_complete:,d}/{batch_size:,d} ({percentage:.2%})\n')
        output.write(f'{_progress_bar(percentage)}\n')

        batch_df = df.loc[[batch_idx]]
        time_estimate = _estimate_total_remaining_time(batch_df)
        try:
            time_estimate_delta = dt.timedelta(seconds=time_estimate)
        except OverflowError:
            time_estimate_delta = dt.timedelta.max
        output.write(f'Time remaining: ~{_format_timedelta(time_estimate_delta)}\n')
        
        time_since_last_finish = now - dt.datetime.fromisoformat(active_file['last file end time'])
        output.write(f'Time since last file: {_format_timedelta(time_since_last_finish)}\n')

        output.write('\n')
    
    output.write(f'\nSUMMARY\n')
    output.write(   '=======\n')
    num_active_workers = (~df['worker pid'].isna()).sum()
    
    output.write(f'Number of active workers: {num_active_workers:n}\n')
    
    num_all_complete = int(df['num complete'].sum())
    num_all_files = int(max(df['batch size'].sum(), 1))
    percentage = num_all_complete / num_all_files
    
    output.write(f'Total progress: {num_all_complete:,d}/{num_all_files:,d} ({percentage:.2%})\n')
    output.write(f'{_progress_bar(percentage)}\n')

    # (might return inf if we've only seen files with no hits for the last few)
    time_estimate = _estimate_total_remaining_time(df)
    try:
        time_estimate_delta = dt.timedelta(seconds=time_estimate)
    except OverflowError:
        time_estimate_delta = dt.timedelta.max
    output.write(f'Time remaining: ~{_format_timedelta(time_estimate_delta)}\n')

    latest_file_finish_time = (df['last file end time']
                               .dropna()
                               .apply(dt.datetime.fromisoformat)
                               .max())
    time_since_last_finish = now - latest_file_finish_time
    output.write(f'Time since last finish: {_format_timedelta(time_since_last_finish)}\n')

    return output.getvalue()
    
def main():
    _parse_args()
    
    # If --once is specified or update_interval is 0, run once and exit
    if args.once or args.update_interval <= 0:
        data = _get_progress_data()
        print(format_progress_data(data))
        return
    
    # Continuous monitoring mode
    try:
        while True:
            # Clear the screen (works on most terminals)
            print('\033[2J\033[H', end='')
            
            try:
                data = _get_progress_data()
                output = format_progress_data(data)
                print(output)
                
                # Add timestamp for the last update
                now = dt.datetime.now()
                print(f"Last updated: {now.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Press Ctrl+C to exit")
                
            except FileNotFoundError:
                print(f"Progress file not found in {args.rundir}")
                print(f"Make sure the RFI pipeline is running in this directory.")
                print(f"Last checked: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Press Ctrl+C to exit")
            except json.JSONDecodeError:
                print(f"Error reading progress file in {args.rundir}")
                print(f"The file might be corrupted or being written to.")
                print(f"Last checked: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Press Ctrl+C to exit")
            except Exception as e:
                print(f"Unexpected error: {e}")
                print(f"Last checked: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Press Ctrl+C to exit")
            
            sleep(args.update_interval)
            
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)

if __name__ == '__main__': main()