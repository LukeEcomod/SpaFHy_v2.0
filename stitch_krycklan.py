#!/usr/bin/env python3
"""Stitch one Krycklan resolution and stream configuration."""

import argparse
from datetime import date
import os
from pathlib import Path
import tempfile

from dotenv import load_dotenv


RESOLUTIONS = (20, 40, 80, 160)
STREAMS = ('channels', '5haStreams')


def parse_date(value):
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f'Invalid date {value!r}; use YYYY-MM-DD') from exc


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--array-task-id', type=int, choices=range(8))
    selection.add_argument('--resolution', type=int, choices=RESOLUTIONS)
    parser.add_argument('--streams', choices=STREAMS)
    parser.add_argument(
        '--run-dir', type=Path,
        help='Directory containing the <streams>_<resolution>m input directories.',
    )
    parser.add_argument('--start-date', type=parse_date, default=date(2015, 4, 1))
    parser.add_argument('--end-date', type=parse_date, default=date(2015, 9, 1))
    args = parser.parse_args()
    if args.array_task_id is not None:
        if args.streams is not None:
            parser.error('--streams cannot be used with --array-task-id')
        args.resolution = RESOLUTIONS[args.array_task_id % len(RESOLUTIONS)]
        args.streams = STREAMS[args.array_task_id // len(RESOLUTIONS)]
    elif args.streams is None:
        parser.error('--resolution requires --streams')
    if args.start_date > args.end_date:
        parser.error('--start-date must not be later than --end-date')
    return args


def main():
    load_dotenv()
    args = parse_args()
    if args.run_dir is not None:
        run_dir = args.run_dir.expanduser().resolve()
    else:
        results_folder = os.getenv('SPAFHY_RESULTS_FOLDER')
        if not results_folder:
            raise RuntimeError('Set SPAFHY_RESULTS_FOLDER or pass --run-dir')
        run_dir = (
            Path(results_folder).expanduser().resolve()
            / 'krycklan' / os.getenv('SPAFHY_RESULTS_RUN', 'run_9_2026')
        )

    configuration = f'{args.streams}_{args.resolution}m'
    input_dir = run_dir / configuration
    output_dir = run_dir / 'stitched'
    output_file = output_dir / f'{configuration}.nc'
    if not input_dir.is_dir():
        raise FileNotFoundError(f'Input directory does not exist: {input_dir}')
    if not any(input_dir.rglob('*.nc')):
        raise FileNotFoundError(f'No NetCDF files found in {input_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Configuration: {configuration}', flush=True)
    print(f'Input: {input_dir}', flush=True)
    print(f'Output: {output_file}', flush=True)
    print(f'Dates (inclusive): {args.start_date} to {args.end_date}', flush=True)

    from iotools import stitch_result_nc_files

    with tempfile.TemporaryDirectory(prefix=f'.stitch-{configuration}-', dir=output_dir) as temp_dir:
        staged_output = Path(temp_dir) / output_file.name
        stitch_result_nc_files(
            str(input_dir), str(staged_output), plot=False,
            start_date=args.start_date.isoformat(), end_date=args.end_date.isoformat(),
        )
        os.replace(staged_output, output_file)
    print(f'Stitched output: {output_file}', flush=True)


if __name__ == '__main__':
    main()
