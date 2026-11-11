import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import xarray as xr

import stitch_krycklan


ROOT = Path(__file__).resolve().parents[1]


class StitchKrycklanTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = os.environ.copy()
        self.env.pop('SPAFHY_RESULTS_FOLDER', None)
        self.env.pop('SPAFHY_RESULTS_RUN', None)

    def run_stitch(self, *args):
        return subprocess.run(
            [sys.executable, str(ROOT / 'stitch_krycklan.py'), *args],
            cwd=self.root, env=self.env, text=True, capture_output=True,
        )

    def create_inputs(self, configuration='channels_20m'):
        input_dir = self.root / configuration
        for index in range(2):
            folder = input_dir / f'catchment-{index}'
            folder.mkdir(parents=True)
            ds = xr.Dataset(
                {'flow': (('time', 'lat', 'lon'), np.full((3, 2, 2), index + 1.0))},
                coords={
                    'time': np.array(['2015-03-31', '2015-04-01', '2015-04-02'], dtype='datetime64[ns]'),
                    'lat': [0.0, 1.0], 'lon': [float(index * 2), float(index * 2 + 1)],
                },
            )
            ds.to_netcdf(folder / 'result.nc')

    def test_real_stitch_and_existing_output_protection(self):
        self.create_inputs()
        args = (
            '--array-task-id', '0', '--run-dir', str(self.root),
            '--start-date', '2015-04-01', '--end-date', '2015-04-02',
        )
        result = self.run_stitch(*args)
        self.assertEqual(result.returncode, 0, result.stderr)
        output = self.root / 'stitched' / 'channels_20m.nc'
        with xr.open_dataset(output) as ds:
            self.assertEqual(ds.sizes, {'time': 2, 'lat': 2, 'lon': 4})
            np.testing.assert_array_equal(ds.flow.values[:, :, :2], 1)
            np.testing.assert_array_equal(ds.flow.values[:, :, 2:], 2)
        original = output.read_bytes()
        result = self.run_stitch(*args)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Output already exists', result.stderr)
        self.assertEqual(output.read_bytes(), original)
        self.assertEqual(list((self.root / 'stitched').glob('.stitch-*')), [])

    def test_parallel_tasks_do_not_share_temporary_files(self):
        self.create_inputs('channels_20m')
        self.create_inputs('channels_40m')
        commands = [
            [sys.executable, str(ROOT / 'stitch_krycklan.py'), '--array-task-id', str(task),
             '--run-dir', str(self.root)]
            for task in (0, 1)
        ]
        processes = [
            subprocess.Popen(command, cwd=self.root, env=self.env, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True)
            for command in commands
        ]
        for process in processes:
            stdout, stderr = process.communicate(timeout=60)
            self.assertEqual(process.returncode, 0, stdout + stderr)
        for configuration in ('channels_20m', 'channels_40m'):
            with xr.open_dataset(self.root / 'stitched' / f'{configuration}.nc') as ds:
                self.assertEqual(ds.sizes['lon'], 4)
        self.assertEqual(list((self.root / 'stitched').glob('.stitch-*')), [])

    def test_empty_time_slice_does_not_publish_output(self):
        self.create_inputs()
        result = self.run_stitch(
            '--array-task-id', '0', '--run-dir', str(self.root),
            '--start-date', '2016-01-01', '--end-date', '2016-02-01',
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('No time steps found', result.stderr)
        self.assertFalse((self.root / 'stitched' / 'channels_20m.nc').exists())
        self.assertEqual(list((self.root / 'stitched').glob('.stitch-*')), [])

    def test_processing_failure_does_not_publish_output(self):
        self.create_inputs()
        args = ['stitch_krycklan.py', '--array-task-id', '0', '--run-dir', str(self.root)]
        with mock.patch.object(sys, 'argv', args), \
             mock.patch.object(xr, 'where', side_effect=ValueError('bad data')):
            with self.assertRaisesRegex(RuntimeError, 'Error processing variable'):
                stitch_krycklan.main()
        self.assertFalse((self.root / 'stitched' / 'channels_20m.nc').exists())
        self.assertEqual(list((self.root / 'stitched').glob('.stitch-*')), [])

    def test_array_mapping_and_default_paths(self):
        self.env['SPAFHY_RESULTS_FOLDER'] = str(self.root)
        self.env['SPAFHY_RESULTS_RUN'] = 'example'
        run_dir = self.root / 'krycklan' / 'example'
        for task in range(8):
            resolution = (20, 40, 80, 160)[task % 4]
            streams = ('channels', '5haStreams')[task // 4]
            result = self.run_stitch('--array-task-id', str(task))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(str(run_dir / f'{streams}_{resolution}m'), result.stderr)

    def test_invalid_inputs_fail(self):
        self.create_inputs()
        for args in (
            ('--array-task-id', '8'),
            ('--resolution', '20'),
            ('--array-task-id', '0', '--streams', 'channels'),
            ('--array-task-id', '0', '--start-date', '2015-10-01', '--end-date', '2015-01-01'),
            ('--array-task-id', '0', '--start-date', 'bad'),
            ('--resolution', '40', '--streams', 'channels', '--run-dir', str(self.root)),
        ):
            with self.subTest(args=args):
                result = self.run_stitch(*args)
                self.assertNotEqual(result.returncode, 0)
                self.assertTrue(result.stderr)
        empty = self.root / 'channels_40m'
        empty.mkdir()
        result = self.run_stitch('--resolution', '40', '--streams', 'channels', '--run-dir', str(self.root))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('No NetCDF files found', result.stderr)

    def test_date_defaults_match_notebook(self):
        original = sys.argv
        self.addCleanup(setattr, sys, 'argv', original)
        sys.argv = ['stitch_krycklan.py', '--resolution', '20', '--streams', 'channels']
        args = stitch_krycklan.parse_args()
        self.assertEqual(args.start_date.isoformat(), '2015-04-01')
        self.assertEqual(args.end_date.isoformat(), '2015-09-01')

    def test_slurm_array_passes_task_and_user_arguments(self):
        shutil.copy(ROOT / 'stitch_krycklan.slurm.sh', self.root)
        binary = self.root / 'bin'
        binary.mkdir()
        fake_python = binary / 'python'
        fake_python.write_text('#!/bin/bash\nprintf "PYTHON=%s\\n" "$*"\n')
        fake_python.chmod(0o755)
        (self.root / '.env').write_text('SLURM_ARRAY_TASK_ID=7\n')
        self.env.update(
            PATH=f"{binary}{os.pathsep}{self.env['PATH']}",
            SLURM_SUBMIT_DIR=str(self.root), SLURM_ARRAY_TASK_ID='2',
        )
        self.env.pop('SPAFHY_FOLDER', None)
        result = subprocess.run(
            ['bash', str(self.root / 'stitch_krycklan.slurm.sh'),
             '--run-dir', '/data/run folder', '--start-date', '2015-05-01'],
            cwd=self.root, env=self.env, text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--array-task-id 2 --run-dir /data/run folder --start-date 2015-05-01', result.stdout)


if __name__ == '__main__':
    unittest.main()
