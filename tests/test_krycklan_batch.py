import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SIZES = {
    'small': ('1 2 3 4 5 6 7', 7),
    'medium': ('8 9 12 13 20 21', 6),
    'large': ('141 142 143 151 152 153 161 162 163 164 22 10', 12),
}
SCRIPT_NAMES = [
    'submit_parallel_krycklan.sh',
    'run_parallel_krycklan.slurm.sh',
    *(f'run_parallel_krycklan_{size}.slurm.sh' for size in SIZES),
]


class KrycklanParametersTest(unittest.TestCase):
    def parameter_values(self, resolution=None, streams=None):
        env = os.environ.copy()
        for key in ('SPAFHY_RESOLUTION', 'SPAFHY_STREAMS', 'SPAFHY_RESULTS_RUN'):
            env.pop(key, None)
        env['SPAFHY_RESULTS_FOLDER'] = '/results'
        if resolution is not None:
            env['SPAFHY_RESOLUTION'] = str(resolution)
        if streams is not None:
            env['SPAFHY_STREAMS'] = streams
        code = """
import json
import parameters_krycklan as p
g, _, b, d = p.parameters('/project/krycklan')
print(json.dumps([p.reso, p.streams, g['gis_folder'], g['results_folder'],
                  b['streams'], d['streams'], d['stream_distance'],
                  d['stream_length'], d['stream_width'], d['stream_depth'],
                  p.auxiliary_grids()['streams']]))
"""
        return subprocess.run(
            [sys.executable, '-c', code], cwd=ROOT, env=env,
            text=True, capture_output=True,
        )

    def test_defaults_and_each_configuration(self):
        for resolution, streams in [(None, None)] + [
            (resolution, streams)
            for streams in ('channels', '5haStreams')
            for resolution in (20, 40, 80, 160)
        ]:
            with self.subTest(resolution=resolution, streams=streams):
                result = self.parameter_values(resolution, streams)
                self.assertEqual(result.returncode, 0, result.stderr)
                resolution = resolution or 20
                streams = streams or 'channels'
                self.assertEqual(json.loads(result.stdout), [
                    resolution, streams, f'/project/krycklan/gis/{resolution}m',
                    f'/results/krycklan/run_9_2026/{streams}_{resolution}m',
                    f'{streams}.asc', f'{streams}.asc',
                    f'{streams}_distance.asc', f'{streams}_length.asc',
                    f'{streams}_width.asc', f'{streams}_depth.asc',
                    f'{streams}.asc',
                ])

    def test_invalid_settings_fail_explicitly(self):
        for resolution, streams in [('bad', None), (0, None), (None, '')]:
            with self.subTest(resolution=resolution, streams=streams):
                result = self.parameter_values(resolution, streams)
                self.assertNotEqual(result.returncode, 0)
                self.assertTrue(result.stderr)


class KrycklanBatchTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for script in SCRIPT_NAMES:
            shutil.copy(ROOT / script, self.root / script)
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        self.env = os.environ.copy()
        for key in (
            'SPAFHY_FOLDER', 'SPAFHY_KRYCKLAN_ARRAY', 'SLURM_ARRAY_TASK_ID',
            'SPAFHY_RESOLUTION', 'SPAFHY_STREAMS', 'RUN_MODE', 'CATCHMENT_NOS',
            'HDF5_USE_FILE_LOCKING',
        ):
            self.env.pop(key, None)
        self.env.update({
            'PATH': f"{self.bin}{os.pathsep}{self.env['PATH']}",
            'SLURM_SUBMIT_DIR': str(self.root),
            'PROJECT_FOLDER': '/project',
        })

    def executable(self, name, contents):
        path = self.bin / name
        path.write_text('#!/bin/bash\nset -eu\n' + contents)
        path.chmod(0o755)

    def test_roihu_resource_directives(self):
        requests = {
            'run_parallel_krycklan.slurm.sh': (12, '16G', '00:15:00'),
            'run_parallel_krycklan_small.slurm.sh': (7, '16G', '01:00:00'),
            'run_parallel_krycklan_medium.slurm.sh': (6, '16G', '04:00:00'),
            'run_parallel_krycklan_large.slurm.sh': (12, '64G', '32:00:00'),
        }
        for script, (cpus, memory, duration) in requests.items():
            with self.subTest(script=script):
                directives = (self.root / script).read_text().splitlines()
                for directive in (
                    '#SBATCH --nodes=1', '#SBATCH --account=project_2000908',
                    '#SBATCH --partition=small', f'#SBATCH --cpus-per-task={cpus}',
                    f'#SBATCH --mem={memory}', f'#SBATCH --time={duration}',
                ):
                    self.assertIn(directive, directives)

    def test_medium_job_disables_hdf5_file_locking(self):
        self.executable('python', 'printf "HDF5=%s\\n" "${HDF5_USE_FILE_LOCKING:-unset}"\n')
        result = subprocess.run(
            ['bash', str(self.root / 'run_parallel_krycklan_medium.slurm.sh')],
            env=self.env, text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('HDF5=FALSE', result.stdout)

    def test_launcher_submits_three_arrays(self):
        record = self.root / 'sbatch-args'
        self.env['SBATCH_RECORD'] = str(record)
        self.env['SPAFHY_FOLDER'] = '/incorrect'
        self.executable('sbatch', 'printf "%s|%s|%s\\n" "$SPAFHY_FOLDER" "$SPAFHY_KRYCKLAN_ARRAY" "$*" >> "$SBATCH_RECORD"\n')
        result = subprocess.run(
            ['bash', str(self.root / 'submit_parallel_krycklan.sh')],
            cwd=self.bin, env=self.env, text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / 'logs').is_dir())
        lines = record.read_text().splitlines()
        self.assertEqual(len(lines), 3)
        for line, size in zip(lines, SIZES):
            folder, enabled, args = line.split('|')
            self.assertEqual(folder, str(self.root))
            self.assertEqual(enabled, '1')
            self.assertIn('--array=0-7', args)
            self.assertIn('--export=ALL', args)
            self.assertIn(f'--chdir={self.root}', args)
            self.assertIn(f'--output={self.root}/logs/%x-%A_%a.out', args)
            self.assertIn(f'--error={self.root}/logs/%x-%A_%a.err', args)
            self.assertIn(f'run_parallel_krycklan_{size}.slurm.sh', args)

    def test_array_combinations_and_catchments_override_dotenv(self):
        (self.root / '.env').write_text(
            'SPAFHY_RESOLUTION=999\nSPAFHY_STREAMS=wrong\n'
            'RUN_MODE=all\nCATCHMENT_NOS=999\n'
        )
        self.executable('python', 'printf "CONFIG=%s,%s ARGS=%s\\n" "$SPAFHY_RESOLUTION" "$SPAFHY_STREAMS" "$*"\n')
        self.env['SPAFHY_KRYCKLAN_ARRAY'] = '1'
        for size, (catchments, workers) in SIZES.items():
            for task in range(8):
                with self.subTest(size=size, task=task):
                    self.env['SLURM_ARRAY_TASK_ID'] = str(task)
                    self.env['SLURM_CPUS_PER_TASK'] = str(workers)
                    result = subprocess.run(
                        ['bash', str(self.root / f'run_parallel_krycklan_{size}.slurm.sh')],
                        env=self.env, text=True, capture_output=True,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    resolution = (20, 40, 80, 160)[task % 4]
                    streams = ('channels', '5haStreams')[task // 4]
                    self.assertIn(f'CONFIG={resolution},{streams}', result.stdout)
                    self.assertIn(f'--catchment-nos {catchments} --n-workers {workers}', result.stdout)

    def test_individual_job_does_not_enable_array_mapping(self):
        (self.root / '.env').write_text('SPAFHY_RESOLUTION=80\nSPAFHY_STREAMS=channels\n')
        self.env['SLURM_ARRAY_TASK_ID'] = '7'
        self.env['SLURM_CPUS_PER_TASK'] = '7'
        self.executable('python', 'printf "CONFIG=%s,%s ARGS=%s\\n" "$SPAFHY_RESOLUTION" "$SPAFHY_STREAMS" "$*"\n')
        result = subprocess.run(
            ['bash', str(self.root / 'run_parallel_krycklan_small.slurm.sh')],
            env=self.env, text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('CONFIG=80,channels', result.stdout)
        self.assertIn('--catchment-nos 1 2 3 4 5 6 7 --n-workers 7', result.stdout)

    def test_invalid_array_task_fails(self):
        self.env.update(SPAFHY_KRYCKLAN_ARRAY='1', SLURM_ARRAY_TASK_ID='8')
        result = subprocess.run(
            ['bash', str(self.root / 'run_parallel_krycklan_small.slurm.sh')],
            env=self.env, text=True, capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Invalid SLURM_ARRAY_TASK_ID', result.stderr)


if __name__ == '__main__':
    unittest.main()
