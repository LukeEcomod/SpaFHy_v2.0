# Running Krycklan simulations on Roihu

The batch workflow runs Krycklan catchments in parallel. The all-configurations
launcher submits the three catchment groups for each combination of:

- Resolution: 20, 40, 80 and 160 m.
- Stream definition: `channels` and `5haStreams`.

That is three Slurm arrays with eight tasks each: 24 jobs in total. Each task
runs the catchments in one group for one configuration.

## 1. Prepare the Roihu working directory

Transfer the current repository files to Roihu, including the Python files and
all five batch/submission scripts:

- `submit_parallel_krycklan.sh`
- `run_parallel_krycklan.slurm.sh`
- `run_parallel_krycklan_small.slurm.sh`
- `run_parallel_krycklan_medium.slurm.sh`
- `run_parallel_krycklan_large.slurm.sh`

In the repository directory on Roihu, configure `.env` with paths appropriate
to your project. For example, replace the placeholders below:

```bash
PROJECT_FOLDER=/path/to/SpaFHy_RUNS
SPAFHY_RESULTS_FOLDER=/path/to/SpaFHy_RESULTS
SPAFHY_RESULTS_RUN=run_9_2026
```

`PROJECT_FOLDER` is required. Set `SPAFHY_RESULTS_FOLDER` explicitly on Roihu:
its fallback in `parameters_krycklan.py` is a local macOS path. The run name
is optional and defaults to `run_9_2026`. The batch script sources `.env` as
a shell file, so use valid shell assignments and quote values containing
spaces. Alternatively, export these variables before submission.

The input paths expected by the Krycklan parameters are:

```text
$PROJECT_FOLDER/krycklan/
    forcing/FORCING.csv
    gis/20m/
    gis/40m/
    gis/80m/
    gis/160m/
```

Each GIS directory must contain the model's required grids, including the
appropriate `channels` and `5haStreams` grids and related stream grids.
Confirm the data and results paths are accessible from the compute nodes.

The shared batch script loads the `python-data` module when the `module`
command is available, then activates the repository's `.venv` if present.
It does not activate a conda environment. Ensure that the resulting Python
environment contains the model's dependencies.

## 2. Submit all configurations

From the repository directory on Roihu:

```bash
bash submit_parallel_krycklan.sh
```

The launcher creates `logs/` before submitting and prints the Slurm submission
response for each array. Keep the three returned job IDs for monitoring.

Every array uses this task mapping:

| Array task | Resolution | Stream definition |
|---:|---:|---|
| 0 | 20 m | `channels` |
| 1 | 40 m | `channels` |
| 2 | 80 m | `channels` |
| 3 | 160 m | `channels` |
| 4 | 20 m | `5haStreams` |
| 5 | 40 m | `5haStreams` |
| 6 | 80 m | `5haStreams` |
| 7 | 160 m | `5haStreams` |

The array task determines the resolution and stream definition even if
`SPAFHY_RESOLUTION` or `SPAFHY_STREAMS` is set in `.env`. Do not edit the
parameter file between submissions. Other model settings, including
simulation dates, are read from `parameters_krycklan.py` when jobs run.
Avoid changing code, input data or shared settings while jobs are queued or
running if they are intended to use the same setup.

### Catchment groups and resources

The launcher submits each group script as an array; its Slurm directives apply
to every task in that array.

| Group | Catchment numbers | CPUs per task | Memory | Time limit |
|---|---|---:|---:|---:|
| Small | 1, 2, 3, 4, 5, 6, 7 | 7 | 16 GB | 1 hour |
| Medium | 8, 9, 12, 13, 20, 21 | 6 | 16 GB | 4 hours |
| Large | 141, 142, 143, 151, 152, 153, 161, 162, 163, 164, 22, 10 | 12 | 64 GB | 32 hours |

Catchment 3 is excluded from the small group at 160 m for both stream
definitions. It remains in the small group at 20, 40 and 80 m. This exclusion
also applies when submitting the small group individually at 160 m; it does
not apply to direct Python runs or direct submissions of the shared script.

All three request one node, account `project_2000908`, and partition `small`.
The medium script also sets `HDF5_USE_FILE_LOCKING=FALSE`. Check that these
requests and the account are appropriate for the current Roihu configuration.
The shared script's own Slurm directives apply only when it is submitted
directly, not when it is called by a group script.

## 3. Monitor jobs and inspect output

Check queued and running jobs:

```bash
squeue -u "$USER"
```

For a submitted array, inspect accounting information using its job ID:

```bash
sacct -j <array-job-id> --format=JobID,JobName,State,ExitCode,Elapsed
```

Array logs are created under the repository's `logs/` directory:

```text
logs/<job-name>-<array-job-id>_<task-id>.out
logs/<job-name>-<array-job-id>_<task-id>.err
```

The beginning of each output log identifies its resolution, stream definition
and array task. Check both the Slurm state and the logs for failures; a
submitted job is not necessarily a completed simulation.

Results are written under:

```text
$SPAFHY_RESULTS_FOLDER/krycklan/$SPAFHY_RESULTS_RUN/<streams>_<resolution>m/
```

For example, the 40 m `5haStreams` task writes under `5haStreams_40m/`.
Each catchment simulation creates its own timestamped subdirectory containing
the netCDF output and parameter text files. The output log reports the
resulting file paths.

## Running one group and configuration

Individual group scripts remain usable without arrays. Set
`SPAFHY_RESOLUTION` and `SPAFHY_STREAMS` in `.env` or export them before
submission. They default to `20` and `channels` when unset. For example, to
run the small group, first configure the desired values, then submit:

```bash
mkdir -p logs
sbatch run_parallel_krycklan_small.slurm.sh
```

Substitute the medium or large script as needed. `logs/` must exist **before**
`sbatch`, because Slurm opens the output files before the script runs.
Values sourced from `.env` override values inherited from the submitting
shell in these batch scripts. Avoid setting `RUN_MODE` or `CATCHMENT_NOS` in
`.env` when using the individual group scripts, as those values can override
the group selection. Individual-job logs use `logs/<job-name>-<job-id>.out`
and `.err`.

## Local runs

The array launcher is for Slurm; local Python runs do not need it. Existing
local defaults remain 20 m and `channels`. To select a different configuration
for one local run, for example:

```bash
SPAFHY_RESOLUTION=40 SPAFHY_STREAMS=5haStreams \
    python run_parallel_krycklan.py --catchment-no 1
```

Set `PROJECT_FOLDER` and, if needed, `SPAFHY_RESULTS_FOLDER` for the local
environment as well. The Python entry point accepts `--help` for the available
catchment and output options.

## Stitching subcatchment results

After the simulation jobs have **completed successfully**, the stitching
array combines the subcatchment NetCDF files into one file per configuration.
Transfer `stitch_krycklan.py`, `stitch_krycklan.slurm.sh` and the current
`iotools.py` to Roihu. From the repository directory:

```bash
mkdir -p logs
sbatch stitch_krycklan.slurm.sh
```

This submits one eight-task array using the same task mapping as above. Each
task works on one resolution/stream combination. The script uses the same
`python-data` module and optional repository `.venv` as the simulation jobs.
It does not wait for simulations automatically.

By default, the input run directory is
`$SPAFHY_RESULTS_FOLDER/krycklan/$SPAFHY_RESULTS_RUN`, matching the simulation
output settings. `SPAFHY_RESULTS_RUN` defaults to `run_9_2026`. The notebook's
older directory layout includes an extra `results` component; pass `--run-dir`
if that is where the data actually are:

```bash
sbatch stitch_krycklan.slurm.sh --run-dir /path/to/SpaFHy_RESULTS/krycklan/results/run_9_2026
```

For each task the input is `<run-dir>/<streams>_<resolution>m/`, searched
recursively for **all** `.nc` files. Ensure it contains only the intended
subcatchment results: duplicate runs, spinup files, or partial results may
produce an incorrect merge. The stitcher does not verify that all expected
catchments are present. Catchment 3 is absent from 160 m simulation output.
The output is `<run-dir>/stitched/<streams>_<resolution>m.nc`. Existing output
files are replaced after a successful stitch; a failed stitch leaves the
previous output intact. If multiple tasks target the same output concurrently,
the last successful task to finish replaces the others. Avoid submitting
overlapping tasks when their inputs or date ranges differ.

The default time slice is the notebook's **2015-04-01 through 2015-09-01**,
inclusive. Override it by passing both dates, for example:

```bash
sbatch stitch_krycklan.slurm.sh --start-date 2015-05-01 --end-date 2015-08-31
```

The batch script requests one CPU, 64 GB and 32 hours per task on the `small`
partition. These are starting values, not measured requirements: the existing
stitcher allocates full-domain arrays and fine resolutions may need much more
memory or time. Check `sacct` and adjust with Slurm options where appropriate,
for example `sbatch --mem=128G --time=48:00:00 stitch_krycklan.slurm.sh`,
subject to Roihu's partition limits. To limit concurrent tasks, submit with
`sbatch --array=0-7%2 stitch_krycklan.slurm.sh`; to rerun only specific tasks,
use e.g. `sbatch --array=3,7 stitch_krycklan.slurm.sh`.

Check `squeue`, `sacct`, and `logs/stitch-krycklan-<array-job-id>_<task-id>.out`
and `.err` for completion and errors. Tasks use separate temporary directories
under `stitched/` to prevent collisions, and publish the final output only after
the stitcher succeeds. A forcibly terminated job may leave a hidden
`.stitch-...` temporary directory; inspect and remove that task's directory
after confirming it is no longer running.

The Python script can also stitch one configuration locally:

```bash
python stitch_krycklan.py --resolution 20 --streams channels --run-dir /path/to/run
```
