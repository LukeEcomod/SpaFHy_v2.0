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
