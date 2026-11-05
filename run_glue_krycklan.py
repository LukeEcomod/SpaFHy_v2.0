import contextlib
import importlib
import io
import os
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from model_driver import driver
from read_runoff import read_runoff, kge, mbe, rmse
from scripts.create_soil_params import _KMIN, create_soil_params

load_dotenv()

SOIL_TYPE = 'Medium'
DEFAULT_N_SAMPLES = 10_000
DEFAULT_SEED = 42
F_BOUNDS = (6.0, 24.0)
KMAX_BOUNDS = (1e-5, 0.1)

WORKER_CONTEXT = {}


def init_worker(context):
    global WORKER_CONTEXT
    WORKER_CONTEXT = context


def sample_glue_parameters(n_samples, seed=None):
    rng = np.random.default_rng(seed)
    samples = []

    while len(samples) < n_samples:
        batch_size = max(1000, n_samples - len(samples))
        kmax_candidates = 10 ** rng.uniform(np.log10(KMAX_BOUNDS[0]), np.log10(KMAX_BOUNDS[1]), batch_size)
        f_candidates = rng.uniform(F_BOUNDS[0], F_BOUNDS[1], batch_size)

        valid = _KMIN[SOIL_TYPE] < kmax_candidates
        for kmax, f in zip(kmax_candidates[valid], f_candidates[valid]):
            samples.append((float(kmax), float(f)))
            if len(samples) >= n_samples:
                break

    return samples


def extract_runoff_from_results(results):
    surface_runoff = np.nanmean(results['bucket_surface_runoff'], axis=(1, 2))
    netflow_to_ditch = np.nanmean(results['deep_netflow_to_ditch'], axis=(1, 2))
    netflow_to_lake = np.nanmean(results['deep_netflow_to_lake'], axis=(1, 2))
    return surface_runoff, netflow_to_ditch, netflow_to_lake


def run_sample(sample):
    sample_id, kmax, f = sample
    context = WORKER_CONTEXT

    folder = context['folder']
    catchment = context['catchment']
    catchment_no = context['catchment_no']
    runoff_observed = context['q_obs']
    soil_type = context['soil_type']

    kmax_values = {soil_type: kmax}
    f_values = {soil_type: f}
    psoil = create_soil_params(
        kmax_values=kmax_values, f_values=f_values, write=False, verbose=False, plot=False
    )

    parameters_module = importlib.import_module('parameters_krycklan')
    pgen, _, _, _ = parameters_module.parameters(folder, soil_params=psoil)

    dates = pd.date_range(pgen['start_date'], pgen['end_date'])
    spinup_steps = (pd.to_datetime(pgen['spinup_end']) - pd.to_datetime(pgen['start_date'])).days + 1
    sim_dates = dates[spinup_steps:]

    with contextlib.redirect_stdout(io.StringIO()):
        results, _, _, _, _, _ = driver(
            catchment=catchment,
            catchment_no=catchment_no,
            create_ncf=False,
            create_spinup=False,
            output=True,
            folder=folder,
            psoil=psoil,
            save_outputs=False,
        )

    surface_runoff, netflow_to_ditch, netflow_to_lake = extract_runoff_from_results(results)
    q_sim_tot = surface_runoff + netflow_to_ditch + netflow_to_lake
    q_sim_sbsrf = netflow_to_ditch + netflow_to_lake

    q_df = pd.DataFrame(
        {
            'q_obs': runoff_observed.reindex(sim_dates),
            'q_sim': pd.Series(q_sim_tot, index=sim_dates),
            'q_sim_srf': pd.Series(surface_runoff, index=sim_dates),
            'q_sim_sbsrf': pd.Series(q_sim_sbsrf, index=sim_dates),
        }
    ).dropna()

    return {
        'sample_id': sample_id,
        f'kmax_{soil_type}': kmax,
        f'kmin_{soil_type}': _KMIN[soil_type],
        f'f_{soil_type}': f,
        'kge_tot': kge(q_df['q_obs'], q_df['q_sim']),
        'rmse_tot': rmse(q_df['q_obs'], q_df['q_sim']),
        'mbe_tot': mbe(q_df['q_obs'], q_df['q_sim']),
        'kge_sbsrf': kge(q_df['q_obs'], q_df['q_sim_sbsrf']),
        'rmse_sbsrf': rmse(q_df['q_obs'], q_df['q_sim_sbsrf']),
        'mbe_sbsrf': mbe(q_df['q_obs'], q_df['q_sim_sbsrf']),
    }


def summarize_best(summary, metric_name):
    if metric_name in ('rmse', 'mbe'):
        return summary[metric_name].astype(float).idxmin()
    return summary[metric_name].astype(float).idxmax()


if __name__ == '__main__':
    _project_folder = os.environ.get('PROJECT_FOLDER')
    if _project_folder is None:
        raise EnvironmentError('PROJECT_FOLDER is not set. Define it in .env or the environment.')
    folder = str(Path(_project_folder) / 'krycklan')
    catchment_no = 2
    runoff_folder = os.path.join(folder, 'obs', 'DISCHARGE')

    n_samples = int(os.environ.get('N_SAMPLES', DEFAULT_N_SAMPLES))
    seed = int(os.environ.get('GLUE_SEED', DEFAULT_SEED))
    n_workers = min(int(os.environ.get('SLURM_CPUS_PER_TASK', cpu_count())), n_samples)

    q_obs = read_runoff(catchment_no, runoff_folder)['Q']
    samples = sample_glue_parameters(n_samples, seed=seed)
    print(f'Generated {len(samples)} GLUE samples')
    print(f'Using {n_workers} worker processes')

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    output_dir = Path(folder) / 'results' / 'glue_krycklan' / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    context = {
        'folder': folder,
        'catchment': 'krycklan',
        'catchment_no': catchment_no,
        'q_obs': q_obs,
        'soil_type': SOIL_TYPE,
    }

    worker_args = [(index + 1, *sample) for index, sample in enumerate(samples)]
    results = []
    progress_path = output_dir / 'glue_results_partial.csv'

    with Pool(processes=n_workers, initializer=init_worker, initargs=(context,)) as pool:
        for index, result in enumerate(pool.imap_unordered(run_sample, worker_args), start=1):
            results.append(result)
            if index % 100 == 0:
                pd.DataFrame(results).to_csv(progress_path, index=False)
                print(f'Saved progress after {index}/{len(worker_args)} samples')

    summary = pd.DataFrame(results).sort_values('sample_id').reset_index(drop=True)
    summary_path = output_dir / 'glue_results.csv'
    summary.to_csv(summary_path, index=False)

    best_run_tot = summarize_best(summary, 'rmse_tot')
    best_run_sbsrf = summarize_best(summary, 'rmse_sbsrf')

    print('\n--- GLUE summary ---')
    print(f'Output folder: {output_dir}')
    print(f'Samples: {len(summary)}')
    print(f'Best total RMSE run: {int(best_run_tot) + 1}')
    print(f'Best subsurface RMSE run: {int(best_run_sbsrf) + 1}')

    best_tot = summary.loc[best_run_tot]
    best_sbsrf = summary.loc[best_run_sbsrf]
    print('\n--- Best total runoff parameters ---')
    for param, val in best_tot.items():
        if param.startswith(('kmax_', 'kmin_', 'f_')) or param.endswith(('_tot', '_sbsrf')):
            print(f'  {param:<20} {float(val):.6g}')

    print('\n--- Best subsurface runoff parameters ---')
    for param, val in best_sbsrf.items():
        if param.startswith(('kmax_', 'kmin_', 'f_')) or param.endswith(('_tot', '_sbsrf')):
            print(f'  {param:<20} {float(val):.6g}')