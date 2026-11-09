import argparse
import contextlib
import importlib
import io
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from model_driver import driver
from read_runoff import extract_discharge_results_2D, kge, mbe, read_runoff, rmse
from scripts.create_soil_params import create_soil_params

load_dotenv()

MINIMIZE_METRICS = {'rmse_tot', 'mbe_tot', 'rmse_sbsrf', 'mbe_sbsrf'}



def resolve_model_root():
    return Path(__file__).resolve().parents[1] / 'SpaFHy_RUNS' / 'krycklan'


def read_glue_results(glue_results_csv, soil_type):
    glue_results_csv = Path(glue_results_csv).expanduser().resolve()
    if not glue_results_csv.exists():
        raise FileNotFoundError(f'GLUE results file not found: {glue_results_csv}')

    summary = pd.read_csv(glue_results_csv)
    required_columns = {'sample_id', f'kmax_{soil_type}', f'kmin_{soil_type}', f'f_{soil_type}'}
    missing = required_columns - set(summary.columns)
    if missing:
        raise ValueError(f'Missing required columns in {glue_results_csv}: {sorted(missing)}')

    return summary, glue_results_csv


def select_best_rows(summary, n_best, metric_name):
    if metric_name not in summary.columns:
        raise ValueError(f'Metric column not found in GLUE results: {metric_name}')

    ascending = metric_name in MINIMIZE_METRICS
    ranked = summary.dropna(subset=[metric_name]).sort_values(metric_name, ascending=ascending)
    return ranked.head(n_best).copy()


def build_parameter_dict(row, soil_type):
    kmax = float(row[f'kmax_{soil_type}'])
    kmin = float(row[f'kmin_{soil_type}'])
    f = float(row[f'f_{soil_type}'])

    return {
        'kmax_values': {soil_type: kmax},
        'kmin_values': {soil_type: kmin},
        'f_values': {soil_type: f},
    }


def run_single_simulation(
    row, folder, catchment, catchment_no, q_obs, soil_type, create_ncf=False
):
    """Run a single SpaFHy simulation for given parameters.
    
    Args:
        row: DataFrame row with kmax, kmin, f parameters
        folder: Path to model folder
        catchment: Catchment name (e.g. 'krycklan')
        catchment_no: Catchment number
        q_obs: Observed discharge Series
        soil_type: Soil type name (e.g. 'Till')
        create_ncf: Whether to create netcdf output
        
    Returns:
        Tuple of (q_df, params, output_path_or_None)
        - q_df: DataFrame with q_obs, q_sim, q_sim_srf, q_sim_sbsrf (spinup-trimmed)
        - params: Dict with parameters and 6 metrics (kge/rmse/mbe × tot/sbsrf)
        - output_path: NetCDF path if create_ncf=True, else None
    """
    param_sets = build_parameter_dict(row, soil_type)

    psoil = create_soil_params(
        kmax_values=param_sets['kmax_values'],
        kmin_values=param_sets['kmin_values'],
        f_values=param_sets['f_values'],
        write=False,
        verbose=False,
        plot=False,
    )

    parameters_module = importlib.import_module('parameters_krycklan')
    pgen, _, _, _ = parameters_module.parameters(folder, soil_params=psoil)
    dates = pd.date_range(pgen['start_date'], pgen['end_date'])
    spinup_steps = (pd.to_datetime(pgen['spinup_end']) - pd.to_datetime(pgen['start_date'])).days + 1
    sim_dates = dates[spinup_steps:]

    with contextlib.redirect_stdout(io.StringIO()):
        result_or_tuple = driver(
            catchment=catchment,
            catchment_no=catchment_no,
            create_ncf=create_ncf,
            create_spinup=False,
            output=True,
            folder=folder,
            psoil=psoil,
            save_outputs=create_ncf,
        )

    if create_ncf:
        if isinstance(result_or_tuple, (list, tuple)):
            output_path = result_or_tuple[0]
        else:
            output_path = result_or_tuple
        surface_runoff, netflow_to_ditch, netflow_to_lake = extract_discharge_results_2D(output_path)
    else:
        results, _, _, _, _, _ = result_or_tuple
        output_path = None
        surface_runoff, netflow_to_ditch, netflow_to_lake = extract_runoff_from_results(results)

    q_sim_tot = surface_runoff + netflow_to_ditch + netflow_to_lake
    q_sim_sbsrf = netflow_to_ditch + netflow_to_lake

    q_df = pd.concat(
        {
            'q_obs': q_obs.reindex(sim_dates),
            'q_sim': pd.Series(q_sim_tot, index=sim_dates),
            'q_sim_srf': pd.Series(surface_runoff, index=sim_dates),
            'q_sim_sbsrf': pd.Series(q_sim_sbsrf, index=sim_dates),
        },
        axis=1,
        join='inner',
    ).dropna()
    q_df.index.name = 'time'

    params = {
        f'kmax_{soil_type}': float(row[f'kmax_{soil_type}']),
        f'kmin_{soil_type}': float(row[f'kmin_{soil_type}']),
        f'f_{soil_type}': float(row[f'f_{soil_type}']),
        'kge_tot': kge(q_df['q_obs'], q_df['q_sim']),
        'rmse_tot': rmse(q_df['q_obs'], q_df['q_sim']),
        'mbe_tot': mbe(q_df['q_obs'], q_df['q_sim']),
        'kge_sbsrf': kge(q_df['q_obs'], q_df['q_sim_sbsrf']),
        'rmse_sbsrf': rmse(q_df['q_obs'], q_df['q_sim_sbsrf']),
        'mbe_sbsrf': mbe(q_df['q_obs'], q_df['q_sim_sbsrf']),
    }

    return q_df, params, output_path


def extract_runoff_from_results(results):
    """Extract runoff components from model results dict."""
    import numpy as np
    surface_runoff = np.nanmean(results['bucket_surface_runoff'], axis=(1, 2))
    netflow_to_ditch = np.nanmean(results['deep_netflow_to_ditch'], axis=(1, 2))
    netflow_to_lake = np.nanmean(results['deep_netflow_to_lake'], axis=(1, 2))
    return surface_runoff, netflow_to_ditch, netflow_to_lake


def find_existing_ncf_files(output_folder, suitable_df, soil_type):
    """Find which simulations already have netcdf files in output_folder.
    
    Returns:
        Set of indices that have existing ncf files
    """
    if not Path(output_folder).exists():
        return set()
    
    output_folder = Path(output_folder)
    existing = set()
    
    for idx, (sim_idx, row) in enumerate(suitable_df.iterrows()):
        kmax = float(row[f'kmax_{soil_type}'])
        kmin = float(row[f'kmin_{soil_type}'])
        f = float(row[f'f_{soil_type}'])
        pattern = f"kmax_{kmax:.6e}_kmin_{kmin:.6e}_f_{f:.6f}*.nc"
        
        if list(output_folder.glob(pattern)):
            existing.add(sim_idx)
    
    return existing


def run_batch_simulations(
    suitable_df,
    folder,
    q_obs,
    output_folder=None,
    save_ncf=False,
    catchment='krycklan',
    catchment_no=2,
    soil_type='Till',
    skip_existing=True,
    verbose=True,
):
    """Run batch of SpaFHy simulations for suitable parameter sets.
    
    Args:
        suitable_df: DataFrame of suitable simulations (from GLUE analysis)
        folder: Path to model folder
        q_obs: Observed discharge Series
        output_folder: Folder to save netcdf outputs (if save_ncf=True)
        save_ncf: Whether to save netcdf files
        catchment: Catchment name
        catchment_no: Catchment number
        soil_type: Soil type name
        skip_existing: Skip re-running if netcdf already exists
        verbose: Print progress messages
        
    Returns:
        Dict mapping row index → (q_df, params, output_path_or_None)
    """
    if save_ncf and output_folder is None:
        raise ValueError("output_folder required when save_ncf=True")
    
    if save_ncf:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
    
    existing = set()
    if skip_existing and save_ncf:
        existing = find_existing_ncf_files(output_folder, suitable_df, soil_type)
        if existing:
            if verbose:
                print(f"Found {len(existing)} existing simulations in {output_folder}")
    
    results = {}
    total = len(suitable_df)
    
    for idx, (sim_idx, row) in enumerate(suitable_df.iterrows(), 1):
        if skip_existing and sim_idx in existing:
            if verbose:
                print(f"[{idx}/{total}] Skipping existing simulation (index {sim_idx})")
            results[sim_idx] = None
            continue
        
        if verbose:
            print(f"[{idx}/{total}] Running simulation (index {sim_idx})...")
        
        q_df, params, output_path = run_single_simulation(
            row=row,
            folder=folder,
            catchment=catchment,
            catchment_no=catchment_no,
            q_obs=q_obs,
            soil_type=soil_type,
            create_ncf=save_ncf,
        )
        
        results[sim_idx] = (q_df, params, output_path)
    
    return results


def save_run_outputs(output_ncf_path, q_df, params):
    """Save calibration params and runoff comparison to results folder."""
    out_dir = Path(output_ncf_path).parent

    params_df = pd.DataFrame.from_dict(params, orient='index', columns=['value'])
    params_df.index.name = 'parameter'
    params_df.to_csv(out_dir / 'calibration_params.txt')
    q_df.to_csv(out_dir / 'runoff.txt')

    return out_dir


def run_best_row(row, folder, catchment, catchment_no, q_obs, soil_type):
    """Run best row and save outputs (legacy CLI function)."""
    q_df, params, output_path = run_single_simulation(
        row=row,
        folder=folder,
        catchment=catchment,
        catchment_no=catchment_no,
        q_obs=q_obs,
        soil_type=soil_type,
        create_ncf=True,
    )
    return save_run_outputs(output_path, q_df, params)



def parse_args():
    parser = argparse.ArgumentParser(
        description='Run SpaFHy simulations for the best GLUE parameter sets and save outputs like the calibration script.'
    )
    parser.add_argument('glue_results_csv', help='Path to glue_results.csv produced by run_glue_krycklan.py')
    parser.add_argument('n_best', type=int, help='Number of best simulations to run')
    parser.add_argument(
        '--metric',
        default='rmse_tot',
        help='Metric column used to rank GLUE results (default: rmse_tot)',
    )
    parser.add_argument(
        '--catchment-no',
        type=int,
        default=2,
        help='Catchment number to simulate (default: 2)',
    )
    parser.add_argument(
        '--soil-type',
        default='Till',
        help='Soil type to extract from the GLUE result columns (default: Till)',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    folder = str(resolve_model_root())
    runoff_folder = str(Path(folder) / 'obs' / 'DISCHARGE')

    q_obs = read_runoff(args.catchment_no, runoff_folder)
    glue_results, glue_results_path = read_glue_results(args.glue_results_csv, args.soil_type)

    selected = select_best_rows(glue_results, args.n_best, args.metric)
    print(f'Reading GLUE results from: {glue_results_path}')
    print(f'Selecting top {len(selected)} rows by {args.metric}')

    out_dirs = []
    for rank, (_, row) in enumerate(selected.iterrows(), start=1):
        print(f'\nRunning best simulation {rank}/{len(selected)}')
        out_dir = run_best_row(
            row=row,
            folder=folder,
            catchment='krycklan',
            catchment_no=args.catchment_no,
            q_obs=q_obs,
            soil_type=args.soil_type,
        )
        out_dirs.append(out_dir)
        print(f'  Saved outputs to {out_dir}')

    print('\n--- Finished best-parameter runs ---')
    for out_dir in out_dirs:
        print(f'  {out_dir}')