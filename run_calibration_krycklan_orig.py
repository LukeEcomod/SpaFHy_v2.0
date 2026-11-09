import os
import itertools
import numpy as np
import pandas as pd
from scripts.create_soil_params import create_soil_params
from pathlib import Path
from model_driver import parallel_driver
from read_runoff import read_runoff, kge, rmse, mbe, extract_discharge_results_2D
import matplotlib.pyplot as plt
from dotenv import load_dotenv
load_dotenv()

if __name__ == '__main__':
    io_path = str(Path(os.getenv('PROJECT_FOLDER')))
    folder = os.path.join(io_path, 'krycklan')  # io repo
    catchment_no = 2 # C2 catchment
    runoff_folder = os.path.join(folder, 'obs', 'DISCHARGE')  # runoff data folder

    # evaluate or plot?
    evaluate = True
    ev_metric = 'rmse'
    plot = True

    ### CALIBRATION SETUP ###
    # soil type to calibrate (add more blocks below for additional soil types)
    soil_type = 'Till'
    # f values for calibration
    f_range = np.array([6., 8., 12., 14., 16.]) # up to crazy range
    #f_range = np.array([2., 4., 6., 8., 10., 12.]) # reasonable range
    f_range = np.array([14.]) # small test
    # kmax values for calibration
    kmax_range = np.array([1e-5, 1e-4, 1e-3, 1e-2, 0.1]) # up to crazy range
    #kmax_range = np.array([1e-5, 5e-4, 1e-4, 5e-4]) # reasonable range
    kmax_range = np.array([0.1, 0.01]) # small test
    # kmin values for calibration
    kmin_range = np.array([1e-9, 1e-8, 1e-7, 1e-6]) # 
    #kmin_range = np.array([1e-7, 1e-6, 1e-5, 5e-5]) # reasonable range
    kmin_range = np.array([1e-7]) # small test
    # all possible combinations of kmax, kmin and f values (kmin must be strictly less than kmax)
    combinations = []
    for kmax, kmin, f in itertools.product(kmax_range, kmin_range, f_range):
        if kmin < kmax:
            combinations.append((kmax, kmin, f))
    print(f"Total combinations: {len(combinations)}")

    # reading runoff data
    q_obs = read_runoff(catchment_no, runoff_folder)

    out_dirs = []  # track simulation folders for post-loop evaluation

    for i, (kmax, kmin, f) in enumerate(combinations, start=1):
        kmax_values = {soil_type: kmax}
        kmin_values = {soil_type: kmin}
        f_values = {soil_type: f}

        # Write soil parameters and run the model
        print(f"\nRunning simulation {i}/{len(combinations)}")
        create_soil_params(kmax_values, f_values, write=True, verbose=False)
        outputfile = parallel_driver(catchment='krycklan', catchment_no=catchment_no, create_ncf=True, create_spinup=False, output=True, folder=folder)

        # Read simulated discharge and combine with observations
        surface_runoff, netflow_to_ditch, netflow_to_lake = extract_discharge_results_2D(outputfile[0])
        q_sim_tot = surface_runoff + netflow_to_ditch + netflow_to_lake
        q_sim_sbrf = netflow_to_ditch + netflow_to_lake
        q_df = pd.concat({'q_obs': q_obs['Q'], 
                          'q_sim': q_sim_tot, 
                          'q_sim_srf': surface_runoff, 
                          'q_sim_sbsrf': q_sim_sbrf}, axis=1, join='inner').dropna()
        q_df.index.name = 'time'
        kge_val_tot    = kge(q_df['q_obs'],  q_df['q_sim'])
        rmse_val_tot   = rmse(q_df['q_obs'], q_df['q_sim'])
        mbe_val_tot    = mbe(q_df['q_obs'],  q_df['q_sim'])
        kge_val_sbsrf  = kge(q_df['q_obs'],  q_df['q_sim_sbsrf'])
        rmse_val_sbsrf = rmse(q_df['q_obs'], q_df['q_sim_sbsrf'])
        mbe_val_sbsrf  = mbe(q_df['q_obs'],  q_df['q_sim_sbsrf'])

        # Save parameters and runoff comparison next to the output file
        out_dir = Path(outputfile[0]).parent
        params = {}
        for st, val in kmax_values.items():
            params[f'kmax_{st}'] = val
        for st, val in kmin_values.items():
            params[f'kmin_{st}'] = val
        for st, val in f_values.items():
            params[f'f_{st}'] = val
        params['kge_tot']    = kge_val_tot
        params['rmse_tot']   = rmse_val_tot
        params['mbe_tot']    = mbe_val_tot
        params['kge_sbsrf']  = kge_val_sbsrf
        params['rmse_sbsrf'] = rmse_val_sbsrf
        params['mbe_sbsrf']  = mbe_val_sbsrf
        params_df = pd.DataFrame.from_dict(params, orient='index', columns=['value'])
        params_df.index.name = 'parameter'
        params_df.to_csv(out_dir / 'calibration_params.txt')
        q_df.to_csv(out_dir / 'runoff.txt')
        out_dirs.append(out_dir)

    if evaluate:
        all_params = []
        for d in out_dirs:
            df = pd.read_csv(d / 'calibration_params.txt', index_col='parameter')
            df.columns = [d.name]
            all_params.append(df)
        summary = pd.concat(all_params, axis=1).T
        summary.index.name = 'run'
        ev_var_tot = f'{ev_metric}_tot'
        ev_var_sbsrf = f'{ev_metric}_sbsrf'
        best_fn = summary[ev_var_tot].astype(float).idxmin if ev_metric in ('rmse', 'mbe') else summary[ev_var_tot].astype(float).idxmax
        best_run_tot   = best_fn()
        best_fn = summary[ev_var_sbsrf].astype(float).idxmin if ev_metric in ('rmse', 'mbe') else summary[ev_var_sbsrf].astype(float).idxmax
        best_run_sbsrf = best_fn()
        print('\n--- Calibration ranges ---')
        print(f'  {"kmax_" + soil_type:<20} {kmax_range}')
        print(f'  {"kmin_" + soil_type:<20} {kmin_range}')
        print(f'  {"f_" + soil_type:<20} {f_range}')
        print(f'  Total combinations: {len(combinations)}')
        print(f'\n--- Best {ev_metric.upper()} (total runoff) ---')
        best = summary.loc[best_run_tot]
        for param, val in best.items():
            v = float(val)
            formatted = f'{v:.2e}' if abs(v) < 0.01 and v != 0 else f'{v:.4f}'
            print(f'  {param:<20} {formatted}')
        print(f'\n--- Best {ev_metric.upper()} (subsurface runoff) ---')
        best = summary.loc[best_run_sbsrf]
        for param, val in best.items():
            v = float(val)
            formatted = f'{v:.2e}' if abs(v) < 0.01 and v != 0 else f'{v:.4f}'
            print(f'  {param:<20} {formatted}')

    if plot:
        fig, (ax, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
        obs_plotted = False
        for d in out_dirs:
            q = pd.read_csv(d / 'runoff.txt', index_col='time', parse_dates=True)
            is_best_tot   = (d.name == best_run_tot)
            is_best_sbsrf = (d.name == best_run_sbsrf)
            is_best = is_best_tot or is_best_sbsrf
            color = 'black' if is_best_tot else ('tab:red' if is_best_sbsrf else 'grey')
            ax.plot(q.index, q['q_sim'],
                    color=color,
                    linewidth=1.5 if is_best else 0.7,
                    zorder=3 if is_best else 2,
                    label=('best total' if is_best_tot else ('best subsurface' if is_best_sbsrf else '_nolegend_')))
            if not obs_plotted:
                ax.plot(q.index, q['q_obs'], color='steelblue', linewidth=1.2,
                        zorder=4, label='observed')
                obs_plotted = True

        # second subplot: stacked for best total runoff
        for d in out_dirs:
            if d.name == best_run_tot:
                best_dir_tot = d
                break
        q_best_tot = pd.read_csv(best_dir_tot / 'runoff.txt', index_col='time', parse_dates=True)
        ax2.stackplot(q_best_tot.index,
                      q_best_tot['q_sim_sbsrf'], q_best_tot['q_sim_srf'],
                      labels=['subsurface', 'surface'],
                      colors=['tab:orange', 'tab:green'], alpha=0.8)
        ax2.plot(q_best_tot.index, q_best_tot['q_obs'], color='steelblue', linewidth=1.2,
                 zorder=4, label='observed')

        # third subplot: stacked for best subsurface runoff
        for d in out_dirs:
            if d.name == best_run_sbsrf:
                best_dir_sbsrf = d
                break
        q_best_sbsrf = pd.read_csv(best_dir_sbsrf / 'runoff.txt', index_col='time', parse_dates=True)
        ax3.stackplot(q_best_sbsrf.index,
                      q_best_sbsrf['q_sim_sbsrf'], q_best_sbsrf['q_sim_srf'],
                      labels=['subsurface', 'surface'],
                      colors=['tab:orange', 'tab:green'], alpha=0.8)
        ax3.plot(q_best_sbsrf.index, q_best_sbsrf['q_obs'], color='steelblue', linewidth=1.2,
                 zorder=4, label='observed')

        ax.set_ylabel('Runoff [mm/d]')
        ax.legend(loc='upper right')
        ax2.set_ylabel('Runoff [mm/d]')
        ax2.set_title(f'Best total {ev_metric.upper()}', fontsize=9)
        ax2.legend()
        ax3.set_ylabel('Runoff [mm/d]')
        ax3.set_title(f'Best subsurface {ev_metric.upper()}', fontsize=9)
        ax3.legend()

        # Ksat inset in ax — all runs as grey, best as black
        max_depth = 5.0
        z = np.linspace(0, max_depth, 500)
        ax_inset = ax.inset_axes([0.3, 0.6, 0.2, 0.3])
        for run, row in summary.iterrows():
            kmax_r = float(row[f'kmax_{soil_type}'])
            kmin_r = float(row[f'kmin_{soil_type}'])
            f_r    = float(row[f'f_{soil_type}'])
            K_r = (kmax_r - kmin_r) * np.exp(-f_r * z) + kmin_r
            is_best_tot   = (run == best_run_tot)
            is_best_sbsrf = (run == best_run_sbsrf)
            is_best = is_best_tot or is_best_sbsrf
            color = 'black' if is_best_tot else ('tab:red' if is_best_sbsrf else 'grey')
            ax_inset.plot(K_r, -z,
                          linewidth=1.5 if is_best else 0.5,
                          color=color,
                          zorder=3 if is_best else 2)
        ax_inset.set_xscale('log')
        ax_inset.set_xlabel('$K_{sat}$', fontsize=7)
        ax_inset.set_ylabel('Depth [m]', fontsize=7)
        ax_inset.tick_params(labelsize=6)
        ax_inset.grid(True, which='both', alpha=0.3, linestyle='--')
        ax_inset.set_title(f'{soil_type}', fontsize=7)

        fig_path = os.path.join(folder, 'figs', f'calibration_q_{ev_metric}.png')
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'Plot saved to {fig_path}')