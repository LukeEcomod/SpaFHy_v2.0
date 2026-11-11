#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
create_soil_params.py
Generates soil_params.py (org_properties, root_properties, deep_properties)
from the parameter definitions below.

Usage:
    python create_soil_params.py                    # uses default parameters
    from create_soil_params import create_soil_params
    create_soil_params(kmax_values={...}, kmin_values={...}, f_values={...})  # override for calibration

Output:
    SpaFHy_v2.0/soil_params.py
"""

from pathlib import Path
import sys
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
scripts_folder = Path(__file__).parent
project_root   = scripts_folder.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(scripts_folder))

from soil_helpers import (
    discretized_depths, round_sig, expand_pf_layers,
    deep_properties_function_text, org_properties_function_text, root_properties_function_text,
    wrc, plot_ksat_profiles,
)

# Unit conversion: alpha [kPa-1] → alpha [cm-1]
kPa_to_cm = 1.0 / (100.0 / 9.81)   # = 0.09810

# ── Default parameters (override via function arguments during calibration) ────

_DEFAULT_KMAX = {
    'Bedrock':  1e-6,
    'Peat':     1e-3,
    'Silt':     1e-4,
    'Till':     1e-2,
    'Sand':     1e-3,
}

_DEFAULT_F = {                  # Ksat decay coefficient [m-1]
    'Bedrock':  10.0,
    'Peat':     6.0,
    'Silt':      4.0,
    'Till':      7.3,
    'Sand':     1.5,
}

_KMIN = {
    'Bedrock':  1e-9,
    'Peat':     1e-9,
    'Silt':      1e-9,
    'Till':      1e-9,
    'Sand':     1e-9,
}

_MAX_DEPTH = {
    'Bedrock':  15.0,
    'Peat':     15.0,
    'Silt':      15.0,
    'Till':      15.0,
    'Sand':     15.0,
}

_CONST_SURF_KMAX = {             # depth [m] over which Ksat is held constant at Kmax before exponential decay
    'Bedrock':  0.1,
    'Peat':     0.1,
    'Silt':     0.3,
    'Till':     0.3,
    'Sand':     0.3,
}

# van Genuchten / ThetaS – Sources:
#   Bedrock – assumed
#   Peat    – Leppä et al. 2020, Sphagnum
#   Silt    – Launiainen et al. 2022, C3
#   Till    – Launiainen et al. 2022, C5
#   Sand    – Launiainen et al. 2022, C4

_THETAS_MAX = {
    'Bedrock':  0.20,
    'Peat':     0.89,
    'Silt':      0.46,
    'Till':      0.54,
    'Sand':     0.47,
}

_THETAS_MIN = {
    'Bedrock':  0.10,
    'Peat':     0.70,
    'Silt':      0.2,
    'Till':      0.25,
    'Sand':     0.25,
}

_F_THETA = {
    'Bedrock':  1.0,
    'Peat':     1.0,
    'Silt':      1.0,
    'Till':      1.0,
    'Sand':     1.0,
}

_PF_ALPHA = {                   # cm-1 (converted from kPa-1 where noted)
    'Bedrock': 0.024,
    'Peat':    0.07,
    'Silt':    2.02 * kPa_to_cm,
    'Till':    3.35 * kPa_to_cm,
    'Sand':    4.49 * kPa_to_cm,
}

_PF_N = {
    'Bedrock': 1.20,
    'Peat':    1.37,
    'Silt':    1.07,
    'Till':    1.18,
    'Sand':    1.27,
}

_PF_WR = {
    'Bedrock': 0.0,
    'Peat':    0.098,
    'Silt':    0.0,
    'Till':    0.0,
    'Sand':    0.0,
}


def create_soil_params(
    kmax_values=None,
    kmin_values=None,
    f_values=None,
    const_surf_values=None,
    max_depth_values=None,
    write=True,
    verbose=True,
    plot=True,
    fig_dir=None,
):
    """
    Build and optionally write soil_params.py.

    Parameters
    ----------
    kmax_values : dict, optional
        Surface saturated hydraulic conductivity [m/s] per soil type.
        Defaults to the values defined at module level (_DEFAULT_KMAX).
    kmin_values : dict, optional
        Minimum saturated hydraulic conductivity [m/s] per soil type.
        Defaults to the values defined at module level (_KMIN).
    f_values : dict, optional
        Ksat exponential decay coefficient [m-1] per soil type.
        Defaults to the values defined at module level (_DEFAULT_F).
    const_surf_values : dict, optional
        Depth [m] over which Ksat is held constant at Kmax per soil type.
        Defaults to the values defined at module level (_CONST_SURF_KMAX).
    max_depth_values : dict, optional
        Maximum soil profile depth [m] per soil type.
        Defaults to the values defined at module level (_MAX_DEPTH).
    write : bool
        If True (default), write soil_params.py to SpaFHy_v2.0/.
    verbose : bool
        If True (default), print the fc/wp table and the output path.
    plot : bool
        If True (default), plot the exponential Ksat(z) profile for each soil type.
    fig_dir : str or Path, optional
        Directory to save the Ksat profile plot. Defaults to SpaFHy_v2.0/figs/.

    Returns
    -------
    dict  – {'org': org_properties, 'root': root_properties, 'deep': deep_properties_exp}
    """
    # Apply defaults for any key not supplied
    kmax = _DEFAULT_KMAX.copy()
    if kmax_values:
        kmax.update(kmax_values)

    kmin = _KMIN.copy()
    if kmin_values:
        kmin.update(kmin_values)

    f = _DEFAULT_F.copy()
    if f_values:
        f.update(f_values)

    const_surf_depth = _CONST_SURF_KMAX.copy()
    if const_surf_values:
        const_surf_depth.update(const_surf_values)

    max_depth = _MAX_DEPTH.copy()
    if max_depth_values:
        max_depth.update(max_depth_values)

    vertical_ksat_factor = 0.1
    exponential_ThetaS   = True

    # ── Soil property dicts ────────────────────────────────────────────────────
    org_properties = {
        'Bedrock': {'org_id': 1, 'org_depth': 0.05, 'org_poros': 0.9, 'org_fc': None, 'org_rw': 0.11, 'org_ksat': vertical_ksat_factor * kmax['Bedrock'], 'org_beta': 6.0},
        'Peat':    {'org_id': 2, 'org_depth': 0.05, 'org_poros': 0.9, 'org_fc': None, 'org_rw': 0.11, 'org_ksat': vertical_ksat_factor * kmax['Peat'],    'org_beta': 6.0},
        'Silt':    {'org_id': 3, 'org_depth': 0.05, 'org_poros': 0.9, 'org_fc': None, 'org_rw': 0.11, 'org_ksat': vertical_ksat_factor * kmax['Silt'],    'org_beta': 6.0},
        'Till':    {'org_id': 4, 'org_depth': 0.05, 'org_poros': 0.9, 'org_fc': None, 'org_rw': 0.11, 'org_ksat': vertical_ksat_factor * kmax['Till'],  'org_beta': 6.0},
        'Sand':    {'org_id': 5, 'org_depth': 0.05, 'org_poros': 0.9, 'org_fc': None, 'org_rw': 0.11, 'org_ksat': vertical_ksat_factor * kmax['Sand'],  'org_beta': 6.0},
    }

    root_properties = {
        'Bedrock': {'root_id': 1, 'root_poros': _THETAS_MAX['Bedrock'], 'root_fc': None, 'root_wp': None, 'root_ksat': vertical_ksat_factor * kmax['Bedrock'], 'root_beta': 4.0, 'root_alpha': _PF_ALPHA['Bedrock'], 'root_n': _PF_N['Bedrock'], 'root_wr': _PF_WR['Bedrock'], 'root_depth': 0.3},
        'Peat':    {'root_id': 2, 'root_poros': _THETAS_MAX['Peat'],    'root_fc': None, 'root_wp': None, 'root_ksat': vertical_ksat_factor * kmax['Peat'],    'root_beta': 4.0, 'root_alpha': _PF_ALPHA['Peat'],    'root_n': _PF_N['Peat'],    'root_wr': _PF_WR['Peat'],    'root_depth': 0.3},
        'Silt':    {'root_id': 3, 'root_poros': _THETAS_MAX['Silt'],    'root_fc': None, 'root_wp': None, 'root_ksat': vertical_ksat_factor * kmax['Silt'],    'root_beta': 4.0, 'root_alpha': _PF_ALPHA['Silt'],    'root_n': _PF_N['Silt'],    'root_wr': _PF_WR['Silt'],    'root_depth': 0.3},
        'Till':    {'root_id': 4, 'root_poros': _THETAS_MAX['Till'],  'root_fc': None, 'root_wp': None, 'root_ksat': vertical_ksat_factor * kmax['Till'],  'root_beta': 4.0, 'root_alpha': _PF_ALPHA['Till'],  'root_n': _PF_N['Till'],  'root_wr': _PF_WR['Till'],  'root_depth': 0.3},
        'Sand':    {'root_id': 5, 'root_poros': _THETAS_MAX['Sand'],  'root_fc': None, 'root_wp': None, 'root_ksat': vertical_ksat_factor * kmax['Sand'],  'root_beta': 4.0, 'root_alpha': _PF_ALPHA['Sand'],  'root_n': _PF_N['Sand'],  'root_wr': _PF_WR['Sand'],  'root_depth': 0.3},
    }

    deep_properties = {
        'Bedrock': {'deep_id': 1, 'deep_z': [-10.0], 'pF': {'ThetaS': [_THETAS_MAX['Bedrock']], 'ThetaR': [_PF_WR['Bedrock']], 'alpha': [_PF_ALPHA['Bedrock']], 'n': [_PF_N['Bedrock']]}, 'deep_ksat': None},
        'Peat':    {'deep_id': 2, 'deep_z': [-10.0], 'pF': {'ThetaS': [_THETAS_MAX['Peat']],    'ThetaR': [_PF_WR['Peat']],    'alpha': [_PF_ALPHA['Peat']],    'n': [_PF_N['Peat']]},    'deep_ksat': None},
        'Silt':    {'deep_id': 3, 'deep_z': [-10.0], 'pF': {'ThetaS': [_THETAS_MAX['Silt']],    'ThetaR': [_PF_WR['Silt']],    'alpha': [_PF_ALPHA['Silt']],    'n': [_PF_N['Silt']]},    'deep_ksat': None},
        'Till':    {'deep_id': 4, 'deep_z': [-10.0], 'pF': {'ThetaS': [_THETAS_MAX['Till']],  'ThetaR': [_PF_WR['Till']],  'alpha': [_PF_ALPHA['Till']],  'n': [_PF_N['Till']]},  'deep_ksat': None},
        'Sand':    {'deep_id': 5, 'deep_z': [-10.0], 'pF': {'ThetaS': [_THETAS_MAX['Sand']],  'ThetaR': [_PF_WR['Sand']],  'alpha': [_PF_ALPHA['Sand']],  'n': [_PF_N['Sand']]},  'deep_ksat': None},
    }

    # ── Build exp_params ───────────────────────────────────────────────────────
    exp_params = {}
    for soil_type, props in deep_properties.items():
        exp_params[soil_type] = {
            'deep_id':          props['deep_id'],
            'Kmax':             kmax[soil_type],
            'f':                f[soil_type],
            'Kmin':             kmin[soil_type],
            'max_depth':        max_depth[soil_type],
            'ThetaS_max':       _THETAS_MAX[soil_type],
            'ThetaS_min':       _THETAS_MIN[soil_type],
            'f_theta':          _F_THETA[soil_type],
            'const_surf_depth': const_surf_depth[soil_type],
        }

    # ── Plot Ksat(z) profiles ──────────────────────────────────────────────────
    if plot:
        fig_dir = Path(fig_dir) if fig_dir else project_root / 'figs'
        fig_dir.mkdir(parents=True, exist_ok=True)
        out_path = fig_dir / 'deep_soil_exponential_profiles.png'
        plot_ksat_profiles(exp_params, out_path=out_path)
        if verbose:
            print(f'Ksat profile plot → {out_path}')

    # ── Discretize deep_properties_exp ────────────────────────────────────────
    deep_properties_exp = {}
    for soil_type, props in deep_properties.items():
        params = exp_params[soil_type]
        depth_positive = discretized_depths(params['max_depth'])
        deep_z = np.round(-depth_positive, 3).tolist()
        n_layers = len(deep_z)

        # Ksat constant at Kmax down to const_surf_depth, then exponential decay shifted below it
        below_const = np.maximum(depth_positive - params['const_surf_depth'], 0.0)
        deep_ksat_raw = np.where(
            depth_positive <= params['const_surf_depth'],
            params['Kmax'],
            (params['Kmax'] - params['Kmin']) * np.exp(-params['f'] * below_const) + params['Kmin'],
        )
        deep_ksat = []
        for v in deep_ksat_raw.tolist():
            deep_ksat.append(round_sig(v, sig_figs=3))

        pf_base = expand_pf_layers(props.get('pF', {}), n_layers)
        if exponential_ThetaS:
            thetas_raw = (params['ThetaS_max'] - params['ThetaS_min']) * np.exp(-params['f_theta'] * depth_positive) + params['ThetaS_min']
            pf_base['ThetaS'] = [round(v, 4) for v in thetas_raw.tolist()]

        deep_properties_exp[soil_type] = dict(props)
        deep_properties_exp[soil_type]['deep_z'] = deep_z
        deep_properties_exp[soil_type]['pF'] = pf_base
        deep_properties_exp[soil_type]['deep_ksat'] = deep_ksat

    # ── Compute fc and wp ─────────────────────────────────────────────────────
    if verbose:
        print(f"{'Soil type':10s}  {'fc':>8}  {'wp':>8}")
        print('-' * 32)
    for soil_type in root_properties:
        pF = {
            'ThetaS': [root_properties[soil_type]['root_poros']],
            'ThetaR': [root_properties[soil_type]['root_wr']],
            'alpha':  [root_properties[soil_type]['root_alpha']],
            'n':      [root_properties[soil_type]['root_n']],
        }
        _, fc, wp = wrc(pF)
        root_properties[soil_type]['root_fc'] = round(fc, 4)
        root_properties[soil_type]['root_wp'] = round(wp, 4)
        org_properties[soil_type]['org_fc']   = round(fc, 4)
        org_properties[soil_type]['org_wp']   = round(wp, 4)
        if verbose:
            print(f"  {soil_type:10s}  {fc:8.4f}  {wp:8.4f}")

    # ── Write soil_params.py ──────────────────────────────────────────────────
    if write:
        org_code  = org_properties_function_text(org_properties)
        root_code = root_properties_function_text(root_properties)
        deep_code = deep_properties_function_text(deep_properties_exp)
        out_path = project_root / 'parameters_krycklan_soil.py'
        out_path.write_text('\n\n'.join([org_code, root_code, deep_code]) + '\n', encoding='utf-8')
        if verbose:
            print(f'Written → {out_path}')

    return {'org': org_properties, 'root': root_properties, 'deep': deep_properties_exp}


if __name__ == '__main__':
    create_soil_params()
