# -*- coding: utf-8 -*-
"""
PARAMETERS
@author: slauniai & khaahti & jpnousu
"""

import pathlib
import os
import time
from datetime import datetime

reso = 20
streams = 'channels'
#streams = '5haStreams'
results_run = os.getenv('SPAFHY_RESULTS_RUN', 'run_9_2026')


def _resolve_results_folder():
    results_root = pathlib.Path(os.getenv('SPAFHY_RESULTS_FOLDER', '/Users/jpnousu/Data/SpaFHy_RESULTS'))
    return results_root / 'krycklan' / results_run / f'{streams}_{reso}m'


def _constant_properties(properties):
    def _properties():
        return properties

    return _properties

def _configure_soil_overrides(soil_params=None):
    """Optionally replace the default soil property functions in this module."""
    global org_properties, root_properties, deep_properties

    if soil_params is not None:
        org_properties = _constant_properties(soil_params['org'])
        root_properties = _constant_properties(soil_params['root'])
        deep_properties = _constant_properties(soil_params['deep'])
        return

    try:
        import parameters_krycklan_soil as _sp

        if hasattr(_sp, 'org_properties'):
            org_properties = _sp.org_properties
        if hasattr(_sp, 'root_properties'):
            root_properties = _sp.root_properties
        if hasattr(_sp, 'deep_properties'):
            deep_properties = _sp.deep_properties
    except ImportError:
        pass


def parameters(folder='', soil_params=None):
    _configure_soil_overrides(soil_params)

    pgen = {'description': 'final_run',  # description written in result file
            'simtype': '2D', # 1D, TOP, 2D,
            'start_date': '2014-10-01', # 2013-01-01 full run
            'end_date': '2015-10-01', # 2018-12-31 full run
            #'spinup_file': r'F:\SpaFHy_2D_2021/testcase_input_202304051037_spinup.nc',
            'spinup_end': '2015-04-01',  # 2014-09-01 full run
            'dt': 86400.0,
            'spatial_cpy': True,  # if False uses parameters from cpy['state']
            # else needs cf.dat, hc.dat, LAI_decid.dat, LAI_spruce.dat, LAI_pine.dat, (cmask.dat)
            'spatial_soil': True,  # if False uses soil_id, stream_depth from psp
            'spatial_deep': True,
            'org_drain': True, # organic layer drainage True/False
            'overland_flow': True, # use BucketOLFGrid (overland flow routing) True/False
            'explicit_rootzone': False, # if False, BucketGrid/BucketOLFGrid simulate only the organic
                                        # layer; transpiration and infiltration are handled by
                                        # SoilGrid_2Dflow instead. Requires simtype == '2D'.
            'ditch_boundary': 'Cauchy',  # ditch boundary condition: 'Cauchy' (flux) or 'Dirichlet' (constant head)
            'transmissivity_mean': 'harmonic', # interface transmissivity averaging: 'harmonic' or 'geometric'
            # SoilGrid_2Dflow adaptive sub-stepping / Picard iteration tuning (defaults
            # match soilprofile2D.py; tune per-catchment/resolution if convergence warnings appear)
            'max_substep_halvings': 6,  # max sub-step halvings before accepting a non-converged result
            'min_substep_dt': 0.01171875,  # absolute floor on sub-step size [d] (16.875 min)
            'early_exit_iter': 20,  # Picard iterations before bailing to try a smaller sub-step
            'maxiter': 100,  # Picard iterations allowed once no smaller sub-step is possible
            'picard_relax': 0.7,  # <1 damps oscillatory non-convergence (e.g. 0.6-0.8); 1.0 = no damping
            'topmodel': True,
            # else needs soil_id.dat, stream_depth.dat
            'spatial_forcing': False,  # if False uses forcing from forcing file with pgen['forcing_id'] and cpy['loc']
            'spatial_radiation_file': None, # if spatial radiation file, otherwise None
            # else needs Ncoord.dat, Ecoord.dat, forcing_id.dat
            'gis_folder': str(pathlib.Path(folder+f'/gis/{reso}m')),
            'forcing_file': str(pathlib.Path(folder+r'/forcing/FORCING.csv')),
            'forcing_id': 0,  # used if spatial_forcing == False
            'ncf_file': datetime.now().strftime('%Y%m%d%H%M%S%f') + r'.nc',  # timestamp to result file name to avoid saving problem when running repeatedly
            'cmask' : 'catchment_mask.asc',
            'mask': None, # 'cmask/streams', 'cmask', 'streams', None
            #'results_folder': r'/scratch/project_2000908/nousu/SpaFHy_RESULTS',
            'results_folder': str(_resolve_results_folder()),
            'save_interval': 366, # interval for writing results to file (decreases need for memory during computation)
            'variables':[ # list of output variables (rows can be commented away if not all variables are of interest)
                    #['parameters_lai_conif', 'leaf area index of conifers [m2 m-2]'],
                    ['parameters_LAI_conif', 'leaf area index of conifers [m2 m-2]'],
                    ['parameters_LAI_decid', 'leaf area index of conifers [m2 m-2]'],
                    #['parameters_lai_shrub', 'leaf area index of shrubs [m2 m-2]'],
                    #['parameters_lai_grass', 'leaf area index of grass [m2 m-2]'],
                    ['parameters_canopy_height', 'canopy height [m]'],
                    ['parameters_canopy_fraction', 'canopy closure [-]'],
                    ['parameters_org_id', 'soil class index'],
                    ['parameters_root_id', 'soil class index'],
                    ['parameters_deep_id', 'soil class index'],
                    ['parameters_elevation', 'elevation from dem [m]'],
                    ['parameters_deep_z', 'deep soil layer thickness [m]'],
                    #['parameters_lat', 'latitude [deg]'],
                    #['parameters_lon', 'longitude [deg]'],
                    ['parameters_streams', 'streams'],
                    ['parameters_stream_length', 'total stream length'],
                    ['parameters_stream_distance', 'average distance to stream'],
                    ['parameters_stream_width', 'average stream width'],
                    ['parameters_lakes', 'lakes'],
                    ['parameters_cmask', 'cmask'],
                    #['parameters_twi', 'twi'],
                    #['parameters_slope', 'slope'],
                    ['forcing_air_temperature', 'air temperature [degC]'],
                    ['forcing_relative_humidity', 'relative humidity [%]'],
                    ['forcing_precipitation', 'precipitation [mm d-1]'],
                    ['forcing_vapor_pressure_deficit', 'vapor pressure deficit [kPa]'],
                    ['forcing_global_radiation', 'global radiation [Wm-2]'],
                    ['forcing_wind_speed','wind speed [m s-1]'],
                    ['forcing_wind_direction','wind direction [degrees]'],                
                    ['bucket_pond_storage', 'pond storage [m]'],
                    ['bucket_moisture_top', 'volumetric water content of moss layer [m3 m-3]'],
                    ['bucket_moisture_root', 'volumetric water content of rootzone [m3 m-3]'],
                    ['bucket_lateral_netflow', 'lateral overland netflow [mm d-1]'],
                    #['bucket_psi_root', 'soil water potential of rootzone [MPa]'],                    
                    #['bucket_potential_infiltration', 'potential infiltration [mm d-1]'],
                    ['bucket_surface_runoff', 'surface runoff [mm d-1]'],
                    ['bucket_evaporation', 'evaporation from soil surface [mm d-1]'],
                    ['bucket_drainage', 'drainage from root layer [mm d-1]'],
                    ['bucket_water_storage', 'bucket water storage (top and root) [mm d-1]'],
                    #['bucket_water_storage_top', 'bucket water storage (top) [mm d-1]'],
                    #['bucket_water_storage_root', 'bucket water storage (root) [mm d-1]'],
                    #['bucket_storage_change', 'bucket water storage change (top and root) [mm d-1]'],
                    ['bucket_water_closure', 'bucket water balance error [mm d-1]'],
                    ['bucket_return_flow', 'return flow from deepzone to bucket [mm d-1]'],
                    ['deep_water_storage', 'soil water storage (deeplayer) [m]'],
                    ['deep_ground_water_level', 'ground water level [m]'],
                    ['deep_lateral_netflow', 'subsurface lateral netflow [mm d-1]'],
                    ['deep_netflow_to_lake', 'netflow to lake [mm d-1]'],
                    ['deep_netflow_to_ditch', 'netflow to stream [mm d-1]'],
                    ['deep_moisture_deep', 'volumetric water content of deepzone [m3 m-3]'],
                    ['deep_water_closure', 'soil water balance error [mm d-1]'],                   
                    ['deep_return_flow', 'return flow from deepzone to bucket [mm d-1]'],
                    #['deep_transmissivity_W', 'transmissivity west [m2 d-1]'],
                    #['deep_transmissivity_E', 'transmissivity east [m2 d-1]'],
                    #['deep_transmissivity_N', 'transmissivity north [m2 d-1]'],
                    #['deep_transmissivity_S', 'transmissivity south [m2 d-1]'],
                    ['deep_leakage', 'leakage [mm d-1]'],
                    #['canopy_interception', 'canopy interception [mm d-1]'],
                    ['canopy_evaporation', 'evaporation from interception storage [mm d-1]'],
                    ['canopy_transpiration','transpiration [mm d-1]'],
                    #['canopy_stomatal_conductance','stomatal conductance [m s-1]'],
                    #['canopy_throughfall', 'throughfall to moss or snow [mm d-1]'],
                    ['canopy_snow_water_equivalent', 'snow water equivalent [mm]'],
                    ['canopy_water_closure', 'canopy water balance error [mm d-1]'],
                    #['canopy_phenostate', 'canopy phenological state [-]'],
                    #['canopy_leaf_area_index', 'canopy leaf area index [m2 m-2]'],
                    #['canopy_degree_day_sum', 'sum of degree days [degC]'],
                    #['canopy_fLAI', 'state of LAI'],
                    ['canopy_water_storage', 'canopy intercepted water storage [mm d-1]'],
                    #['canopy_snowfall', 'canopy snowfall'],
                    ['top_baseflow', 'topmodel baseflow [mm d-1]'],
                    ['top_water_closure', 'topmodel water balance error [mm d-1]'],
                    ['top_returnflow', 'topmodel returnflow [mm d-1]'],
                    ['top_local_returnflow', 'topmodel local returnflow [mm d-1]'],
                    ['top_drainage_in', 'topmodel inflow from drainage [mm d-1]'],
                    ['top_saturation_deficit', 'topmodel saturation deficit [m]'],
                    ['top_local_saturation_deficit', 'topmodel local saturation deficit [mm]'],
                    ['top_saturated_area', 'topmodel saturated area [-]'],
                    ['top_storage_change', 'topmodel_water_storage_change [mm d-1]']
                    ]
             }

    f=1.0

    # canopygrid
    pcpy = {
            'flow' : {  # flow field
                     'zmeas': 10.0,
                     'zground': 0.5,
                     'zo_ground': 0.01
                     },
            'interc': {  # interception
                        'wmax': 0.5, #1.5,  # storage capacity for rain (mm/LAI)
                        'wmaxsnow': 4.5,  # storage capacity for snow (mm/LAI)
                        },
            'snow': {  # degree-day snow model
                    'kmelt': 2.8934e-05,  # melt coefficient in open (mm/s)
                    'kfreeze': 5.79e-6,  # freezing coefficient (mm/s)
                    'r': 0.05  # maximum fraction of liquid in snow (-)
                    },
            'physpara': {  # canopy conductance
                        'amax': 10.0, # maximum photosynthetic rate (umolm-2(leaf)s-1) # MOVING THESE TO SPEC_PARA
                        'g1_conif': f * 2.1, # stomatal parameter, conifers
                        'g1_decid': f * 3.5, # stomatal parameter, deciduous
                        'g1_shrub': f * 3.0, # stomatal parameter, deciduous
                        'g1_grass': f * 5.0, # stomatal parameter, deciduous
                        'q50': 50.0, # light response parameter (Wm-2)
                        'kp': 0.6, # light attenuation parameter (-)
                        'rw': 0.20, # critical value for REW (-),
                        'rwmin': 0.02, # minimum relative conductance (-)
                        # soil evaporation
                        'gsoil': 1e-2 # soil surface conductance if soil is fully wet (m/s)
                        },
            'spec_para': {
                        'conif': {  'amax': 10.0, # maximum photosynthetic rate (umolm-2(leaf)s-1)
                                    'g1': 2.1, # stomatal parameter
                                    'q50': 50.0, # light response parameter (Wm-2)
                                    'lai_cycle': False,
                                     },
                        'decid': {  'amax': 10.0, # maximum photosynthetic rate (umolm-2(leaf)s-1)
                                    'g1': 3.5, # stomatal parameter
                                    'q50': 50.0, # light response parameter (Wm-2)
                                    'lai_cycle': True,
                                     },
                        'shrub': {  'amax': 10.0, # maximum photosynthetic rate (umolm-2(leaf)s-1)
                                    'g1': 3.0, # stomatal parameter
                                    'q50': 50.0, # light response parameter (Wm-2)
                                    'lai_cycle': False,
                                     },
                        'grass': {  'amax': 10.0, # maximum photosynthetic rate (umolm-2(leaf)s-1)
                                    'g1': 5.0, # stomatal parameter
                                    'q50': 50.0, # light response parameter (Wm-2)
                                    'lai_cycle': True,
                                     },
                        },
            'phenopara': {
                        # seasonal cycle of physiology: smax [degC], tau[d], xo[degC],fmin[-](residual photocapasity)
                        'smax': 18.5, # degC
                        'tau': 13.0, # days
                        'xo': -4.0, # degC
                        'fmin': 0.05, # minimum photosynthetic capacity in winter (-)
                        # deciduos phenology
                        'LAI_decid_min': 0.1, # minimum relative LAI (-)
                        'ddo': 45.0, # degree-days for bud-burst (5degC threshold)
                        'ddur': 23.0, # duration of leaf development (days)
                        'sdl': 9.0, # daylength for senescence start (h)
                        'sdur': 30.0, # duration of leaf senescence (days),
                         },
            'state': {  # spatial_cpy = False -> floats | spatial_cpy = True -> filenames in gispath
                       'LAI_conif': 'LAI_conif.asc', # conifer 1-sided LAI (m2 m-2)
                       'LAI_decid': 'LAI_decid.asc',  # maximum annual deciduous 1-sided LAI (m2 m-2)
                       'LAI_shrub': 0.1,
                       'LAI_grass': 0.2,
                       'canopy_height': 'canopy_height.asc', # canopy height (m)
                       'canopy_fraction': 'canopy_fraction.asc', # canopy closure fraction (-)
                       # initial state of canopy storage [mm] and snow water equivalent [mm]
                       'w': 0.0, # canopy storage mm
                       'swe': 0.0, # snow water equivalent mm
                       },
            'loc': {  # following coordinates used if spatial_forcing == False
                    'lat': 67.995,  # decimal degrees
                    'lon': 24.224
                    }
            }

    # soil profile (bucket)
    pbu = {
            # soil profile, following properties are used if spatial_soil = False
            # overlandflow model
            'flowacc': 'flow_accumulation_d8.asc',
            'fdir': 'flowp_d8.asc',
            'streams': f'{streams}.asc',
            'lakes': 'lake_mask.asc',
            # organic moss-humus layer
            'org_id': 'soil_grouped.asc', # uniform (float) OR path to grid in gispath (str)       
            'org_depth': 0.05, # depth of organic top layer (m)
            'org_poros': 0.448, # porosity (-)
            'org_fc': 0.33, # field capacity (-)
            'org_rw': 0.15, # critical vol. moisture content (-) for decreasing phase in Ef
            'org_ksat': 1E-04, # org zone hydraulic conductivity
            'org_beta': 6.0, # 
            'maxpond': 0.0, # max ponding depth (m)
            # rootzone layer
            'root_id': 'soil_grouped.asc', # uniform (float) OR path to grid in gispath (str)     
            'root_depth': 0.3, #0.3, # depth of rootzone layer (m)
            'root_sat': 0.6, # saturation ratio (-)
            'root_fc': 0.33, # field capacity
            'root_poros': 0.448, #  porosity
            'root_wp': 0.13, # wilting point
            'root_ksat': 1e-05, # hydraulic conductivity
            'root_beta': 4.7, #
            'root_alpha': 4.48, #
            'root_n': 1.20, # 
            'root_wr': 0.0, #
            # initial states
            'org_sat': 1.0, # organic top layer saturation ratio (-)
            'pond_storage': 0.0,  # initial pond depth at surface [m]
            }

    # soil profile (2D, deep)
    pspd = {
            # deep soil profile, following properties are used if spatial_deep = False
            'deep_id': 'soil_grouped.asc', # uniform (float) OR path to grid in gispath (str)
            'elevation': 'processed_dem.asc', # uniform (float) OR path to grid in gispath (str) 
            'streams': f'{streams}.asc',
            'stream_distance': f'{streams}_distance.asc', # average distance to stream
            'stream_length': f'{streams}_length.asc', # total stream length
            'stream_width': f'{streams}_width.asc', # average stream width
            'lakes': 'lake_mask.asc',
            #'deep_z': 'soildepth.asc',
            'deep_z': 10.0,
            'deep_poros': 0.41,
            'deep_wr': 0.05,
            'deep_alpha': 0.024,
            'deep_n': 1.2,
            'deep_ksat': 1E-05,
            # initial states
            'ground_water_level': -1.0,  # groundwater depth [m]
            'stream_depth': f'{streams}_depth.asc', #  # initial stream water level relative to ground surface (currently not dynamic) [m]
            'lake_depth': -1.0  # initial lake water level relative to ground surface (currently not dynamic) [m]
            }

    return pgen, pcpy, pbu, pspd


def ptopmodel():
    """
    parameters of topmodel submodel
    """
    ptopmodel = {
            'dem': 'processed_dem.asc',
            'flow_accumulation': 'flow_accumulation_dinf.asc',
            'slope': 'slope.asc',
            'twi': 'twi_dinf.asc',
            'dt': 86400.0, # timestep (s)
            'm': 0.025, # 0.025 calibrated by Samuli, scaling depth (m), testin 0.01
            'ko': 0.001, # transmissivity parameter (ms-1)
            'twi_cutoff': 99.5,  # cutoff of cumulative twi distribution (%)
            'so': 0.05 # initial saturation deficit (m)
           }
    return ptopmodel

def auxiliary_grids():
    """
    paths to auxiliary grids such as cmask, lakes, streams
    """
    grids = {
            'cmask':    'catchment_mask.asc',
            'streams':  f'{streams}.asc',
            'lakes':    'lake_mask.asc'
            }
    return grids

def deep_properties():
    """
    Properties of soil profiles generated from exponential conductivity parameters.
    Note z is elevation of lower boundary of layer (soil surface at 0.0).
    """
    deepp = {
        'Bedrock': {
            'deep_id': 1,
            'deep_z': [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0, -1.2, -1.4, -1.6, -1.8, -2.0, -2.5, -3.0, -3.5, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0],
            'pF': {
                'ThetaS': [0.3702, 0.3212, 0.2811, 0.2483, 0.2214, 0.1994, 0.1814, 0.1666, 0.1545, 0.1447, 0.1299, 0.1201, 0.1135, 0.109, 0.106, 0.1022, 0.1008, 0.1003, 0.1001, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                'ThetaR': [0.05] * 25,
                'alpha': [0.024] * 25,
                'n': [1.2] * 25,
            },
            'deep_ksat': [7.67E-07, 5.94E-07, 4.66E-07, 3.71E-07, 3.01E-07, 2.49E-07, 2.1E-07, 1.82E-07, 1.6E-07, 1.45E-07, 1.25E-07, 1.13E-07, 1.07E-07, 1.04E-07, 1.02E-07, 1E-07, 1E-07, 1E-07, 1E-07, 1E-07, 1E-07, 1E-07, 1E-07, 1E-07, 1E-07],
        },
        'Peat': {
            'deep_id': 2,
            'deep_z': [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0, -1.2, -1.4, -1.6, -1.8, -2.0, -2.5, -3.0, -3.5, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0],
            'pF': {
                'ThetaS': [0.8193, 0.7614, 0.714, 0.6752, 0.6435, 0.6175, 0.5962, 0.5787, 0.5645, 0.5528, 0.5354, 0.5237, 0.5159, 0.5107, 0.5071, 0.5026, 0.501, 0.5004, 0.5001, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                'ThetaR': [0.196] * 25,
                'alpha': [0.072] * 25,
                'n': [1.255] * 25,
            },
            'deep_ksat': [7.41E-05, 5.49E-05, 4.07E-05, 3.02E-05, 2.24E-05, 1.66E-05, 1.23E-05, 9.16E-06, 6.81E-06, 5.07E-06, 2.83E-06, 1.6E-06, 9.22E-07, 5.51E-07, 3.48E-07, 1.55E-07, 1.12E-07, 1.03E-07, 1.01E-07, 1E-07, 1E-07, 1E-07, 1E-07, 1E-07, 1E-07],
        },
        'Fine': {
            'deep_id': 3,
            'deep_z': [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0, -1.2, -1.4, -1.6, -1.8, -2.0, -2.5, -3.0, -3.5, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0],
            'pF': {
                'ThetaS': [0.5927, 0.5857, 0.5791, 0.5728, 0.5668, 0.5611, 0.5557, 0.5505, 0.5456, 0.541, 0.5323, 0.5245, 0.5174, 0.511, 0.5052, 0.493, 0.4835, 0.4761, 0.4703, 0.4623, 0.4575, 0.4545, 0.4527, 0.4517, 0.451],
                'ThetaR': [0.07] * 25,
                'alpha': [0.018] * 25,
                'n': [1.16] * 25,
            },
            'deep_ksat': [4.11E-06, 3.38E-06, 2.79E-06, 2.3E-06, 1.9E-06, 1.58E-06, 1.31E-06, 1.09E-06, 9.1E-07, 7.63E-07, 5.45E-07, 3.98E-07, 3E-07, 2.34E-07, 1.9E-07, 1.33E-07, 1.12E-07, 1.04E-07, 1.02E-07, 1E-07, 1E-07, 1E-07, 1E-07, 1E-07, 1E-07],
        },
        'Medium': {
            'deep_id': 4,
            'deep_z': [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0, -1.2, -1.4, -1.6, -1.8, -2.0, -2.5, -3.0, -3.5, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0],
            'pF': {
                'ThetaS': [0.3824, 0.35, 0.3222, 0.2982, 0.2775, 0.2598, 0.2445, 0.2313, 0.22, 0.2102, 0.1946, 0.1831, 0.1745, 0.1681, 0.1634, 0.1563, 0.153, 0.1514, 0.1507, 0.1501, 0.15, 0.15, 0.15, 0.15, 0.15],
                'ThetaR': [0.05] * 25,
                'alpha': [0.024] * 25,
                'n': [1.2] * 25,
            },
            'deep_ksat': [2.74E-04, 1.51E-04, 8.27E-05, 4.54E-05, 2.5E-05, 1.38E-05, 7.6E-06, 4.21E-06, 2.36E-06, 1.34E-06, 4.73E-07, 2.12E-07, 1.34E-07, 1.1E-07, 1.03E-07, 1E-07, 1E-07, 1E-07, 1E-07, 1E-07, 1E-07, 1E-07, 1E-07, 1E-07, 1E-07],
        },
        'Coarse': {
            'deep_id': 5,
            'deep_z': [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0, -1.2, -1.4, -1.6, -1.8, -2.0, -2.5, -3.0, -3.5, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0],
            'pF': {
                'ThetaS': [0.4038, 0.3892, 0.3759, 0.364, 0.3531, 0.3433, 0.3344, 0.3264, 0.3191, 0.3125, 0.3012, 0.2919, 0.2843, 0.2781, 0.273, 0.264, 0.2585, 0.2551, 0.2531, 0.2511, 0.2504, 0.2502, 0.2501, 0.25, 0.25],
                'ThetaR': [0.05] * 25,
                'alpha': [0.024] * 25,
                'n': [1.2] * 25,
            },
            'deep_ksat': [3.35E-04, 2.25E-04, 1.51E-04, 1.02E-04, 6.85E-05, 4.63E-05, 3.13E-05, 2.13E-05, 1.46E-05, 1.01E-05, 5.11E-06, 2.85E-06, 1.83E-06, 1.37E-06, 1.17E-06, 1.02E-06, 1E-06, 1E-06, 1E-06, 1E-06, 1E-06, 1E-06, 1E-06, 1E-06, 1E-06],
        },
    }
    return deepp

def root_properties():
    """
    swedish_soilmap_root
    """
    rootp = {
            'Bedrock': # 
                {
                 'root_id': 1,
                 'root_poros': 0.43,
                 'root_fc': 0.33,
                 'root_wp': 0.13,
                 'root_ksat': 1e-05,
                 'root_beta': 4.7,
                 'root_alpha': 0.024, # UNIT?
                 'root_n': 1.2,
                 'root_wr': 0.05,
                 #'root_depth': 0.10, # !! CHECK IF WORKS STRAIGHT
                 },
            'Peat':
                {
                 'root_id': 2,
                 'root_poros': 0.89,
                 'root_fc': 0.54, # Leppä et al. Spaghnum -10 kPa (-1m)
                 'root_wp': 0.22, # Leppä et al. Spaghnum -1500 kPa (-150m)
                 'root_alpha': 0.4, # kPa-1
                 'root_beta': 4.0,                 
                 'root_n': 1.46,
                 'root_wr': 0.178,
                 'root_ksat': 1e-5,
                 },
            'Clay_silt': #  C3
                {'root_id': 3,
                 'root_poros': 0.55,
                 'root_fc': 0.26,
                 'root_wp': 0.09,
                 'root_alpha': 0.448,
                 'root_beta': 4.0,                    
                 'root_n': 1.20,
                 'root_wr': 0.0,
                 'root_ksat': 1e-5,
                },
            'Moraine': # C5
                {
                 'root_id': 4,
                 'root_poros': 0.41,
                 'root_fc': 0.14,
                 'root_wp': 0.04,
                 'root_alpha': 0.38,
                 'root_beta': 4.0,                 
                 'root_n': 1.42,
                 'root_wr': 0.03,
                 'root_ksat': 1e-4,
                 },
            'Postglacial_sand': # C4
                {
                 'root_id': 5,
                 'root_poros': 0.53,
                 'root_fc': 0.22, #0.24,
                 'root_wp': 0.06, #0.08,
                 'root_alpha': 0.37,
                 'root_beta': 4.0,                 
                 'root_n': 1.24,
                 'root_wr': 0.0,
                 'root_ksat': 5e-5,
                 },
            }

    return rootp

def org_properties():
    """
    swedish_soilmap
    """
    orgp = {
        'Bedrock':{
            'org_id': 1,
            'org_depth': 0.05,
            'org_poros': 0.9,
            'org_fc': 0.3,
            'org_rw': 0.2,
            'org_ksat': 1E-03,
            'org_beta': 6.0
            },
        'Peat':{
            'org_id': 2,
            'org_depth': 0.05,
            'org_poros': 0.9,
            'org_fc': 0.65,
            'org_rw': 0.3,
            'org_ksat': 1E-03,
            'org_beta': 6.0
            },
        'Clay_silt':{
            'org_id': 3,
            'org_depth': 0.05,
            'org_poros': 0.9,
            'org_fc': 0.3,
            'org_rw': 0.2,
            'org_ksat': 1E-03,
            'org_beta': 6.0
            },
        'Moraine':{
            'org_id': 4,
            'org_depth': 0.05,
            'org_poros': 0.9,
            'org_fc': 0.3,
            'org_rw': 0.2,
            'org_ksat': 1E-03,
            'org_beta': 6.0
            },
        'Postglacial_sand':{
            'org_id': 5,
            'org_depth': 0.05,
            'org_poros': 0.9,
            'org_fc': 0.3,
            'org_rw': 0.2,
            'org_ksat': 1E-03,
            'org_beta': 6.0
            },
        }
    return orgp

