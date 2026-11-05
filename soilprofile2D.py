# -*- coding: utf-8 -*-
"""
2D lateral groundwater flow model for gridded simulation of deep soil water
storage and drainage to streams in SpaFHy.

Solves the 2D Darcy equation on a finite-difference grid using a sparse
matrix system (Crank-Nicolson by default). Hydraulic head is the primary
state variable; water storage and transmissivity are derived from
pre-computed interpolation functions of groundwater level.

Lake interiors are excluded from the solution; lake and stream boundaries
are treated as constant-head (Dirichlet) conditions.

The module also provides helper functions to pre-compute the soil column
lookup tables (gwl ↔ Wsto, Tr, C) and the van Genuchten–Mualem water
retention model.

References:
    Nousu et al. (2024). Hydrol. Earth Syst. Sci., 28, 4643-4666.
    van Genuchten (1980). Soil Sci. Soc. Am. J., 44, 892-898.

@authors: alauren, khaahti, jpnousu
"""

import numpy as np
from scipy.stats.mstats import gmean
from scipy.interpolate import interp1d
from scipy.sparse import diags, linalg
eps = np.finfo(float).eps

class SoilGrid_2Dflow(object):
    """
    Gridded 2D lateral groundwater flow model based on Annamari Lauren's SUSI2D.

    Simulates deep soil water storage and lateral drainage to streams and lakes
    on a 2D raster grid. The hydraulic head H [m] is the state variable; water
    storage, transmissivity, and differential water capacity are evaluated via
    pre-computed scipy interpolation functions of groundwater level (gwl).

    Three solver modes are available via self.implic:
        0.0 — explicit (forward Euler)
        1.0 — implicit (backward Euler)
        0.5 — Crank-Nicolson (default; most stable near impermeable bottom)
    """
    def __init__(self, spara):
        """
        Initializes the 2D groundwater flow grid.

        Args:
            spara (dict): Spatial parameter dictionary. All array values share the
                grid shape (rows, cols). Expected keys:

                Spatial grids:
                    'deep_id'            [-]    soil/peat type index (NaN outside catchment)
                    'soiltype'           [-]    soil type index used for parameter lookup
                    'elevation'          [m]    soil surface elevation above datum
                    'streams'            [m]    stream/ditch water depth; negative where
                                                stream exists, 0 or np.nan elsewhere
                    'lakes'              [m]    lake water depth; negative inside lakes, 0 or np.nan elsewhere
                    'deep_z'             [m]    depth to impermeable bottom (positive downward)
                    'ground_water_level' [m]    initial groundwater level below surface (<=0)
                    'dxy'                [m]    horizontal grid cell size (dx = dy)

                Soil hydraulic lookup functions (scipy interp1d or 2D array of interp1d):
                    'wtso_to_gwl'    gwl(Wsto)   — water storage to groundwater level
                    'gwl_to_wsto'    Wsto(gwl)   — groundwater level to water storage [m]
                    'gwl_to_Tr'      Tr(gwl)     — groundwater level to transmissivity [m2 d-1]
                    'gwl_to_C'       C(gwl)      — differential water capacity dWsto/dh [m m-1]
                    'gwl_to_rootmoist' theta(gwl) — groundwater level to root zone moisture [m3 m-3]

                When 'wtso_to_gwl' is a 2D np.ndarray of interp1d objects with the
                same shape as 'deep_id', lookup functions are applied cell-wise
                (z_from_gis=True). Otherwise they are applied per soil type
                (z_from_gis=False).
        """

        # deep soil
        # soil/peat type
        self.soiltype = spara['soiltype']
        self.deep_id = spara['deep_id']
        
        # catchment mask
        self.cmask = np.full_like(spara['deep_id'], np.nan)
        self.cmask[np.isfinite(spara['deep_id'])] = 1.0

        self.ditch_boundary = spara.get('ditch_boundary', 'Dirichlet')

        if self.ditch_boundary == 'Cauchy':
            # stream geometry needed for Cauchy flux (currently np.nan for non-streams)
            self.ditch_l = spara['stream_length']  # total stream length
            self.ditch_d = spara['stream_distance']  # average distance to stream
            self.ditch_d = np.where(self.ditch_d > 0.0, self.ditch_d, 1.0)  # avoid division by zero in S_dd

        # interpolated functions for soil column groundwater depth vs. water storage, transmissivity etc.
        self.wsto_to_gwl = spara['wtso_to_gwl']
        self.gwl_to_wsto = spara['gwl_to_wsto']
        self.gwl_to_Tr = spara['gwl_to_Tr']
        self.gwl_to_C = spara['gwl_to_C']
        self.gwl_to_rootmoist = spara['gwl_to_rootmoist']

        # initial h (= gwl) and boundaries [m]
        self.ditch_h = spara['streams']
        self.lake_h = spara['lakes']
        self.ditch_h[~np.isfinite(spara['deep_id'])] = 0.

        if self.ditch_boundary == 'Cauchy':
            # deactivate ditch cells with incomplete stream geometry (missing length or distance)
            # to avoid NaN in S_dd computation
            ditch_incomplete = (self.ditch_h < -1e-6) & (np.isnan(self.ditch_l) | np.isnan(self.ditch_d))
            self.ditch_h[ditch_incomplete] = 0.0
            if np.any(ditch_incomplete):
                print(f'  WARNING: {np.sum(ditch_incomplete)} ditch cell(s) deactivated due to incomplete stream geometry')
        self.gwl = spara['ground_water_level']
        # soil surface elevation and hydraulic head [m]
        self.ele = spara['elevation']
        self.H = self.ele + self.gwl
        # Identify lake interior and boundary cells
        lake_boundary = np.zeros_like(self.lake_h)
        self.lake_interior = np.zeros_like(self.lake_h)

        # grid
        self.rows = np.shape(self.gwl)[0]
        self.cols = np.shape(self.gwl)[1]
        self.n = self.rows * self.cols  # length of flattened array
        self.dxy = spara['dxy']  # horizontal distance between nodes dx=dy [m]
        
        # Iterate through each cell in lake_h to find if its lake boundary
        for i in range(self.rows):
            for j in range(self.cols):
                if self.lake_h[i, j] < -eps:
                    # Check if this cell is a boundary
                    is_boundary = False
                    # Check all 4 possible neighbors (west, east, north, south)
                    if (j > 0 and self.lake_h[i,j-1] == 0) or (j < self.cols - 1 and self.lake_h[i,j+1] == 0) or \
                       (i > 0 and self.lake_h[i-1,j] == 0) or (i < self.rows - 1 and self.lake_h[i+1,j] == 0):
                        is_boundary = True
                    if is_boundary:
                        lake_boundary[i,j] = 1       

        self.lake_interior[(lake_boundary != 1) & (self.lake_h < -eps)] = 1 # saving lake interior array

        if self.ditch_boundary == 'Dirichlet':
            # merge lake levels into ditch_h so both are treated as constant-head boundaries
            self.ditch_h[self.lake_h < -eps] = self.lake_h[self.lake_h < -eps]

        # nan to lake interiors (lake interiors should not be solved)
        self.soiltype[self.lake_interior == 1] = np.nan
        self.cmask[self.lake_interior == 1] = np.nan
        self.gwl[self.lake_interior == 1] = np.nan
        self.H[self.lake_interior == 1] = -999

        # lower boundaries
        #print('spara[deep_z]', spara['deep_z'])
        self.deep_z = spara['deep_z']*-1

        # physical floor for H (gwl cannot go below the impermeable bottom,
        # self.deep_z); precomputed once since self.ele/self.deep_z are static.
        # deep_z is NaN outside the catchment mask like deep_id, so the floor
        # there must NOT be NaN (np.maximum(x, nan) == nan, which would corrupt
        # the finite -999 outside-catchment sentinel set below) - such cells
        # get -inf instead, i.e. no effective floor, leaving them untouched.
        _h_floor_2d = self.ele + self.deep_z
        self._H_floor_2d = np.where(np.isfinite(_h_floor_2d), _h_floor_2d, -np.inf)
        self._H_floor = np.ravel(self._H_floor_2d)

        # replace nans (values outside catchment area)
        self.H[np.isnan(self.H)] = -999

        # water storage [m]
        self.Wsto_deep_max = np.full_like(self.gwl, 0.0)  # storage of fully saturated profile
        self.Wsto_deep = np.full_like(self.gwl, 0.0)  

        # deep moisture [m3 m-3]
        self.deepmoist = np.full_like(self.gwl, 0.0)

        # self.z_from_gis == True OR False
        # determines whether the deep_z and thus interpolation functions are made cell-wise (True) or soiltype-wise (False)
        # if-elif statements later in the code made accordingly
        self.z_from_gis = (
            isinstance(spara['wtso_to_gwl'], np.ndarray) and
            spara['deep_id'].shape == spara['wtso_to_gwl'].shape
            )
        
        # initial water storages according to gwl
        if not self.z_from_gis: # soiltype-wise calculation
            for key, value in self.gwl_to_wsto.items():
                self.Wsto_deep_max[self.soiltype == key] = value(0.0)
                self.Wsto_deep[self.soiltype == key] = value(self.gwl[self.soiltype == key]) # storage corresponding to h
            for key, value in self.gwl_to_rootmoist.items():
                self.deepmoist[self.soiltype == key] = value(self.gwl[self.soiltype == key])
        elif self.z_from_gis: # cell-wise calculation
            for i in range(self.gwl_to_wsto.shape[0]):
                for j in range(self.gwl_to_wsto.shape[1]):
                    if np.isfinite(self.cmask[i,j]): 
                        self.Wsto_deep_max[i,j] = self.gwl_to_wsto[i,j](0.0) # max storage with gwl = 0
                        self.Wsto_deep[i,j] = self.gwl_to_wsto[i,j](self.gwl[i,j]) # storage corresponding to h
                        self.deepmoist[i,j] = self.gwl_to_rootmoist[i,j](self.gwl[i,j])
            
        self.deepmoist[np.isnan(self.gwl)] = np.nan

        # reference groundwater levels for the Koivusalo et al. (2008) Rew formulation,
        # used by spafhy.py only when BucketGrid/BucketOLFGrid has no explicit root zone
        # (mirrors SpaFHy_Peat/soilprofile.py)
        self.rew_gwl_fc0 = spara.get('rew_gwl_fc0', -0.8)
        self.rew_gwl_fc1 = spara.get('rew_gwl_fc1', -1.3)
        self.rew_gwl_wp  = spara.get('rew_gwl_wp', -150.1)
        self.root_fc0 = self._eval_rootmoist_at_gwl(self.rew_gwl_fc0)
        self.root_fc1 = self._eval_rootmoist_at_gwl(self.rew_gwl_fc1)
        self.root_wp  = self._eval_rootmoist_at_gwl(self.rew_gwl_wp)
        self.Rew = np.full_like(self.gwl, 1.0)

        # air volume and returnflow
        self.airv_deep = np.maximum(0.0, self.Wsto_deep_max - self.Wsto_deep)
        self.qr = np.full_like(self.gwl, 0.0)

        # parameters for 2D solution
        # parameters for solving
        # 0.5 seems to work better when gwl is close to impermeable bottom
        # (probably because transmissivity does not switch between 0. and > 0 as much)
        self.implic = 0.5  # solving method: 0-forward Euler, 1-backward Euler, 0.5-Crank-Nicolson
        # interface transmissivity averaging: 'harmonic' (default, more restrictive,
        # dominated by lower-conductivity neighbour) or 'geometric' (earlier model version)
        self.transmissivity_mean = spara.get('transmissivity_mean', 'harmonic')

        # create arrays needed in computation only once
        # previous time step neighboring hydraylic head H (West, East, North, South)
        self.HW = np.zeros((self.rows,self.cols))
        self.HE = np.zeros((self.rows,self.cols))
        self.HN = np.zeros((self.rows,self.cols))
        self.HS = np.zeros((self.rows,self.cols))
        # previous time step transmissivities (West, East, North, South)
        self.TrW0 = np.zeros((self.rows,self.cols))
        self.TrE0 = np.zeros((self.rows,self.cols))
        self.TrN0 = np.zeros((self.rows,self.cols))
        self.TrS0 = np.zeros((self.rows,self.cols))
        # current time step transmissivities (West, East, North, South)
        self.TrW1 = np.zeros((self.rows,self.cols))
        self.TrE1 = np.zeros((self.rows,self.cols))
        self.TrN1 = np.zeros((self.rows,self.cols))
        self.TrS1 = np.zeros((self.rows,self.cols))
        # computation matrix
        # self.A = np.zeros((self.n,self.n))

        self.CC = np.ones((self.rows,self.cols))
        self.Tr0 = np.zeros((self.rows,self.cols))
        self.Tr1 = np.zeros((self.rows,self.cols))
        self.Wtso1_deep = np.zeros((self.rows,self.cols))
        self.tmstep = 0
        self.spinup_steps = spara.get('spinup_steps', 0)
        self.conv99 = 99
        # adaptive sub-stepping: if a sub-step fails to converge, retry it as two
        # half-length sub-steps, up to this many halvings
        self.max_substep_halvings = spara.get('max_substep_halvings', 6)
        # absolute floor on sub-step size [d]: halving stops here regardless of
        # max_substep_halvings, so consecutive hard timesteps (each starting from
        # the previous day's already-shrunk trial size) can't ratchet down forever
        self.min_substep_dt = spara.get('min_substep_dt', 0.01171875)  # 16.875 min
        # bail out of the solution loop and let _adaptive_solve halve dt instead
        # of going until maxiter - but only while halving is still possible; once
        # _adaptive_solve can no longer halve (depth/floor limit reached), the full
        # maxiter budget is used instead since there's no cheaper alternative left
        self.early_exit_iter = spara.get('early_exit_iter', 10)
        self.maxiter = spara.get('maxiter', 50)
        # Picard update damping: Htmp1 = Htmp + picard_relax*(Htmp1_solved - Htmp).
        # 1.0 (default) is undamped, matching previous behaviour. The per-iteration
        # |dH|<=0.5 m clamp above only bounds a single iteration's worst-case jump -
        # it does not stop a genuine oscillating fixed point (head swinging back and
        # forth by a few cm, well under that clamp, every iteration without ever
        # settling - seen for deep_id soil types with a steep Tr(gwl) transition
        # near the water table). Sub-stepping down to min_substep_dt does not fix
        # this either, since it is not a temporal-stiffness problem. Values <1
        # (e.g. 0.6-0.8) damp such oscillations.
        self.picard_relax = spara.get('picard_relax', 1.0)
        #self.totit = 0

    def _eval_rootmoist_at_gwl(self, gwl_value):
        """
        Evaluates gwl_to_rootmoist at a fixed reference groundwater level [m],
        soiltype-wise or cell-wise depending on self.z_from_gis. Used to derive
        root_fc0/root_fc1/root_wp reference moistures for the Koivusalo et al.
        (2008) Rew formulation.
        """
        out = np.full_like(self.gwl, np.nan)
        if not self.z_from_gis:
            for key, value in self.gwl_to_rootmoist.items():
                out[self.soiltype == key] = value(gwl_value)
        else:
            for i in range(self.gwl_to_rootmoist.shape[0]):
                for j in range(self.gwl_to_rootmoist.shape[1]):
                    if np.isfinite(self.cmask[i, j]):
                        out[i, j] = self.gwl_to_rootmoist[i, j](gwl_value)
        return out

    def rolling_window(self, a, window):
        """
        Returns a strided view of array a with a sliding window along the last axis.
        Used to compute geometric/harmonic-mean transmissivities at cell interfaces.

        Args:
            a      (array): 2D input array.
            window   (int): Window size (typically 2 for pairwise averaging).

        Returns:
            view (array): Shape (..., N - window + 1, window), no data copied.
        """

        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    def _interface_transmissivity(self, Tr):
        """
        Computes transmissivity at cell interfaces (E-W and N-S) from the two
        neighbouring cells' transmissivities, using either the harmonic mean
        (self.transmissivity_mean == 'harmonic', default; more restrictive,
        dominated by the lower-conductivity neighbour) or the geometric mean
        (self.transmissivity_mean == 'geometric'; the earlier model version's
        behaviour, allows relatively more flow).
        """
        if self.transmissivity_mean == 'geometric':
            TrTmpEW = gmean(self.rolling_window(Tr, 2), -1)
            TrTmpNS = np.transpose(gmean(self.rolling_window(np.transpose(Tr), 2), -1))
        else:
            w = self.rolling_window(Tr, 2)
            d = np.where(w[...,0]+w[...,1] > 0, w[...,0]+w[...,1], 1.0)  # safe denominator
            TrTmpEW = np.where(w[...,0]+w[...,1] > 0, 2*w[...,0]*w[...,1] / d, 0.0)
            w = self.rolling_window(np.transpose(Tr), 2)
            d = np.where(w[...,0]+w[...,1] > 0, w[...,0]+w[...,1], 1.0)  # safe denominator
            TrTmpNS = np.transpose(np.where(w[...,0]+w[...,1] > 0, 2*w[...,0]*w[...,1] / d, 0.0))
        return TrTmpEW, TrTmpNS

    def run_timestep(self, dt=1.0, RR=0.0, TR=0.0):
        """
        Advances the 2D groundwater flow model by one timestep by going
        through it in adaptive sub-steps: if the iteration in
        _solve_step fails to converge within early_exit_iter iterations (bailing
        out before the full maxiter number), the sub-step is retried as two
        half-length sub-steps (recursively, up to max_substep_halvings), which
        keeps the solve stable when transmissivity is very high
        (e.g. shallow water table over high-Ksat near-surface soil).

        The sub-step size is remembered across calls: each new sub-step starts
        at twice the size of the last one that worked (capped at dt), so once a
        stable smaller step size is found it is reused (and cautiously grown)
        on subsequent timesteps.

        Args:
            dt  (float): Timestep duration [days]. Default 1.0 (daily).
            RR  (array): Drainage input from BucketGrid to the saturated zone [m].
            TR  (array): Transpiration sink taken directly from the soil water
                storage [m]; used when BucketGrid/BucketOLFGrid is run without an
                explicit root zone (pgen['explicit_rootzone'] = False). Capped so
                storage cannot be drawn below zero.

        Returns:
            dict with keys (see _solve_step for full list): rate-type keys
            (lateral_netflow, netflow_to_ditch, netflow_to_lake, water_closure)
            are time-weighted averages over the sub-steps; depth-type
            keys (return_flow, transpiration) are summed; all other keys
            reflect the state at the end of the full timestep.
        """
        self.tmstep += 1

        # convergence criteria: looser during spin-up, tighter afterwards
        crit = 1e-2 if self.tmstep <= self.spinup_steps else 1e-3
        # implicit solution for spinup, crank-nicholson afterwards
        self.implic = 1.0 if self.tmstep <= self.spinup_steps else 0.5

        # start at the sub-step size that worked last timestep (doubled), capped at dt,
        # instead of always retrying (and re-failing) the full dt from scratch
        substep_dt = min(dt, getattr(self, '_trial_dt', dt))

        combined = None
        t_elapsed = 0.0
        overall_min_dt = dt  # smallest sub-step used anywhere this timestep, for logging only
        while dt - t_elapsed > 1e-9:
            this_substep = min(substep_dt, dt - t_elapsed)
            substep_RR = RR * (this_substep / dt)
            substep_TR = TR * (this_substep / dt)

            self._min_substep_dt = this_substep  # set by _adaptive_solve to the size that actually worked
            substep_result = self._adaptive_solve(this_substep, substep_RR, substep_TR, crit)

            overall_min_dt = min(overall_min_dt, self._min_substep_dt)
            w_prev = t_elapsed / (t_elapsed + this_substep)
            w_new = this_substep / (t_elapsed + this_substep)
            combined = substep_result if combined is None else self._merge_weighted(combined, substep_result, w_prev, w_new)
            t_elapsed += this_substep

            # grow after an easy sub-step, shrink to whatever size actually worked after a hard one
            substep_dt = min(dt, 2.0 * self._min_substep_dt)

        self._trial_dt = substep_dt  # seeds the first sub-step of the next timestep

        # self.qr is set inside _solve_step, once per sub-step, so after a
        # multi-sub-step day it only holds the LAST sub-step's return flow -
        # spafhy.py reads this raw attribute (not the returned dict) as QR to
        # feed BucketGrid/BucketOLFGrid's retflow next timestep, so resync it
        # to the correctly-summed full-timestep total to avoid silently
        # dropping the earlier sub-steps' return flow from the water balance
        self.qr = combined['return_flow'] * 1e-3

        if overall_min_dt < dt:
            print(f'  Timestep {self.tmstep}: dt={dt:.5f} d required sub-stepping down to '
                  f'{overall_min_dt:.5f} d')
        return combined

    # rate-type outputs [mm d-1]: time-weighted average across sub-steps
    _RATE_RESULT_KEYS = ('lateral_netflow', 'netflow_to_ditch', 'netflow_to_lake', 'water_closure')
    # depth-type outputs [mm]: accumulated depth over the (sub-)period, summed across sub-steps
    _DEPTH_RESULT_KEYS = ('return_flow', 'transpiration')

    def _snapshot_state(self):
        """Saves the primary prognostic state (H, gwl, Wsto_deep) before a solve attempt."""
        return (self.H.copy(), self.gwl.copy(), self.Wsto_deep.copy())

    def _restore_state(self, snapshot):
        """Restores primary prognostic state saved by _snapshot_state (discards a failed attempt)."""
        self.H, self.gwl, self.Wsto_deep = (a.copy() for a in snapshot)

    def _merge_weighted(self, earlier, later, w_earlier, w_later):
        """
        Combines results from two chronologically sequential (sub-)periods.
        Rate-type keys are weighted-averaged (w_earlier + w_later should sum to
        1); depth-type keys are summed; everything else (state) is taken from
        the later period.
        """
        combined = dict(later)
        for key in self._RATE_RESULT_KEYS:
            if key in earlier:
                combined[key] = earlier[key] * w_earlier + later[key] * w_later
        for key in self._DEPTH_RESULT_KEYS:
            if key in earlier:
                combined[key] = earlier[key] + later[key]
        return combined

    def _adaptive_solve(self, dt_sub, RR_sub, TR_sub, crit, depth=0):
        """
        Solves one sub-interval of length dt_sub; on non-convergence, rolls back
        state and retries as two half-length sub-steps (recursively).

        While further halving is still possible, _solve_step bails out after
        only self.early_exit_iter Picard iterations - cheaper to retry as a
        smaller, less stiff sub-step than to grind through more iterations.
        Once halving is no longer possible (depth or floor limit reached),
        there's no cheaper alternative left, so _solve_step is allowed to
        iterate all the way to self.maxiter instead of bailing out early.
        """
        snapshot = self._snapshot_state()
        depth_exhausted = depth >= self.max_substep_halvings
        floor_reached = dt_sub <= self.min_substep_dt
        at_floor = depth_exhausted or floor_reached
        iter_budget = self.maxiter if at_floor else self.early_exit_iter
        converged, results = self._solve_step(dt_sub, RR_sub, TR_sub, crit, iter_budget)

        if converged or at_floor:
            if not converged:
                reason = (f'reached the {self.min_substep_dt:.5f} d minimum sub-step'
                          if floor_reached else f'after {self.max_substep_halvings} halvings')
                print(f'  WARNING: sub-step dt={dt_sub:.5f} d (depth {depth}) did not converge '
                      f'{reason} within {iter_budget} iterations; accepting last iterate')
                n_cells, deep_id_counts = self._last_non_conv_summary
                print(f'  Non-converged cells: {n_cells}')
                if deep_id_counts is not None:
                    ids, counts = deep_id_counts
                    for did, cnt in zip(ids, counts):
                        print(f'    deep_id={int(did)}: {cnt} cells')
            self._min_substep_dt = min(self._min_substep_dt, dt_sub)
            return results

        # retry the same interval as two half-length sub-steps, in chronological order
        self._restore_state(snapshot)
        half_dt, half_RR, half_TR = dt_sub / 2.0, RR_sub / 2.0, TR_sub / 2.0
        first = self._adaptive_solve(half_dt, half_RR, half_TR, crit, depth + 1)
        second = self._adaptive_solve(half_dt, half_RR, half_TR, crit, depth + 1)
        return self._merge_weighted(first, second, 0.5, 0.5)

    def _solve_step(self, dt, RR, TR, crit, max_iter):
        """
        Attempts to advance the 2D groundwater flow model by dt (single Picard/
        Crank-Nicolson attempt, no sub-stepping). Transmissivity is updated
        inside the iteration loop. Stream/lake cells are treated as
        constant-head boundaries when the neighbouring water table is above
        the ditch/lake water level.

        Note:
            dt is in days. Transmissivity lookup tables are pre-converted to
            [m2 d-1] in gwl_Wsto / gwl_Wsto_vectorized.

        Args:
            dt       (float): Timestep (or sub-step) duration [days].
            RR       (array): Drainage input from BucketGrid to the saturated zone [m].
            TR       (array): Transpiration sink taken directly from the soil water
                storage [m]. Capped so storage cannot be drawn below zero.
            crit     (float): Convergence criterion [m] for the Picard iteration.
            max_iter   (int): Iteration count at which to give up (converged=False)
                if not yet converged; set by _adaptive_solve to self.early_exit_iter
                while halving is still possible, self.maxiter once it isn't. The
                loop itself never runs past self.maxiter regardless of max_iter.

        Returns:
            (converged, results):
                converged (bool): whether conv1 < crit was reached before maxiter
                    or early_exit_iter, whichever comes first.
                results (dict) with keys:
                    'ground_water_level'  [m]:      updated groundwater level below surface
                    'lateral_netflow'     [mm d-1]: net lateral flow (positive = outflow)
                    'netflow_to_ditch'    [mm d-1]: net flow into streams/ditches;
                        includes lake flow when ditch_boundary is Dirichlet
                    'netflow_to_lake'     [mm d-1]: net flow into lakes in Cauchy mode
                    'water_closure'       [mm d-1]: mass balance error (should be ~0)
                    'water_storage'       [mm]:     deep soil water storage
                    'return_flow'         [mm]:     return flow to BucketGrid (when gwl > 0)
                    'transpiration'       [mm]:     transpiration actually extracted (after capping)
                    'transpiration_limitation' [-]: relative extractable water (REW), Koivusalo et al. (2008)
                    'transmissivity'      [m2 d-1]: mean transmissivity of the grid
        """

        
        #***********REMIND: map of array*******************
        #2D array: indices i row, j col
        #Flattened array: n from 0 to rows*cols: n=i*cols+j
        #West element: n=i*cols-1
        #East element: n=i*cols+1
        #North element: n=i*cols+j-cols
        #South element: n=i*cols-j+cols

        # transpiration cannot draw storage below zero (Rew already limits demand upstream in CanopyGrid)
        TR = np.minimum(TR, np.maximum(self.Wsto_deep, 0.0))
        self.tr_deep = TR

        # physical floor: gwl cannot go below the profile's impermeable bottom
        # (self.deep_z, a negative depth) - applied here (before self.H is used
        # anywhere below) so a state that already drifted past it on a previous
        # timestep can't keep feeding the interp1d extrapolation region that
        # drives a self-sustaining Picard oscillation
        self.H = np.maximum(self.H, self._H_floor_2d)

        # for computing mass balance later, RR: drainage from bucketgrid; TR: transpiration sink
        S = RR - TR
        S[np.isnan(S)] = 0.0

        state0 = self.Wsto_deep + S # [m]

        # Head in four neighbouring cells
        self.HW[:,1:] = self.H[:,:-1]
        self.HE[:,:-1] = self.H[:,1:]
        self.HN[1:,:] = self.H[:-1,:]
        self.HS[:-1,:] = self.H[1:,:]
        
        # ravel 2D arrays
        HW = np.ravel(self.HW)
        HE = np.ravel(self.HE)
        HN = np.ravel(self.HN)
        HS = np.ravel(self.HS)
        H = np.ravel(self.H)
        Wsto_deep = np.ravel(self.Wsto_deep)
        ditch_h = np.ravel(self.ditch_h)
        lake_h = np.ravel(self.lake_h)
        lake_interior = np.ravel(self.lake_interior)
        ele = np.ravel(self.ele)

        if self.ditch_boundary == 'Cauchy':
            ditch_l = np.ravel(self.ditch_l)
            ditch_d = np.ravel(self.ditch_d)
            # Dupuit-Forchheimer for all drainage cells; C_dd updated each iteration inside loop (head-dependent)
            C_dd = np.zeros_like(H)
        else:
            C_dd = np.zeros_like(H)
        
        # Boundary condition cells: lakes only for Cauchy; ditches+lakes for Dirichlet
        if self.ditch_boundary == 'Cauchy':
            bc_h = lake_h          # flat array
            bc_h_2d = self.lake_h  # 2D array
        else:
            bc_h = ditch_h
            bc_h_2d = self.ditch_h

        # calculate mean H of neighboring non-BC nodes to determine whether BC is active
        # done outside iteration loop to avoid boundary switching during iteration
        # valid neighbors must be inside the catchment (finite cmask) and not carry
        # the outside-domain value H=-999
        cmask_flat = np.ravel(self.cmask)
        H_neighbours = bc_h.copy()
        for k in np.where(bc_h < -eps)[0]:
            H_ave = 0
            n_neigh = 0
            if k%self.cols != 0 and bc_h[k-1] > -eps and np.isfinite(cmask_flat[k-1]) and H[k-1] != -999:
                    H_ave += H[k-1]
                    n_neigh += 1
            if (k+1)%self.cols != 0 and bc_h[k+1] > -eps and np.isfinite(cmask_flat[k+1]) and H[k+1] != -999:
                    H_ave += H[k+1]
                    n_neigh += 1
            if k-self.cols >= 0 and bc_h[k-self.cols] > -eps and np.isfinite(cmask_flat[k-self.cols]) and H[k-self.cols] != -999:
                    H_ave += H[k-self.cols]
                    n_neigh += 1
            if k+self.cols < self.n and bc_h[k+self.cols] > -eps and np.isfinite(cmask_flat[k+self.cols]) and H[k+self.cols] != -999:
                    H_ave += H[k+self.cols]
                    n_neigh += 1
            if n_neigh > 0:
                H_neighbours[k] = H_ave / n_neigh
            else:
                H_neighbours[k] = ele[k] + bc_h[k] + eps

        # lake interiors do not have neighbours
        for k in np.where(lake_interior == 1)[0]:
            H_neighbours[k] = ele[k] + lake_h[k] + eps

        H_neighbours_2d = np.reshape(H_neighbours,(self.rows,self.cols))

        # Transmissivity: under Dirichlet, ditch+lake cells share bc_h_2d and use
        # mean H of active neighbours (no separate mean-of-Tr mechanism exists
        # there, unlike Cauchy's ditch/lake treatment below), instead of the
        # (possibly deep) BC level which would restrict transmissivity too much.
        # Under Cauchy, bc_h_2d is lake_h only - lakes get the mean-of-neighbours'
        # own Tr treatment below instead, since substituting a neighbour's HEAD
        # here could land the lake's Tr(gwl) lookup in a neighbour-specific
        # extreme near-surface Ksat zone that has nothing to do with the lake.
        if self.ditch_boundary == 'Dirichlet':
            H_for_Tr = np.where((bc_h_2d < -eps) & (H_neighbours_2d > self.ele + bc_h_2d),
                                H_neighbours_2d, self.H)
        else:
            H_for_Tr = self.H

        # transmissivities based on gwl
        if not self.z_from_gis:
            for key, value in self.gwl_to_Tr.items():
                self.Tr0[self.soiltype == key] = value(H_for_Tr[self.soiltype == key] - self.ele[self.soiltype == key])
        elif self.z_from_gis:
            for i in range(self.gwl_to_Tr.shape[0]):
                for j in range(self.gwl_to_Tr.shape[1]):
                    if np.isfinite(self.cmask[i,j]):
                        self.Tr0[i,j] = self.gwl_to_Tr[i,j](H_for_Tr[i,j] - self.ele[i,j])

        # For Cauchy lake cells: set Tr to mean of aquifer neighbours (run before
        # the ditch block below, so a ditch cell adjacent to a lake sees the
        # lake's already-corrected Tr rather than a stale/extreme value)
        if self.ditch_boundary == 'Cauchy':
            for i in range(self.rows):
                for j in range(self.cols):
                    if self.lake_h[i,j] < -eps and np.isfinite(self.cmask[i,j]):
                        nbr = []
                        if i > 0 and self.lake_h[i-1,j] > -eps and np.isfinite(self.cmask[i-1,j]):
                            nbr.append(self.Tr0[i-1,j])
                        if i < self.rows-1 and self.lake_h[i+1,j] > -eps and np.isfinite(self.cmask[i+1,j]):
                            nbr.append(self.Tr0[i+1,j])
                        if j > 0 and self.lake_h[i,j-1] > -eps and np.isfinite(self.cmask[i,j-1]):
                            nbr.append(self.Tr0[i,j-1])
                        if j < self.cols-1 and self.lake_h[i,j+1] > -eps and np.isfinite(self.cmask[i,j+1]):
                            nbr.append(self.Tr0[i,j+1])
                        if nbr:
                            self.Tr0[i,j] = np.mean(nbr)

        # For Cauchy ditch cells: set Tr to mean of aquifer neighbours
        if self.ditch_boundary == 'Cauchy':
            for i in range(self.rows):
                for j in range(self.cols):
                    if self.ditch_h[i,j] < -eps and np.isfinite(self.cmask[i,j]):
                        nbr = []
                        if i > 0 and self.ditch_h[i-1,j] > -eps and np.isfinite(self.cmask[i-1,j]):
                            nbr.append(self.Tr0[i-1,j])
                        if i < self.rows-1 and self.ditch_h[i+1,j] > -eps and np.isfinite(self.cmask[i+1,j]):
                            nbr.append(self.Tr0[i+1,j])
                        if j > 0 and self.ditch_h[i,j-1] > -eps and np.isfinite(self.cmask[i,j-1]):
                            nbr.append(self.Tr0[i,j-1])
                        if j < self.cols-1 and self.ditch_h[i,j+1] > -eps and np.isfinite(self.cmask[i,j+1]):
                            nbr.append(self.Tr0[i,j+1])
                        if nbr:
                            self.Tr0[i,j] = np.mean(nbr)

        # transmissivity at cell interfaces (harmonic or geometric mean, see self.transmissivity_mean)
        TrTmpEW, TrTmpNS = self._interface_transmissivity(self.Tr0)
        self.TrW0[:,1:] = TrTmpEW
        self.TrE0[:,:-1] = TrTmpEW
        self.TrN0[1:,:] = TrTmpNS
        self.TrS0[:-1,:] = TrTmpNS
        del TrTmpEW, TrTmpNS

        # Head in four neighbouring cells
        self.HW[:,1:] = self.H[:,:-1]
        self.HE[:,:-1] = self.H[:,1:]
        self.HN[1:,:] = self.H[:-1,:]
        self.HS[:-1,:] = self.H[1:,:]

        # ravel 2D arrays
        # to avoid reshaping, save in other variable
        TrW0 = np.ravel(self.TrW0)
        TrE0 = np.ravel(self.TrE0)
        TrN0 = np.ravel(self.TrN0)
        TrS0 = np.ravel(self.TrS0)

        # from previous timestep
        TrW1 = TrW0.copy()
        TrE1 = TrE0.copy()
        TrN1 = TrN0.copy()
        TrS1 = TrS0.copy()

        # hydraulic heads, new iteration and old iteration
        # physical floor: gwl cannot go below the profile's impermeable bottom
        # (self.deep_z, a negative depth). Without this, a cell can numerically
        # drift far outside the tabulated gwl_to_Tr/gwl_to_C range, where
        # interp1d(fill_value='extrapolate') can behave wildly and drive a
        # self-sustaining oscillation that no amount of sub-stepping/damping
        # escapes (seen as gwl stuck at ~-24 m next to a -0.26 m ditch cell)
        H_floor = self._H_floor  # raveled version used inside the loop below
        Htmp = np.maximum(self.H.copy(), self._H_floor_2d)
        Htmp1 = Htmp.copy()

        # crit and self.implic are set once per external timestep in run_timestep()
        # (constant across any internal sub-steps of the same timestep)

        # Precompute transmissivity at ditch water level (constant throughout iteration)
        # T(ditch_h) is the transmissivity of the saturated zone below the ditch level.
        # C_dd = (T(gwl) - T(ditch_h)) * L / d  captures only the slice above the ditch.
        if self.ditch_boundary == 'Cauchy':
            Tr_ditch_2d = np.zeros_like(self.Tr0)
            if not self.z_from_gis:
                for key, value in self.gwl_to_Tr.items():
                    mask = (self.soiltype == key) & (self.ditch_h < -eps)
                    Tr_ditch_2d[mask] = value(self.ditch_h[mask])
            elif self.z_from_gis:
                for i in range(self.gwl_to_Tr.shape[0]):
                    for j in range(self.gwl_to_Tr.shape[1]):
                        if np.isfinite(self.cmask[i, j]) and self.ditch_h[i, j] < -eps:
                            Tr_ditch_2d[i, j] = self.gwl_to_Tr[i, j](self.ditch_h[i, j])
            Tr_ditch = np.ravel(Tr_ditch_2d)

        converged = False

        for it in range(self.maxiter):
            # transmissivity [m2 d-1] to neighbouring cells with HTmp1
            # see the pre-loop Tr0 computation above for why lakes are
            # excluded from the H_neighbours substitution under Cauchy
            if self.ditch_boundary == 'Dirichlet':
                H_for_Tr = np.where((bc_h_2d < -eps) & (H_neighbours_2d > self.ele + bc_h_2d),
                                    H_neighbours_2d, Htmp)
            else:
                H_for_Tr = Htmp
            # transmissivities based on gwl
            if not self.z_from_gis:
                for key, value in self.gwl_to_Tr.items():
                    self.Tr1[self.soiltype == key] = value(H_for_Tr[self.soiltype == key] - self.ele[self.soiltype == key])
            elif self.z_from_gis:
                for i in range(self.gwl_to_Tr.shape[0]):
                    for j in range(self.gwl_to_Tr.shape[1]):
                        if np.isfinite(self.cmask[i,j]):
                            self.Tr1[i,j] = self.gwl_to_Tr[i,j](H_for_Tr[i,j] - self.ele[i,j])

            # For Cauchy lake cells: set Tr to mean of aquifer neighbours
            # (run before the ditch block below, see pre-loop comment above)
            if self.ditch_boundary == 'Cauchy':
                for i in range(self.rows):
                    for j in range(self.cols):
                        if self.lake_h[i,j] < -eps and np.isfinite(self.cmask[i,j]):
                            nbr = []
                            if i > 0 and self.lake_h[i-1,j] > -eps and np.isfinite(self.cmask[i-1,j]):
                                nbr.append(self.Tr1[i-1,j])
                            if i < self.rows-1 and self.lake_h[i+1,j] > -eps and np.isfinite(self.cmask[i+1,j]):
                                nbr.append(self.Tr1[i+1,j])
                            if j > 0 and self.lake_h[i,j-1] > -eps and np.isfinite(self.cmask[i,j-1]):
                                nbr.append(self.Tr1[i,j-1])
                            if j < self.cols-1 and self.lake_h[i,j+1] > -eps and np.isfinite(self.cmask[i,j+1]):
                                nbr.append(self.Tr1[i,j+1])
                            if nbr:
                                self.Tr1[i,j] = np.mean(nbr)

            # For Cauchy ditch cells: set Tr to mean of aquifer neighbours
            if self.ditch_boundary == 'Cauchy':
                for i in range(self.rows):
                    for j in range(self.cols):
                        if self.ditch_h[i,j] < -eps and np.isfinite(self.cmask[i,j]):
                            nbr = []
                            if i > 0 and self.ditch_h[i-1,j] > -eps and np.isfinite(self.cmask[i-1,j]):
                                nbr.append(self.Tr1[i-1,j])
                            if i < self.rows-1 and self.ditch_h[i+1,j] > -eps and np.isfinite(self.cmask[i+1,j]):
                                nbr.append(self.Tr1[i+1,j])
                            if j > 0 and self.ditch_h[i,j-1] > -eps and np.isfinite(self.cmask[i,j-1]):
                                nbr.append(self.Tr1[i,j-1])
                            if j < self.cols-1 and self.ditch_h[i,j+1] > -eps and np.isfinite(self.cmask[i,j+1]):
                                nbr.append(self.Tr1[i,j+1])
                            if nbr:
                                self.Tr1[i,j] = np.mean(nbr)
            
            TrTmpEW, TrTmpNS = self._interface_transmissivity(self.Tr1)
            self.TrW1[:,1:] = TrTmpEW
            self.TrE1[:,:-1] = TrTmpEW
            self.TrN1[1:,:] = TrTmpNS
            self.TrS1[:-1,:]=TrTmpNS
            del TrTmpEW, TrTmpNS
            # ravel 2D arrays
            TrW1 = np.ravel(self.TrW1); TrE1= np.ravel(self.TrE1)
            TrN1 = np.ravel(self.TrN1); TrS1 = np.ravel(self.TrS1)

            # differential water capacity dSto/dh
            if not self.z_from_gis:
                for key, value in self.gwl_to_C.items():
                    self.CC[self.soiltype == key] = value(Htmp[self.soiltype == key] - self.ele[self.soiltype == key])
                for key, value in self.gwl_to_wsto.items():
                    self.Wtso1_deep[self.soiltype == key] = value(Htmp[self.soiltype == key] - self.ele[self.soiltype == key])
            elif self.z_from_gis:
                for i in range(self.gwl_to_C.shape[0]):
                    for j in range(self.gwl_to_C.shape[1]):
                        if np.isfinite(self.cmask[i,j]): 
                            self.CC[i,j] = self.gwl_to_C[i,j](Htmp[i,j] - self.ele[i,j])
                            self.Wtso1_deep[i,j] = self.gwl_to_wsto[i,j](Htmp[i,j] - self.ele[i,j])

            alfa = np.ravel(self.CC * self.dxy**2 / dt)
            # alfa = np.ravel((0.5*self.CC + 0.5*CCtmp) * self.dxy**2 / dt)

            # Setup of diagonal sparse matrix
            a_d = self.implic * (TrW1 + TrE1 + TrN1 + TrS1) + alfa  # Diagonal
            a_w = -self.implic * TrW1[1:]  # West element
            a_e = -self.implic * TrE1[:-1]  # East element
            a_n = -self.implic * TrN1[self.cols:]  # North element
            a_s = -self.implic * TrS1[:self.n-self.cols]  # South element

            # Knowns: Right hand side of the eq
            Htmp = np.ravel(Htmp)
            hs = (np.ravel(S) * self.dxy**2 / dt + alfa * Htmp
                  - np.ravel(self.Wtso1_deep) * self.dxy**2 / dt + Wsto_deep * self.dxy**2 / dt
                  + (1.-self.implic) * (TrN0*HN) + (1.-self.implic) * (TrW0*HW)
                  - (1.-self.implic) * (TrN0 + TrW0 + TrE0 + TrS0) * H
                  + (1.-self.implic) * (TrE0*HE) + (1.-self.implic) * (TrS0*HS))

            # implicit Cauchy drainage: conductance added to diagonal, threshold contribution to RHS
            # C_dd = (T(gwl) - T(ditch_h)) * L / d  [m2 d-1]
            # integrates K(z) over the saturated slice above the ditch level
            if self.ditch_boundary == 'Cauchy':
                Tr1_flat = np.ravel(self.Tr1)
                C_dd = np.maximum(0.0, Tr1_flat - Tr_ditch) * ditch_l / ditch_d
                ditch_active = (ditch_h < -eps) & (Htmp > ele + ditch_h)
                a_d[ditch_active] += C_dd[ditch_active]
                hs[ditch_active] += C_dd[ditch_active] * (ele[ditch_active] + ditch_h[ditch_active])

            # Constant-head boundary cells (lakes for Cauchy; ditches+lakes for Dirichlet)
            for k in np.where(bc_h < -eps)[0]:
                if H_neighbours[k] > ele[k] + bc_h[k]:
                    hs[k] = ele[k] + bc_h[k]
                    a_d[k] = 1
                    if k%self.cols != 0:  # west node
                        a_w[k-1] = 0
                    if (k+1)%self.cols != 0:  # east node
                        a_e[k] = 0
                    if k-self.cols >= 0:  # north node
                        a_n[k-self.cols] = 0
                    if k+self.cols < self.n:  # south node
                        a_s[k] = 0

            # Guard against singular matrix: NaN in RHS
            bad_rhs = ~np.isfinite(hs)
            if np.any(bad_rhs):
                terms = {
                    'S':        np.ravel(S) * self.dxy**2 / dt,
                    'alfa*Htmp': alfa * Htmp,
                    'Wtso1':    np.ravel(self.Wtso1_deep) * self.dxy**2 / dt,
                    'Wsto':     Wsto_deep * self.dxy**2 / dt,
                    'lateral':  (1.-self.implic) * (TrN0*HN + TrW0*HW + TrE0*HE + TrS0*HS
                                                    - (TrN0+TrW0+TrE0+TrS0)*H),
                }
                nan_terms = [name for name, arr in terms.items() if np.any(~np.isfinite(arr[bad_rhs]))]
                bad_2d = np.argwhere(np.reshape(bad_rhs, (self.rows, self.cols)))
                print(f'  WARNING: {np.sum(bad_rhs)} cell(s) with NaN rhs at timestep {self.tmstep}, it {it}, NaN in: {nan_terms}')
                for idx in bad_2d[:3]:
                    i, j = idx
                    k = i * self.cols + j
                    #print(f'    [{i},{j}] Htmp={Htmp[k]:.3f}, ele={self.ele[i,j]:.3f}, gwl={Htmp[k]-self.ele[i,j]:.3f}'
                    #      f', alfa={alfa[k]:.4g}, Wsto={Wsto_deep[k]:.4g}, Wtso1={np.ravel(self.Wtso1_deep)[k]:.4g}')
                a_d[bad_rhs] = 1.0
                hs[bad_rhs] = Htmp[bad_rhs]

            A = diags([a_d, a_w, a_e, a_n, a_s], [0, -1, 1, -self.cols, self.cols],format='csc')

            # Solve: A*Htmp1 = hs
            Htmp1 = linalg.spsolve(A,hs)

            # keep the iterate within the tabulated gwl range (see H_floor above)
            Htmp1 = np.maximum(Htmp1, H_floor)

            # Use the undamped Picard residual for convergence and diagnostics.
            head_diff = np.abs(Htmp1 - Htmp)
            conv1 = np.max(head_diff)
            max_index = np.unravel_index(np.argmax(head_diff), (self.rows, self.cols))

            # Diagnose cells with large head change before clamping
            large_diff = np.abs(Htmp1 - Htmp) > 0.5
            if np.any(large_diff):
                large_diff_2d = np.reshape(large_diff, (self.rows, self.cols))
                Htmp_2d  = np.reshape(Htmp,  (self.rows, self.cols))
                Htmp1_2d = np.reshape(Htmp1, (self.rows, self.cols))
                problem_indices = np.argwhere(large_diff_2d)
                #print(f'Timestep: {self.tmstep}, it: {it+1}, cells with |dH|>0.5m: {len(problem_indices)}')
                for idx in problem_indices[:2]:  # print at most 5 cells
                    i, j = idx
                    #print(f'  [{i},{j}] gwl: {Htmp_2d[i,j]-self.ele[i,j]:.3f} -> {Htmp1_2d[i,j]-self.ele[i,j]:.3f} m'
                    #      f', ditch_h: {self.ditch_h[i,j]:.3f}'
                    #      f', C_dd: {C_dd[i*self.cols+j]:.4f} m2/d'
                    #      f', Tr: {self.Tr1[i,j]:.4f} m2/d')
                    
            if self.tmstep <= self.spinup_steps:
                Htmp1 = np.where(np.abs(Htmp1-Htmp)> 2.0, Htmp + 0.5*np.sign(Htmp1-Htmp), Htmp1)
            if self.tmstep > self.spinup_steps:
                Htmp1 = np.where(np.abs(Htmp1-Htmp)> 0.5, Htmp + 0.5*np.sign(Htmp1-Htmp), Htmp1)

            # damp the update to stop oscillation (no-op when picard_relax == 1.0)
            if self.picard_relax != 1.0:
                Htmp1 = Htmp + self.picard_relax * (Htmp1 - Htmp)

            # no iteration-count-based relaxation needed: adaptive sub-stepping now
            # handles the oscillation/stiffness this used to be a workaround for
            Htmp = Htmp1.copy()

            Htmp = np.reshape(Htmp,(self.rows,self.cols))

            # print to get sense what's happening when problems in convergence
            if it > 90:
                print('\t', 'iterations:', it, ' con1:', conv1,
                      ' max_index:', max_index, ' self.ditch_h[max_index]', self.ditch_h[max_index],
                      ' H[max_index]', Htmp[max_index]-self.ele[max_index])

            if conv1 < crit:
                converged = True
                break
            if it + 1 >= max_iter:
                break  # bail out early (converged stays False); _adaptive_solve halves dt and retries
            # end of iteration loop
        if not converged:
            self.conv99 += 1
        Htmp = np.reshape(Htmp,(self.rows,self.cols))

        # recompute S_dd [m] from converged head for mass balance and output
        if self.ditch_boundary == 'Cauchy':
            H_conv = np.ravel(Htmp)
            Tr1_flat = np.ravel(self.Tr1)
            C_dd_conv = np.maximum(0.0, Tr1_flat - Tr_ditch) * ditch_l / ditch_d
            S_dd = np.where((ditch_h < -eps) & (H_conv > ele + ditch_h),
                            C_dd_conv * (H_conv - (ele + ditch_h)) * dt / self.dxy**2,
                            0.0)
        else:
            S_dd = np.zeros(self.n)

        i, j = max_index
        deep_id_val = self.deep_id[i, j] if hasattr(self, 'deep_id') else 'N/A'
        #print(f'Timestep: {self.tmstep}, iterations: {it}, worst conv1: {conv1:.4f} m'
        #      f' at [{i},{j}] gwl: {Htmp[i,j]-self.ele[i,j]:.3f} m'
        #      f', ditch_h: {self.ditch_h[i,j]:.3f}'
        #      f', Tr: {self.Tr1[i,j]:.4f} m2/d'
        #      f', deep_id: {deep_id_val}')
        if not converged:
            # uses head_diff from the breaking iteration (Htmp already equals Htmp1 by this point)
            # stored (not printed here) since this attempt may still be resolved by halving;
            # _adaptive_solve prints it only if this non-converged result ends up being accepted
            non_conv = np.reshape(head_diff, (self.rows, self.cols)) > crit
            non_conv &= np.isfinite(self.ele)
            self._last_non_conv_summary = (int(np.sum(non_conv)),
                np.unique(self.deep_id[non_conv], return_counts=True) if hasattr(self, 'deep_id') else None)
        
        # lateral flow [m d-1] is calculated in two parts: one depending on previous time step
        # and other on current time step (lateral flowsee 2/2). Their weighting depends
        # on self.implic
        # lateral flow 1/2 with old heads (and old transmissivities if updated inside the loop)
        # use 1-self.implic
        lateral_flow = ((1-self.implic)*(self.TrW0*(self.H - self.HW)
                        + self.TrE0*(self.H - self.HE)
                        + self.TrN0*(self.H - self.HN)
                        + self.TrS0*(self.H - self.HS)))/ self.dxy**2

        """ update state """
        # soil profile
        self.H = Htmp.copy()
        self.gwl = self.H - self.ele

        # water storages according to new gwl
        if not self.z_from_gis:
            for key, value in self.gwl_to_wsto.items():
                self.Wsto_deep[self.soiltype == key] = value(self.gwl[self.soiltype == key])
        elif self.z_from_gis:
            for i in range(self.gwl_to_wsto.shape[0]):
                for j in range(self.gwl_to_wsto.shape[1]):
                    if np.isfinite(self.cmask[i,j]):
                        self.Wsto_deep[i,j] = self.gwl_to_wsto[i,j](self.gwl[i,j])

        # Head in four neighbouring cells
        self.HW[:,1:] = self.H[:,:-1]
        self.HE[:,:-1] = self.H[:,1:]
        self.HN[1:,:] = self.H[:-1,:]
        self.HS[:-1,:] = self.H[1:,:]

        # lateral flow 2/2 [m d-1] with new heads (and new transmissivities if updated inside the loop)
        # use self.implic here
        lateral_flow += (self.implic*(self.TrW1*(self.H - self.HW)
                        + self.TrE1*(self.H - self.HE)
                        + self.TrN1*(self.H - self.HN)
                        + self.TrS1*(self.H - self.HS)))/ self.dxy**2

        # Reshape directional transmissivities for output
        TrW = np.reshape(self.TrW1, (self.rows, self.cols)) * self.cmask
        TrE = np.reshape(self.TrE1, (self.rows, self.cols)) * self.cmask
        TrN = np.reshape(self.TrN1, (self.rows, self.cols)) * self.cmask
        TrS = np.reshape(self.TrS1, (self.rows, self.cols)) * self.cmask

        # Let's limit head to 0 and assign rest as return flow to bucketgrid
        Wsto_before_qr = self.Wsto_deep.copy()

        # restrict gwl: cap land cells at 0, cap BC cells at their water level
        self.gwl = np.where(bc_h_2d < -eps, np.minimum(self.gwl, bc_h_2d), np.minimum(0.0, self.gwl))
        self.H = self.gwl + self.ele
        self.H[np.isnan(self.H)] = -999

        # Updating the storage according to new head
        if not self.z_from_gis:
            for key, value in self.gwl_to_wsto.items():
                self.Wsto_deep[self.soiltype == key] = value(self.H[self.soiltype == key] - self.ele[self.soiltype == key])
            for key, value in self.gwl_to_rootmoist.items():
                self.deepmoist[self.soiltype == key] = value(self.gwl[self.soiltype == key])
        elif self.z_from_gis:
            for i in range(self.gwl_to_wsto.shape[0]):
                for j in range(self.gwl_to_wsto.shape[1]):
                    if np.isfinite(self.cmask[i,j]): 
                        self.Wsto_deep[i,j] = self.gwl_to_wsto[i,j](self.H[i,j] - self.ele[i,j])  
                        self.deepmoist[i,j] = self.gwl_to_rootmoist[i,j](self.gwl[i,j])

        # Koivusalo et al. 2008 HESS without wet side limit (SpaFHy_Peat/soilprofile.py)
        self.Rew = np.where(self.deepmoist > self.root_fc1,
                            np.minimum(1.0, 0.5*(1 + (self.deepmoist - self.root_fc1)/(self.root_fc0 - self.root_fc1))),
                            np.maximum(0.0, 0.5*(self.deepmoist - self.root_wp)/(self.root_fc1 - self.root_wp))
                            )

        # The difference is the return flow to bucketgrid
        self.qr = Wsto_before_qr - self.Wsto_deep

        # air volume
        self.airv_deep = np.maximum(0.0, self.Wsto_deep_max - self.Wsto_deep)

        if self.ditch_boundary == 'Cauchy':
            # Lakes are constant-head: netflow to lake from mass balance
            netflow_to_lake = np.where(
                (self.lake_h < -eps) & (H_neighbours_2d > self.ele + self.lake_h),
                state0 - np.reshape(S_dd,(self.rows,self.cols)) - self.Wsto_deep - self.qr - lateral_flow * dt,
                0.0)

            # mass balance error [m]
            mbe = (state0 - np.reshape(S_dd,(self.rows,self.cols)) - self.Wsto_deep - self.qr - lateral_flow * dt
                   - netflow_to_lake * dt)
            mbe = np.where(self.lake_h < -eps, 0.0, mbe)

            # outputs multiplied by cmask
            h_out = self.gwl.copy() * self.cmask
            lateral_flow = lateral_flow * self.cmask
            netflow_to_lake = netflow_to_lake * self.cmask
            netflow_to_ditch = np.reshape(S_dd,(self.rows,self.cols)) * self.cmask
            mbe = mbe * self.cmask
            Wsto_deep_out = self.Wsto_deep.copy() * self.cmask

            results = {
                    'ground_water_level': h_out,  # [m]
                    'lateral_netflow': -lateral_flow * 1e3,  # [mm d-1]
                    'netflow_to_ditch': netflow_to_ditch * 1e3 / dt,  # [mm d-1]
                    'netflow_to_lake': netflow_to_lake * 1e3 / dt,  # [mm d-1]
                    'water_closure': mbe * 1e3 / dt,  # [mm d-1]
                    'water_storage': Wsto_deep_out * 1e3,  # [mm]
                    'return_flow': self.qr * 1e3,  # [mm]
                    'transpiration': self.tr_deep * self.cmask * 1e3,  # [mm]
                    'moisture_deep': self.deepmoist * self.cmask,  # [m3 m-3]
                    'transpiration_limitation': self.Rew * self.cmask,  # [-]
                    'transmissivity': np.nanmean([TrW, TrE, TrN, TrS], axis=0),  # [m2 d-1]
                    'transmissivity_W': TrW,  # [m2 d-1]
                    'transmissivity_E': TrE,  # [m2 d-1]
                    'transmissivity_N': TrN,  # [m2 d-1]
                    'transmissivity_S': TrS,  # [m2 d-1]
                    }

        else:  # Dirichlet: ditches+lakes are constant-head, netflow_to_ditch from mass balance
            mbe = (state0 - self.Wsto_deep - self.qr - lateral_flow * dt)
            active_boundary = (bc_h_2d < -eps) & (H_neighbours_2d > self.ele + bc_h_2d)
            netflow_to_ditch = np.where(active_boundary, mbe, 0.0)
            mbe = mbe - netflow_to_ditch

            # outputs multiplied by cmask
            h_out = self.gwl.copy() * self.cmask
            lateral_flow = lateral_flow * self.cmask
            netflow_to_ditch = netflow_to_ditch * self.cmask
            mbe = mbe * self.cmask
            Wsto_deep_out = self.Wsto_deep.copy() * self.cmask

            results = {
                    'ground_water_level': h_out,  # [m]
                    'lateral_netflow': -lateral_flow * 1e3 / dt,  # [mm d-1]
                    'netflow_to_ditch': netflow_to_ditch * 1e3 / dt,  # [mm d-1]
                    # Dirichlet merges lakes into ditch_h (see __init__), so lake flow is
                    'netflow_to_lake': np.zeros_like(netflow_to_ditch) * self.cmask,  # [mm d-1]
                    'water_closure': mbe * 1e3 / dt,  # [mm d-1]
                    'water_storage': Wsto_deep_out * 1e3,  # [mm]
                    'return_flow': self.qr * 1e3,  # [mm]
                    'transpiration': self.tr_deep * self.cmask * 1e3,  # [mm]
                    'moisture_deep': self.deepmoist * self.cmask,  # [m3 m-3]
                    'transpiration_limitation': self.Rew * self.cmask,  # [-]
                    'transmissivity': np.nanmean([TrW, TrE, TrN, TrS], axis=0),  # [m2 d-1]
                    'transmissivity_W': TrW,  # [m2 d-1]
                    'transmissivity_E': TrE,  # [m2 d-1]
                    'transmissivity_N': TrN,  # [m2 d-1]
                    'transmissivity_S': TrS,  # [m2 d-1]
                    }

        return converged, results


def connectivity_scalar(gwl):
    """
    Lateral connectivity scalar as a function of groundwater level.

    Represents how hydraulically connected the soil system is at a given
    water table depth. When the water table is near the surface the system
    is fully connected; when deep it is nearly isolated.

    Currently not used.

    Piecewise function:
        gwl >= -0.3 m :              scalar = 1.0   (fully connected)
        -0.5 m < gwl < -0.3 m :     scalar = linear interpolation 1e-3 → 1.0
        gwl <= -0.5 m :              scalar = 1e-8  (nearly disconnected)

    Args:
        gwl (float or array): groundwater level [m], <= 0

    Returns:
        scalar (float or array): connectivity multiplier [-], in [1e-8, 1.0]
    """
    gwl_high = -0.3   # [m] fully connected above this threshold
    gwl_low  = -0.5   # [m] transition ends here
    s_high   = 1.0    # scalar at gwl_high and above
    s_low    = 1e-3   # scalar at gwl_low (transition lower end)
    s_floor  = 1e-7   # scalar below gwl_low

    # normalised position in transition zone: 0 at gwl_low, 1 at gwl_high
    t = (gwl - gwl_low) / (gwl_high - gwl_low)
    t = np.clip(t, 0.0, 1.0)
    scalar = s_low + t * (s_high - s_low)  # linear from 1e-3 to 1.0

    # apply floor below gwl_low
    scalar = np.where(np.asarray(gwl) <= gwl_low, s_floor, scalar)

    # return same type as input (scalar in, scalar out)
    if np.ndim(gwl) == 0:
        return float(scalar)
    return scalar

def connectivity_scalar_exp(gwl, gwl_high=-0.5, s_ref=1e-7, gwl_ref=-5.0):
    """
    Exponential decay connectivity scalar, clipped to 1.0 above gwl_high.
    Anchored at gwl_high so the transition is smooth (no discontinuity):
        gwl >= gwl_high :  scalar = 1.0
        gwl <  gwl_high :  scalar = exp(k * (gwl - gwl_high))
                        where k = ln(s_ref) / (gwl_ref - gwl_high)

    Currently not used.

    Args:
        gwl      (float or array): groundwater level [m], <= 0
        gwl_high (float): threshold above which scalar = 1.0 [m].
        s_ref    (float): scalar value at gwl_ref. Default 1e-7.
        gwl_ref  (float): reference depth [m] where scalar = s_ref.

    Returns:
        scalar (float or array): connectivity multiplier [-], in (0, 1.0]
    """ 

    k = np.log(s_ref) / (gwl_ref - gwl_high)
    scalar = np.exp(k * (np.asarray(gwl, dtype=float) - gwl_high))
    scalar = np.where(np.asarray(gwl) >= gwl_high, 1.0, scalar)

    if np.ndim(gwl) == 0:
        return float(scalar)
    return scalar

def gwl_Wsto(z, pF, grid_step=-0.01, Ksat=None, root=False):
    """
    Builds scipy interpolation functions relating groundwater level (gwl) to
    soil column water storage, transmissivity, and differential water capacity
    for a single soil profile (soiltype-wise lookup).

    Args:
        z         (array): Depths of soil layer boundaries [m], negative downward
                           (e.g. [-0.1, -0.3, -0.6, -1.0]).
        pF        (dict):  Van Genuchten water retention parameters, each an array
                           with one value per soil layer:
                               'ThetaS' [m3 m-3]  saturated water content
                               'ThetaR' [m3 m-3]  residual water content
                               'alpha'  [cm-1]    air entry suction
                               'n'      [-]        pore size distribution
        grid_step (float): Step size for the internal gwl grid [m]. Default -0.01.
        Ksat      (array): Saturated hydraulic conductivity per layer [m s-1].
                           Required unless root=True.
        root      (bool):  If True, returns only the gwl → root zone moisture
                           function instead of the full set.

    Returns:
        dict with keys (unless root=True, which returns only 'to_rootmoist'):
            'to_gwl'   callable: Wsto → gwl interpolator
            'to_wsto'  callable: gwl → water storage [m] interpolator
            'to_C'     callable: gwl → differential water capacity [m m-1] interpolator
            'to_Tr'    callable: gwl → transmissivity [m2 d-1] interpolator
    """
    z = np.array(z, dtype=np.float64) # profile depths
    dz = abs(z)
    dz[1:] = z[:-1] - z[1:] # profile depths into profile thicknesses

    # finer grid for calculating wsto to avoid discontinuity in C (dWsto/dGWL)
    z_fine= (np.arange(0, min(z), grid_step) - grid_step).astype(np.float64)
    dz_fine = z_fine*0.0 - grid_step
    z_mid_fine = dz_fine / 2 - np.cumsum(dz_fine)

    ix = np.zeros(len(z_fine), dtype=np.float64)

    for depth in z:
        # below makes sure floating point precision doesnt mess with the ix
        ix += np.where((z_fine < depth) & ~np.isclose(z_fine, depth, atol=1e-9), 1, 0)

    pF_fine={}
    for key in pF.keys():
        pp = []
        for i in range(len(z_fine)):
            pp.append(pF[key][int(ix[i])])
        pF_fine.update({key: np.array(pp)})

    # --------- connection between gwl and Wsto, Tr, C------------
    gwl = np.arange(1.0, min(z)-5, grid_step)
    # solve water storage corresponding to gwls
    Wsto_deep = [sum(h_to_cellmoist(pF_fine, g - z_mid_fine, dz_fine) * dz_fine)
            + max(0.0,g) for g in gwl]  # water storage above ground surface == gwl

    if root:
        Wsto_deep = [sum(h_to_cellmoist(pF_fine, g - z_mid_fine, dz_fine) * dz_fine) for g in gwl]
        Wsto_deep = Wsto_deep/sum(dz)
        GwlToWsto = interp1d(np.array(gwl), np.array(Wsto_deep), fill_value='extrapolate')
        return {'to_rootmoist': GwlToWsto}

    # solve transmissivity corresponding to gwls
    Tr = [transmissivity(dz, Ksat, g) * 86400. for g in gwl]  # [m2 d-1]

    #print('np.array(gwl).shape', np.array(gwl).shape)
    #print('np.array(Wsto_deep).shape', np.array(Wsto_deep).shape)

    # interpolate functions
    WstoToGwl = interp1d(np.array(Wsto_deep), np.array(gwl), fill_value='extrapolate')
    GwlToWsto = interp1d(np.array(gwl), np.array(Wsto_deep), fill_value='extrapolate')
    GwlToC = interp1d(np.array(gwl), np.array(np.gradient(Wsto_deep)/np.gradient(gwl)), fill_value='extrapolate')
    GwlToTr = interp1d(np.array(gwl), np.array(Tr), fill_value='extrapolate')
    
    plots = False
    if plots == True:
        import os, time
        import matplotlib.pyplot as plt

        os.makedirs('figs', exist_ok=True)
        _ts = int(time.time() * 1000)

        fig1, ax1 = plt.subplots()
        ax1.plot(np.array(gwl), np.array(np.gradient(Wsto_deep)/np.gradient(gwl)))
        ax1.set_xlabel('Groundwater level (gwl) [m]')
        ax1.set_ylabel('dWsto/dgwl [m m$^{-1}$]')
        ax1.set_title('Differential water capacity (C) vs. groundwater level')
        fig1.tight_layout()
        fig1.savefig(f'figs/gwl_vs_differential_water_capacity_{_ts}.png', dpi=150)
        plt.close(fig1)

        fig2, ax2 = plt.subplots()
        #ax2.plot(np.array(gwl), np.log10(np.array(Tr)), label='log10(Tr)')
        ax2.plot(np.array(gwl), np.array(Tr), label='Tr')
        ax2.set_xlabel('Groundwater level (gwl) [m]')
        ax2.set_ylabel('Transmissivity [m$^2$ d$^{-1}$]')
        ax2.set_title('Transmissivity vs. groundwater level')
        ax2.legend()
        fig2.tight_layout()
        fig2.savefig(f'figs/gwl_vs_transmissivity_{_ts}.png', dpi=150)
        plt.close(fig2)

        fig3, ax3 = plt.subplots()
        ax3.plot(np.array(gwl), np.array(Wsto_deep))
        ax3.set_xlabel('Groundwater level (gwl) [m]')
        ax3.set_ylabel('Water storage (Wsto) [m]')
        ax3.set_title('Water storage vs. groundwater level')
        fig3.tight_layout()
        fig3.savefig(f'figs/gwl_vs_water_storage_{_ts}.png', dpi=150)
        plt.close(fig3)

    return {'to_gwl': WstoToGwl, 'to_wsto': GwlToWsto, 'to_C': GwlToC, 'to_Tr': GwlToTr}

def h_to_cellmoist(pF, h, dz):
    r""" Cell moisture based on vanGenuchten-Mualem soil water retention model.
    Partly saturated cells calculated as thickness weigthed average of
    saturated and unsaturated parts.

    Args:
        pF (dict):
            'ThetaS' (array): saturated water content [m\ :sup:`3` m\ :sup:`-3`\ ]
            'ThetaR' (array): residual water content [m\ :sup:`3` m\ :sup:`-3`\ ]
            'alpha' (array): air entry suction [cm\ :sup:`-1`]
            'n' (array): pore size distribution [-]
        h (array): pressure head [m]
        dz (array): soil conpartment thichness, node in center [m]
    Returns:
        theta (array): volumetric water content of cell [m\ :sup:`3` m\ :sup:`-3`\ ]

    Kersti Haahti, Luke 8/1/2018
    """

    # water retention parameters
    Ts = np.array(pF['ThetaS'])
    Tr = np.array(pF['ThetaR'])
    alfa = np.array(pF['alpha'])
    n = np.array(pF['n'])
    m = 1.0 - np.divide(1.0, n)

    # moisture based on cell center head
    x = np.minimum(h, 0)
    theta = Tr + (Ts - Tr) / (1 + abs(alfa * 100 * x)**n)**m

    # correct moisture of partly saturated cells
    ix = np.where(abs(h) < dz/2)
    if len(Ts) == 1:
        ixx = 0
    else:
        ixx = ix
    # moisture of unsaturated part
    x[ix] = -(dz[ix]/2 - h[ix]) / 2
    theta[ix] = Tr[ixx] + (Ts[ixx] - Tr[ixx]) / (1 + abs(alfa[ixx] * 100 * x[ix])**n[ixx])**m[ixx]
    # total moisture as weighted average
    theta[ix] = (theta[ix] * (dz[ix]/2 - h[ix]) + Ts[ixx] * (dz[ix]/2 + h[ix])) / (dz[ix])

    return theta

def transmissivity(dz, Ksat, gwl):
    r""" Transmissivity of saturated layer.

    Args:
       dz (array):  soil compartment thickness, node in center [m]
       Ksat (array): horizontal saturated hydr. cond. [ms-1]
       gwl (float): ground water level below surface, <0 [m]

    Returns:
       Tr (array): tranmissivity [m2 s-1]
       Tr (array): tranmissivity [m2 s-1]
    """
    
    # midpoint of cell, soil surface at 0
    z = dz / 2 - np.cumsum(dz)

    # saturated layer thickness [m], between [0, dz]
    dz_sat = np.minimum(np.maximum(gwl - (z - dz / 2), 0), dz)
    # in top cell, allow transmissivity to increases when gwl above ground surface level
    dz_sat[0] = np.maximum(gwl - (z[0] - dz[0] / 2), 0)  
    
    # transmissivity of layers  [m2 s-1]
    Trans = Ksat * dz_sat

    # sum over layers
    Tr = np.maximum(sum(Trans), 1e-5 / 86400)

    return Tr #* connectivity_scalar_exp(gwl)


def gwl_Wsto_vectorized(z, pF, grid_step=-0.01, Ksat=None, root=False):
    """
    Builds per-cell scipy interpolation functions relating groundwater level
    to soil column water storage, transmissivity, and differential water
    capacity. Used when soil depth varies spatially (z_from_gis=True).

    Supports two grid_step modes:
        float  — uniform spacing (e.g. -0.01 m)
        'var'  — variable spacing: fine near surface (0.01 m), coarser at depth
                 (0.05 m to -1 m, 0.3 m below -5 m); reduces memory for deep profiles (needs testing)

    Args:
        z         (array or 2D array): Soil layer boundary depths [m], negative
                                       downward. Shape (n_cells, n_layers) or (n_layers,).
        pF        (array of dicts):    Van Genuchten parameters per cell/layer.
                                       Each dict has keys 'ThetaS', 'ThetaR', 'alpha', 'n'.
        grid_step (float or 'var'):    Internal gwl grid spacing. Default -0.01.
        Ksat      (array):             Saturated hydraulic conductivity [m s-1] per
                                       cell/layer. Required unless root=True.
        root      (bool):              If True, returns only gwl → root zone moisture.

    Returns:
        dict with keys (unless root=True, which returns only 'to_rootmoist'):
            'to_gwl'   list of callables: Wsto → gwl per cell
            'to_wsto'  list of callables: gwl → water storage [m] per cell
            'to_C'     list of callables: gwl → differential water capacity per cell
            'to_Tr'    list of callables: gwl → transmissivity [m2 d-1] per cell
    """
    # Ensure z is a NumPy array
    z = np.array(z, dtype=np.float32)
    pF = np.array(pF)

    if z.ndim == 1:
        z = np.expand_dims(z, axis=0)
        pF = np.expand_dims(pF, axis=0)
        if Ksat is not None:
            Ksat = np.array(Ksat)
            Ksat = np.expand_dims(Ksat, axis=0)

    dz = np.abs(z)
    dz = np.hstack((dz[:, :1], np.diff(dz, axis=1)))
    
    if isinstance(grid_step, float):
        z_min = np.min(z, axis=1)
        max_len = int(np.abs(np.nanmin(z_min)) / np.abs(grid_step)) + 1
        z_fine = np.tile(np.arange(0., grid_step * max_len, grid_step), (z.shape[0], 1)) + grid_step
        z_fine = z_fine.astype(np.float32)
        z_fine[z_fine < z_min[:, None]] = np.nan
        dz_fine = z_fine*0.0 - grid_step
        z_mid_fine = dz_fine / 2 - np.cumsum(dz_fine, axis=1)
        ix = np.full((z_fine.shape), np.nan)
        # Expand z along the second axis to match z_fine's shape (broadcasting)
        z_expanded = np.expand_dims(z, axis=1)  # Shape: (rows, 1, cols)
        z_fine_expanded = np.expand_dims(z_fine, axis=2)  # Shape: (rows, fine_steps, 1)    

    elif grid_step == 'var':
        z_min = np.min(z, axis=1).astype(np.float32)
        z_min_min = np.nanmin(z_min)
        limits = [-1, -5, z_min_min]
        steps = [-0.01, -0.05, -0.3]
        z1 = np.arange(0+steps[0], limits[0], steps[0])
        z2 = np.arange(limits[0], limits[1], steps[1])
        z3 = np.arange(limits[1], limits[2] + steps[2], steps[2])  # Ensure we reach z_min
        # Combine all segments
        z_values = np.concatenate([z1, z2, z3])  # Ensure exact z_min
        z_fine = np.tile(z_values, (z.shape[0], 1))
        z_fine = z_fine.astype(np.float32)
        z_fine[z_fine < z_min[:, None]] = np.nan
        # Compute dz_fine
        dz_fine = np.abs(np.diff(z_fine, axis=1))  # Compute differences along the second axis
        # Insert the first element (z_fine[:, 0] - 0) at the beginning
        dz_fine = np.hstack([z_fine[:, [0]], dz_fine])
        z_mid_fine = dz_fine / 2 - np.cumsum(dz_fine, axis=1)
        ix = np.full((z_fine.shape), np.nan)
        # Expand z along the second axis to match z_fine's shape (broadcasting)
        z_expanded = np.expand_dims(z, axis=1)  # Shape: (rows, 1, cols)
        z_fine_expanded = np.expand_dims(z_fine, axis=2)  # Shape: (rows, fine_steps, 1)

    # Compute mask using broadcasting (row-wise comparison)
    mask = (z_fine_expanded < z_expanded) & ~np.isclose(z_fine_expanded, z_expanded, atol=1e-9)

    # Sum along the depth dimension to count how many times z_fine falls below z
    ix = np.sum(mask, axis=2).astype(np.float64)  # Convert to float to retain NaN compatibility

    pF_fine = {}

    for key in pF[0].keys():  # Iterate over each parameter in `pF`
        # Convert pF into an array ensuring consistent shapes
        try:
            pF_array = np.vstack([p[key] for p in pF])  # Ensures (rows, depths) shape
        except ValueError:  # If rows have different lengths, handle it gracefully
            max_depth = max(len(p[key]) for p in pF)  # Find the longest row
            pF_array = np.full((len(pF), max_depth), np.nan)  # Initialize padded array

            # Fill rows with actual values
            for i, p in enumerate(pF):
                pF_array[i, :len(p[key])] = p[key]

        # Ensure `ix` values are within valid range (clip to prevent indexing errors)
        ix_valid = np.clip(ix.astype(int), 0, pF_array.shape[1] - 1)

        # Assign values using vectorized indexing
        pF_fine[key] = np.take_along_axis(pF_array, ix_valid, axis=1)  # Shape: (rows, fine_steps)

    # --------- connection between gwl and Wsto, Tr, C------------
    if isinstance(grid_step, float):
        gwl = np.arange(1.0, min(z_min)-5, grid_step)
    elif grid_step == 'var':
        limits = [-1, -5, min(z_min)-5]
        steps = [-0.01, -0.05, -0.5]
        # First segment: 1m to 0m (step = 1.0m)
        z1 = np.arange(1, limits[0], steps[0])
        z2 = np.arange(limits[0], limits[1], steps[1])
        z3 = np.arange(limits[1], limits[2], steps[2])
        gwl = np.concatenate([z1, z2, z3])  # Ensure exact z_min

    Wsto_deep = np.stack([h_to_cellmoist_vectorized(pF_fine, g - z_mid_fine, dz_fine) + max(0.0, g) for g in gwl]).T

    if root:
        Wsto_deep = np.stack([h_to_cellmoist_vectorized(pF_fine, g - z_mid_fine, dz_fine) for g in gwl]).T
        Wsto_deep = Wsto_deep/np.nansum(dz, axis=1)
        #GwlToWsto = interp1d(np.array(gwl), np.array(Wsto_deep), fill_value='extrapolate')
        GwlToWsto = [interp1d(gwl, wsto_row, kind='linear', fill_value='extrapolate') for wsto_row in Wsto_deep]
        return {'to_rootmoist': GwlToWsto}

    Tr1 = np.stack([transmissivity_vectorized(dz, Ksat, g) * 86400. for g in gwl]).T

    # Generate interpolators for each row of Wsto_deep and Tr1 while keeping gwl the same
    WstoToGwl = [interp1d(wsto_row, gwl, kind='linear', fill_value='extrapolate') for wsto_row in Wsto_deep]
    GwlToWsto = [interp1d(gwl, wsto_row, kind='linear', fill_value='extrapolate') for wsto_row in Wsto_deep]
    GwlToC = [interp1d(gwl, np.gradient(wsto_row) / np.gradient(gwl), kind='linear', fill_value='extrapolate') for wsto_row in Wsto_deep]
    GwlToTr = [interp1d(gwl, tr_row, kind='linear', fill_value='extrapolate') for tr_row in Tr1]
    
    #plt.figure(1)
    #plt.plot(np.array(gwl), np.array(np.gradient(Wsto_deep[0])/np.gradient(gwl)), linestyle='--')
    #plt.figure(2)
    #plt.plot(np.array(gwl), np.log10(np.array(Tr1[0])), linestyle='--')
    #plt.plot(np.array(gwl), np.array(Tr1[0]), linestyle='--')
    #plt.figure(3)
    #plt.plot(np.array(gwl), np.array(Wsto_deep[0]), linestyle='--')

    return {'to_gwl': WstoToGwl, 'to_wsto': GwlToWsto, 'to_C': GwlToC, 'to_Tr': GwlToTr}

def h_to_cellmoist_vectorized(pF, h, dz):
    r""" Cell moisture based on vanGenuchten-Mualem soil water retention model.
    Partly saturated cells calculated as thickness weigthed average of
    saturated and unsaturated parts.

    Args:
        pF (np.ndarray):
            dict
                'ThetaS' (np.ndarray): saturated water content [m\ :sup:`3` m\ :sup:`-3`\ ]
                'ThetaR' (np.ndarray): residual water content [m\ :sup:`3` m\ :sup:`-3`\ ]
                'alpha' (np.ndarray): air entry suction [cm\ :sup:`-1`]
                'n' (np.ndarray): pore size distribution [-]
        h (float): pressure head [m]
        dz (np.ndarray): soil compartment thichness, node in center [m]
    Returns:
        theta (np.ndarray): Total volumetric water content of cell for given gwl
    """

    # water retention parameters
    Ts = np.array(pF['ThetaS'])
    Tr = np.array(pF['ThetaR'])
    alfa = np.array(pF['alpha'])
    n = np.array(pF['n'])
    m = 1.0 - np.divide(1.0, n)

    # moisture based on cell center head
    x = np.minimum(h, 0)
    theta = Tr + (Ts - Tr) / (1 + abs(alfa * 100 * x)**n)**m

    # correct moisture of partly saturated cells
    # use the full 2D h so each cell gets its own partly-saturated correction,
    # not just the correction derived from row 0
    ix = np.where(abs(h) < dz/2)
    ixx = ix
    # moisture of unsaturated part
    x[ix] = -(dz[ix]/2 - h[ix]) / 2
    theta[ix] = Tr[ixx] + (Ts[ixx] - Tr[ixx]) / (1 + abs(alfa[ixx] * 100 * x[ix])**n[ixx])**m[ixx]
    # total moisture as weighted average
    theta[ix] = (theta[ix] * (dz[ix]/2 - h[ix]) + Ts[ixx] * (dz[ix]/2 + h[ix])) / (dz[ix])
    # from vwc to total water content
    Wsto = theta * dz
    # 
    Wsto = np.nansum(Wsto, axis=1)
    
    return Wsto

def transmissivity_vectorized(dz, Ksat, gwl):
    r""" Vectorized transmissivity function for 2D inputs.

    Args:
       dz (np.ndarray):  Soil compartment thickness, node in center [m]
       Ksat (np.ndarray): Horizontal saturated hydraulic conductivity [m/s]
       gwl (float): Groundwater level below surface, <0 [m]

    Returns:
       Tr (np.ndarray): Transmissivity for each cell [m²/s]
    """

    # Compute midpoints of layers
    z = dz / 2 - np.cumsum(dz, axis=1)  # Shape: (n_cells, n_layers)

    # Compute saturated thickness for each layer, between [0, dz]
    dz_sat = np.minimum(np.maximum(gwl - (z - dz / 2), 0), dz)  # Shape: (n_cells, n_layers)
    # In top cell allow transmissivity to increases when gwl above ground surface level
    dz_sat[:, 0] = np.maximum(gwl - (z[:, 0] - dz[:, 0] / 2), 0) 
    # Compute saturated thickness for each layer, between [0, dz]
    dz_sat = np.minimum(np.maximum(gwl - (z - dz / 2), 0), dz)  # Shape: (n_cells, n_layers)
    # In top cell allow transmissivity to increases when gwl above ground surface level
    dz_sat[:, 0] = np.maximum(gwl - (z[:, 0] - dz[:, 0] / 2), 0) 

    # Compute transmissivity of each layer
    Trans = Ksat * dz_sat  # Shape: (n_cells, n_layers)

    #return np.nansum(Trans, axis=1)
    return np.maximum(np.nansum(Trans, axis=1), 1e-5 / 86400) #* connectivity_scalar_exp(gwl)


def wrc(pF, theta=None, psi=None, draw_pF=False):
    """
    vanGenuchten-Mualem soil water retention model 

    References:
        Schaap and van Genuchten (2005). Vadose Zone 5:27-34
        van Genuchten, (1980). Soil Science Society of America Journal 44:892-898

    Args:
        pF (dict):
            ThetaS (float|array): saturated water content [m3 m-3]
            ThetaR (float|array): residual water content [m3 m-3]
            alpha (float|array): air entry suction [cm-1]
            n (float|array): pore size distribution [-]
        theta (float|array): vol. water content [m3 m-3]
        psi (float|array): water potential [m]
        draw_pF (bool): Draw pF-curve.
    Returns:
        y (float|array): water potential [m] or vol. water content [m3 m-3]. Returns None if only curve is drawn.

    """
    
    EPS = np.finfo(float).eps
    
    Ts = np.array(pF['ThetaS'])
    Tr = np.array(pF['ThetaR'])
    alfa = np.array(pF['alpha'])
    n = np.array(pF['n'])
    m = 1.0 - np.divide(1.0, n)

    def theta_psi(x):
        # converts water content [m3 m-3] to potential [m]]
        x = np.minimum(x, Ts)
        x = np.maximum(x, Tr)  # checks limits
        s = (Ts - Tr) / ((x - Tr) + EPS)
        Psi = -1e-2 / alfa*(s**(1.0 / m) - 1.0)**(1.0 / n)  # m
        Psi[np.isnan(Psi)] = 0.0
        return Psi

    def psi_theta(x):
        # converts water potential [m] to water content [m3 m-3]
        x = 100*np.minimum(x, 0)  # cm
        Th = Tr + (Ts - Tr) / (1 + abs(alfa*x)**n)**m
        return Th

    # --- convert between theta <-- --> psi
    # use explicit None checks so scalars and None defaults are handled safely;
    # y is initialised to None so return y is always bound even if draw_pF=True
    y = None
    if theta is not None:
        y = theta_psi(np.atleast_1d(theta))  # 'Theta-->Psi'
    elif psi is not None:
        y = psi_theta(np.atleast_1d(psi))  # 'Psi-->Theta'

    # draws pf-curve
#    if draw_pF:
#        Ts = Ts[0]; Tr = Tr[0]; alpha = alfa[0]; n = n[0]  
#        xx = -np.logspace(-4, 5, 100)  # cm
#        yy = psi_theta(xx)
#
#        #  field capacity and wilting point
#        fc = psi_theta(-1.0)
#        wp = psi_theta(-150.0)
#
#        fig = plt.figure(99)
#        fig.suptitle('vanGenuchten-Mualem WRC', fontsize=16)
#        ttext = r'$\theta_s=$' + str(Ts) + r', $\theta_r=$' + str(Tr) +\
#                r', $\alpha=$' + str(alfa) + ',n=' + str(n)
#
#        plt.title(ttext, fontsize=14)
#        plt.semilogx(-xx, yy, 'g-')
#        plt.semilogx(1, fc, 'ro', 150, wp, 'ro')  # fc, wp
#        plt.text(1, 1.1*fc, 'FC'), plt.text(150, 1.2*wp, 'WP')
#        plt.ylabel(r'$\theta$  $(m^3m^{-3})$', fontsize=14)
#        plt.xlabel('$\psi$ $(m)$', fontsize=14)
#        plt.ylim(0.8*Tr, min(1, 1.1*Ts))
#
#        del xx, yy
#        y = None
    
    return y
