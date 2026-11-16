import unittest

import numpy as np

from iotools import preprocess_dsdata_vec
from parameters_krycklan_soil import deep_properties
from soilprofile2D import SoilGrid_2Dflow, gwl_Wsto, gwl_Wsto_vectorized, wrc


def direct_root_moisture(properties, gwl):
    midpoints = -np.arange(0.005, 0.2, 0.01)
    layers = (midpoints < -0.1).astype(int)
    pf = {key: np.asarray(values)[layers] for key, values in properties['pF'].items()}
    return np.mean(wrc(pf, psi=gwl - midpoints))


class RootMoistureLookupTest(unittest.TestCase):
    def test_builders_cover_reference_and_match_retention_curve(self):
        properties = deep_properties()
        sand = properties['Sand']
        till = properties['Till']
        z = sand['deep_z'][:2]
        sand_pf = {key: values[:2] for key, values in sand['pF'].items()}
        till_pf = {key: values[:2] for key, values in till['pF'].items()}

        builders = (
            ('scalar', lambda: [gwl_Wsto(z, sand_pf, root=True)['to_rootmoist']]),
            ('vectorized', lambda: gwl_Wsto_vectorized(
                [z, z], [sand_pf, till_pf], root=True)['to_rootmoist']),
            ('variable spacing', lambda: gwl_Wsto_vectorized(
                z, sand_pf, grid_step='var', root=True)['to_rootmoist']),
        )
        for name, build in builders:
            with self.subTest(builder=name):
                functions = build()
                for function, profile in zip(functions, (sand, till)):
                    self.assertLessEqual(function.x[0], -150.1)
                    for gwl in (-1.3, -15.037, -150.1):
                        with self.subTest(gwl=gwl):
                            self.assertAlmostEqual(
                                float(function(gwl)), direct_root_moisture(profile, gwl),
                                delta=1e-6)

    def test_preprocessing_covers_custom_reference_and_cell_floors(self):
        profiles = deep_properties()
        profiles = {'Sand': profiles['Sand'], 'Till': profiles['Till']}
        shape = (2, 2)
        gis = {
            'deep_id': np.array([[5.0, 4.0], [5.0, 4.0]]),
            'deep_z': np.array([[15.0, 200.0], [30.0, 5.0]]),
            'elevation': np.full(shape, 245.0),
            'streams': np.zeros(shape),
            'lakes': np.zeros(shape),
            'dxy': 160.0,
        }
        pspd = {'deep_z': 0.0, 'ground_water_level': -15.0, 'lake_depth': -1.0}
        data = preprocess_dsdata_vec(
            pspd, {'deep_z': True}, profiles, gis, root_min_gwl=-180.0)
        data['rew_gwl_wp'] = -180.0
        model = SoilGrid_2Dflow(data)

        for index, name, minimum in (
                ((0, 0), 'Sand', -180.0), ((0, 1), 'Till', -200.0),
                ((1, 0), 'Sand', -180.0), ((1, 1), 'Till', -200.0)):
            with self.subTest(cell=index):
                function = data['gwl_to_rootmoist'][index]
                self.assertLessEqual(function.x[0], minimum)
                self.assertAlmostEqual(
                    model.root_wp[index], direct_root_moisture(profiles[name], -180.0),
                    delta=1e-6)
                self.assertAlmostEqual(
                    float(function(minimum)), direct_root_moisture(profiles[name], minimum),
                    delta=1e-6)

    def test_default_depth_produces_water_stress_at_deep_water_table(self):
        sand = deep_properties()['Sand']
        shape = (2, 2)
        gis = {
            'deep_id': np.full(shape, 5.0),
            'elevation': np.full(shape, 245.0),
            'streams': np.zeros(shape),
            'lakes': np.zeros(shape),
            'dxy': 160.0,
        }
        pspd = {'deep_z': 15.0, 'ground_water_level': -15.0, 'lake_depth': -1.0}
        data = preprocess_dsdata_vec(
            pspd, {'deep_z': False}, {'Sand': sand}, gis, root_min_gwl=-160.0)
        data['rew_gwl_wp'] = -160.0
        model = SoilGrid_2Dflow(data)
        results = model.run_timestep(1.0, np.zeros(shape), np.zeros(shape))

        wp = direct_root_moisture(sand, -160.0)
        fc1 = direct_root_moisture(sand, -1.3)
        moisture = direct_root_moisture(sand, -15.0)
        expected_rew = 0.5 * (moisture - wp) / (fc1 - wp)
        np.testing.assert_allclose(results['transpiration_limitation'], expected_rew, atol=1e-6)
        self.assertGreater(expected_rew, 0.0)
        self.assertLess(expected_rew, 0.2)


if __name__ == '__main__':
    unittest.main()
