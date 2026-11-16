import unittest
from unittest import mock

import numpy as np

from soilprofile2D import SoilGrid_2Dflow


def storage(gwl):
    return 3.5 + 0.2 * np.asarray(gwl)


def transmissivity(gwl):
    return np.maximum(1e-5, 5.0 * (np.asarray(gwl) + 15.0))


def capacity(gwl):
    return np.zeros_like(gwl, dtype=float) + 0.2


def rootmoist(gwl):
    return 0.3 + 0.001 * np.asarray(gwl)


def make_model(gwl, elevation, streams, lakes=None, cellwise=False):
    shape = gwl.shape
    soiltype = np.full(shape, 'soil', dtype=object)
    if cellwise:
        def lookup(function):
            return np.full(shape, function, dtype=object)
    else:
        def lookup(function):
            return {'soil': function}

    return SoilGrid_2Dflow({
        'soiltype': soiltype,
        'deep_id': np.ones(shape),
        'elevation': elevation.copy(),
        'streams': streams.copy(),
        'lakes': np.zeros(shape) if lakes is None else lakes.copy(),
        'deep_z': np.full(shape, 15.0),
        'ground_water_level': gwl.copy(),
        'dxy': 160.0,
        'wtso_to_gwl': lookup(lambda w: (w - 3.5) / 0.2),
        'gwl_to_wsto': lookup(storage),
        'gwl_to_Tr': lookup(transmissivity),
        'gwl_to_C': lookup(capacity),
        'gwl_to_rootmoist': lookup(rootmoist),
        'ditch_boundary': 'Cauchy',
        'stream_length': np.full(shape, 160.0),
        'stream_distance': np.full(shape, 5.0),
    })


class CauchyDitchTransmissivityTest(unittest.TestCase):
    def test_floor_uses_cell_transmissivity_and_conserves_water(self):
        elevation = np.full((3, 3), 245.81)
        elevation[1, 0] = 258.42
        elevation[1, 2] = 228.58
        gwl = np.full((3, 3), -15.0)
        gwl[1, 2] = 0.0
        streams = np.zeros((3, 3))
        streams[1, 1] = -0.333

        for cellwise in (False, True):
            with self.subTest(cellwise=cellwise):
                model = make_model(gwl, elevation, streams, cellwise=cellwise)
                with mock.patch.object(model, '_solve_step', wraps=model._solve_step) as solve:
                    results = model.run_timestep(1.0, np.zeros((3, 3)), np.zeros((3, 3)))

                self.assertEqual(solve.call_count, 1)
                self.assertEqual(model.conv99, 99)
                self.assertAlmostEqual(model.Tr0[1, 1], 1e-5)
                self.assertAlmostEqual(model.Tr1[1, 1], 1e-5, places=7)
                self.assertLess(model.TrE0[1, 1], 2.1e-5)
                self.assertLess(model.TrE1[1, 1], 2.1e-5)
                self.assertEqual(results['netflow_to_ditch'][1, 1], 0.0)
                self.assertGreaterEqual(results['ground_water_level'][1, 1], -15.0)
                self.assertLess(abs(results['water_closure'][1, 1]), 1e-3)
                self.assertLess(abs(np.sum(results['water_closure'])), 1e-3)
                self.assertAlmostEqual(np.sum(results['lateral_netflow']), 0.0, places=8)

    def test_at_or_below_ditch_level_uses_cell_transmissivity(self):
        elevation = np.full((3, 3), 245.81)
        streams = np.zeros((3, 3))
        streams[1, 1] = -0.333
        for center_gwl in (-1.0, -0.333):
            with self.subTest(center_gwl=center_gwl):
                gwl = np.full((3, 3), -0.1)
                gwl[1, 1] = center_gwl
                model = make_model(gwl, elevation, streams)
                model._solve_step(0.001, np.zeros((3, 3)), np.zeros((3, 3)), 1e-3, 50)

                self.assertAlmostEqual(model.Tr0[1, 1], transmissivity(center_gwl))
                self.assertAlmostEqual(model.Tr1[1, 1], transmissivity(center_gwl))

    def test_active_ditch_keeps_neighbour_mean_and_drains(self):
        elevation = np.full((3, 3), 245.81)
        gwl = np.full((3, 3), -0.1)
        gwl[1, 1] = -0.2
        streams = np.zeros((3, 3))
        streams[1, 1] = -0.333
        model = make_model(gwl, elevation, streams)

        converged, results = model._solve_step(
            1.0, np.zeros((3, 3)), np.zeros((3, 3)), 1e-3, 50)

        self.assertTrue(converged)
        self.assertGreater(results['ground_water_level'][1, 1], streams[1, 1])
        self.assertAlmostEqual(model.Tr0[1, 1], transmissivity(-0.1))
        self.assertAlmostEqual(model.Tr1[1, 1], np.mean([
            model.Tr1[0, 1], model.Tr1[2, 1], model.Tr1[1, 0], model.Tr1[1, 2]
        ]))
        self.assertGreater(results['netflow_to_ditch'][1, 1], 0.0)

    def test_transmissivity_switches_when_head_crosses_ditch_level(self):
        elevation = np.full((3, 3), 245.81)
        elevation[1, 2] = 228.58
        streams = np.zeros((3, 3))
        streams[1, 1] = -0.333
        for start, recharge in ((-0.2, 0.0), (-0.5, 0.1)):
            with self.subTest(start=start):
                gwl = np.full((3, 3), -0.1)
                gwl[1, 1] = start
                model = make_model(gwl, elevation, streams)
                rr = np.zeros((3, 3))
                rr[1, 1] = recharge
                converged, results = model._solve_step(
                    1.0, rr, np.zeros((3, 3)), 1e-3, 50)

                self.assertTrue(converged)
                end = results['ground_water_level'][1, 1]
                if start > streams[1, 1]:
                    self.assertLess(end, streams[1, 1])
                    self.assertAlmostEqual(model.Tr0[1, 1], transmissivity(-0.1))
                    self.assertAlmostEqual(model.Tr1[1, 1], transmissivity(end), places=3)
                else:
                    self.assertGreater(end, streams[1, 1])
                    self.assertAlmostEqual(model.Tr0[1, 1], transmissivity(start))
                    self.assertAlmostEqual(model.Tr1[1, 1], np.mean([
                        model.Tr1[0, 1], model.Tr1[2, 1], model.Tr1[1, 0], model.Tr1[1, 2]
                    ]))
                self.assertLess(abs(results['water_closure'][1, 1]), 1e-3)

    def test_lake_transmissivity_still_uses_neighbour_mean(self):
        elevation = np.full((3, 3), 245.81)
        gwl = np.full((3, 3), -0.1)
        gwl[1, 1] = -15.0
        lakes = np.zeros((3, 3))
        lakes[1, 1] = -0.333
        model = make_model(gwl, elevation, np.zeros((3, 3)), lakes=lakes)

        model._solve_step(1.0, np.zeros((3, 3)), np.zeros((3, 3)), 1e-3, 50)

        self.assertAlmostEqual(model.Tr0[1, 1], transmissivity(-0.1))
        self.assertAlmostEqual(model.Tr1[1, 1], np.mean([
            model.Tr1[0, 1], model.Tr1[2, 1], model.Tr1[1, 0], model.Tr1[1, 2]
        ]))


if __name__ == '__main__':
    unittest.main()
