import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NOTEBOOK = Path(__file__).resolve().parents[1] / 'analyze_glue_results_sub.ipynb'


class GlueNotebookSelectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = json.loads(NOTEBOOK.read_text())
        cls.cells = {cell.get('id'): ''.join(cell['source']) for cell in notebook['cells']}
        cls.selection_code = cls.cells['17ad2d94']

    def run_selection(self, prepare_metrics=False, **settings):
        source = self.selection_code
        for name, value in settings.items():
            lines = source.splitlines(keepends=True)
            index = next(i for i, line in enumerate(lines) if line.startswith(f'{name} = '))
            lines[index] = f'{name} = {value!r}\n'
            source = ''.join(lines)
        data = pd.DataFrame({
            'sample_id': ['a', 'b', 'c', 'd'],
            'kmax_Till': [0.001, 0.002, 0.003, 0.004],
            'kmin_Till': [1e-9] * 4,
            'f_Till': [6, 8, 10, 12],
            'kge_tot': [0.8, 0.9, 0.8, 0.1],
            'kge_sbsrf': [0.4, 0.2, 0.7, 0.7],
            'rmse_tot': [1, 2, 3, 4],
            'rmse_sbsrf': [3, 2, 1, 4],
            'mbe_tot': [2, -4, 1, 3],
            'mbe_sbsrf': [-4, 0, 3, -1],
        })
        namespace = {'data': data, 'np': np, 'pd': pd, 'os': os, 'Path': Path, 'nan': float('nan')}
        if prepare_metrics:
            with patch.dict(os.environ, {'SPAFHY_RESULTS_FOLDER': '/test'}), \
                    patch.object(pd, 'read_csv', return_value=data):
                exec(self.cells['562fe44b'], namespace)
            exec(self.cells['779be2f1'], namespace)
            exec(self.cells['c6e52214'], namespace)
        else:
            data['kge_ave'] = data['kge_tot'] / 2 + data['kge_sbsrf'] / 2
            namespace['data'] = data.set_index('sample_id', drop=False)
        exec(source, namespace)
        return namespace

    def test_count_selection_preserves_order_at_ties_and_skips_component(self):
        state = self.run_selection(n_sims_tot=2)
        self.assertEqual(list(state['suitable_simulations_tot'].index), ['b', 'a'])
        self.assertEqual(state['threshold_tot'], 0.8)
        self.assertEqual(state['n_sims_sbsrf'], 0)
        self.assertIsNone(state['threshold_sbsrf'])

    def test_cutoffs_include_boundary_ties_and_override_counts(self):
        state = self.run_selection(kge_cutoff_tot=0.8, kge_cutoff_sbsrf=0.7)
        self.assertEqual(list(state['suitable_simulations_tot'].index), ['b', 'a', 'c'])
        self.assertEqual(list(state['suitable_simulations_sbsrf'].index), ['c', 'd'])
        self.assertEqual((state['n_sims_tot'], state['n_sims_sbsrf']), (3, 2))
        self.assertEqual((state['threshold_tot'], state['threshold_sbsrf']), (0.8, 0.7))

    def test_cutoff_and_count_can_be_used_together(self):
        state = self.run_selection(n_sims_tot=2, kge_cutoff_sbsrf=0.4)
        self.assertEqual(list(state['suitable_simulations_tot'].index), ['b', 'a'])
        self.assertEqual(list(state['suitable_simulations_sbsrf'].index), ['c', 'd', 'a'])

    def test_average_metrics_are_ranked_and_selected_by_count(self):
        state = self.run_selection(prepare_metrics=True, n_sims_tot=0, n_sims_ave=2)
        data = state['data']
        self.assertAlmostEqual(data.loc['a', 'kge_ave'], 0.6)
        self.assertEqual(data.loc['a', 'rmse_ave'], 2)
        self.assertEqual(data.loc['a', 'mbe_ave'], -1)
        self.assertEqual(data.loc['c', 'rank_kge_ave'], 1)
        self.assertEqual(data.loc['c', 'rank_ave'], 1)
        self.assertEqual(list(state['suitable_simulations_ave'].index), ['c', 'a'])
        self.assertEqual(state['n_sims_ave'], 2)

    def test_average_cutoff_includes_boundary_and_can_be_used_alone(self):
        state = self.run_selection(n_sims_tot=0, kge_cutoff_ave=0.6)
        self.assertEqual(list(state['suitable_simulations_ave'].index), ['c', 'a'])
        self.assertEqual(state['threshold_ave'], 0.6)
        self.assertEqual(state['n_sims_ave'], 2)

    def test_average_only_plots_and_skips_time_series(self):
        state = self.run_selection(prepare_metrics=True, n_sims_tot=0, n_sims_ave=2)

        class FakeSeaborn:
            @staticmethod
            def histplot(x, bins, kde, ax):
                ax.hist(x, bins=bins)

        state.update(plt=plt, sns=FakeSeaborn())
        try:
            for cell_id in ('54bf551f', '9b17ab98', 'fcef38f9', 'ff3dd454'):
                exec(self.cells[cell_id], state)
            self.assertEqual(len(state['axes'][2].lines), 2)
            figures_before = plt.get_fignums()
            with patch.dict(os.environ, {'PROJECT_FOLDER': ''}):
                for cell_id in ('3fa2ef20', '4f7ff5a3', 'a981d6f4', 'f8ae12c9', '1c82b953'):
                    exec(self.cells[cell_id], state)
            self.assertEqual(plt.get_fignums(), figures_before)
        finally:
            plt.close('all')

    def test_existing_time_series_reruns_need_only_original_metrics(self):
        state = self.run_selection(prepare_metrics=True)
        params = {name: 0.5 for name in state['base_metrics']}
        result = (pd.DataFrame(), params, None)
        state.update(
            results_tot={'b': result}, results_sbsrf={},
            model_root=Path('/test'), catchment='krycklan', catchment_no=2,
            q_obs={'Q': pd.Series(dtype=float)},
            run_single_simulation=lambda **kwargs: result,
        )
        exec(self.cells['f8ae12c9'], state)
        self.assertEqual(state['best_idx_tot'], 'b')
        self.assertEqual(set(state['ranges_tot']), set(state['base_metrics']))

    def test_invalid_or_empty_selections_fail_explicitly(self):
        for settings in (
            {'kge_cutoff_tot': float('nan')},
            {'kge_cutoff_tot': True},
            {'kge_cutoff_tot': 1.0},
            {'n_sims_tot': -1},
            {'n_sims_tot': 5},
            {'n_sims_tot': 0},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                self.run_selection(**settings)


if __name__ == '__main__':
    unittest.main()
