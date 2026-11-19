import ast
from contextlib import redirect_stdout
from io import StringIO
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

    def test_parameter_plot_labels_first_active_panel_per_row(self):
        for settings, first_active in (
            ({'n_sims_tot': 1, 'n_sims_sbsrf': 1, 'n_sims_ave': 1}, 0),
            ({'n_sims_tot': 0, 'n_sims_ave': 1}, 2),
        ):
            with self.subTest(settings=settings):
                state = self.run_selection(prepare_metrics=True, **settings)
                state['plt'] = plt
                metric_function = ast.parse(self.cells['54bf551f']).body[0]
                exec(compile(ast.Module(body=[metric_function], type_ignores=[]), '<metric>', 'exec'), state)
                try:
                    exec(self.cells['fcef38f9'], state)
                    self.assertEqual([ax.get_title() for ax in state['axes'][0]],
                                     ['Total', 'Subsurface', 'Average'])
                    for ax, label in zip(state['axes'].flat, ('A)', 'B)', 'C)', 'D)', 'E)', 'F)')):
                        self.assertEqual([text.get_text() for text in ax.texts], [label])
                    self.assertTrue(all(not ax.get_xlabel() for ax in state['axes'][0]))
                    self.assertTrue(all(ax.get_xlabel() for ax in state['axes'][1] if ax.axison))
                    for row, expected in ((0, 0), (1, first_active)):
                        axes = state['axes'][row]
                        self.assertEqual([i for i, ax in enumerate(axes) if ax.get_ylabel()], [expected])
                        self.assertTrue(any(label.get_visible() for label in axes[expected].get_yticklabels()))
                    for ax in state['axes'][0]:
                        self.assertEqual(len(ax.collections[0].get_offsets()), len(state['data']))
                    for ax, suffix in zip(state['axes'][1], ('tot', 'sbsrf', 'ave')):
                        selected = state[f'suitable_simulations_{suffix}']
                        self.assertEqual(ax.axison, not selected.empty)
                        if not selected.empty:
                            self.assertEqual(len(ax.collections[0].get_offsets()), len(selected))
                    full_x, selected_x = state['axes'][0, 0].get_xlim(), state['axes'][1, 0].get_xlim()
                    full_y, selected_y = state['axes'][0, 0].get_ylim(), state['axes'][1, 0].get_ylim()
                    self.assertLessEqual(full_x[0], selected_x[0])
                    self.assertGreaterEqual(full_x[1], selected_x[1])
                    self.assertLessEqual(full_y[0], selected_y[0])
                    self.assertGreaterEqual(full_y[1], selected_y[1])
                finally:
                    plt.close('all')

    def test_parameter_colorbars_share_rows_with_different_precisions(self):
        state = self.run_selection(prepare_metrics=True, n_sims_tot=1, n_sims_sbsrf=1, n_sims_ave=1)
        for suffix, value in zip(('tot', 'sbsrf', 'ave'), (0.835, 0.836, 0.837)):
            state['data'][f'kge_{suffix}'] = value
            state[f'suitable_simulations_{suffix}'][f'kge_{suffix}'] = value
        state['plt'] = plt
        metric_function = ast.parse(self.cells['54bf551f']).body[0]
        exec(compile(ast.Module(body=[metric_function], type_ignores=[]), '<metric>', 'exec'), state)
        try:
            exec(self.cells['fcef38f9'], state)
            fig = state['fig']
            fig.canvas.draw()
            colorbars = [ax._colorbar for ax in fig.axes if getattr(ax, '_colorbar', None)]
            self.assertEqual(len(colorbars), 2)
            for row_axes, expected_min, expected_max in (
                (state['axes'][0], 0.8, 0.9),
                (state['axes'][1], 0.83, 0.84),
            ):
                for ax in row_axes:
                    self.assertAlmostEqual(ax.collections[0].norm.vmin, expected_min)
                    self.assertAlmostEqual(ax.collections[0].norm.vmax, expected_max)
            for colorbar, expected_labels in zip(colorbars, (['0.83', '0.84'], ['0.8', '0.9'])):
                labels = [label.get_text() for label in colorbar.ax.get_yticklabels()]
                self.assertEqual(labels, expected_labels)
                self.assertEqual(colorbar.ax.get_ylabel(), 'KGE')
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


class ResolutionComparisonPlotTest(unittest.TestCase):
    def test_resolution_plot_formatting_and_shared_scales(self):
        notebook = json.loads((NOTEBOOK.parent / 'analyze_glue_results_reso.ipynb').read_text())
        cell = next(cell for cell in notebook['cells'] if cell.get('id') == 'reso-parameters')
        source = ''.join(cell['source'])
        for resolutions in ((20,), (20, 40, 80, 160)):
            with self.subTest(resolutions=resolutions):
                frames = {
                    resolution: pd.DataFrame({
                        'kmax_Till': [0.001, 0.002],
                        'f_Till': [6, 8],
                        'kge_tot': [0.835 + index * 0.001, 0.837 + index * 0.001],
                        'kge_sbsrf': [0.825 + index * 0.001, 0.826 + index * 0.001],
                    })
                    for index, resolution in enumerate(resolutions)
                }
                state = {'np': np, 'pd': pd, 'plt': plt, 'soil_type': 'Till',
                         'results_by_resolution': frames}
                try:
                    exec(source, state)
                    fig, axes = state['fig'], state['axes']
                    fig.canvas.draw()
                    self.assertEqual([text.get_text() for text in fig.legends[0].get_texts()],
                                     ['KGE >= 0.825 samples (no contour)'])
                    renderer = fig.canvas.get_renderer()
                    legend_gap = (fig.legends[0].get_window_extent(renderer).y0
                                  - max(ax.title.get_window_extent(renderer).y1 for ax in axes[0]))
                    self.assertGreaterEqual(legend_gap, 0)
                    self.assertLess(legend_gap, 0.15 * fig.dpi)
                    self.assertEqual(axes.shape, (len(resolutions), 3))
                    self.assertEqual([ax.get_title() for ax in axes[0]], ['Total', 'Subsurface', 'Average'])
                    for index, ax in enumerate(axes.flat):
                        labels = [text.get_text() for text in ax.texts]
                        self.assertIn(f'{chr(ord("A") + index)})', labels)
                        self.assertIn(f'{resolutions[index // 3]} m', labels)
                    for row in range(len(resolutions)):
                        self.assertEqual([i for i, ax in enumerate(axes[row]) if ax.get_ylabel()], [0])
                        self.assertTrue(all(bool(ax.get_xlabel()) == (row == len(resolutions) - 1)
                                            for ax in axes[row]))
                        frame = frames[resolutions[row]]
                        expected_ave = frame['kge_tot'] / 2 + frame['kge_sbsrf'] / 2
                        np.testing.assert_allclose(axes[row, 2].collections[0].get_array(), expected_ave)
                        self.assertNotIn('kge_ave', frame.columns)
                    norms = [ax.collections[0].norm for ax in axes.flat]
                    self.assertTrue(all(norm is norms[0] for norm in norms))
                    self.assertAlmostEqual(norms[0].vmin, 0.82)
                    self.assertAlmostEqual(norms[0].vmax, 0.84)
                    colorbars = [ax._colorbar for ax in fig.axes if getattr(ax, '_colorbar', None)]
                    self.assertEqual(len(colorbars), 1)
                    self.assertIs(colorbars[0].mappable.norm, norms[0])
                    self.assertEqual(colorbars[0].ax.get_ylabel(), 'KGE')
                    self.assertEqual([label.get_text() for label in colorbars[0].ax.get_yticklabels()],
                                     ['0.82', '0.83', '0.84'])
                finally:
                    plt.close('all')

    def test_interpolated_threshold_contour_and_fallbacks(self):
        notebook = json.loads((NOTEBOOK.parent / 'analyze_glue_results_reso.ipynb').read_text())
        cell = next(cell for cell in notebook['cells'] if cell.get('id') == 'reso-parameters')
        source = ''.join(cell['source'])
        frames = {
            20: pd.DataFrame({
                'kmax_Till': [0.001, 0.01, 0.001, 10 ** -2.8],
                'f_Till': [2, 2, 10, 3],
                'kge_tot': [0.9, 0.9, 0.9, 0.1],
                'kge_sbsrf': [0.8, 0.8, 0.8, 0.1],
            }),
            40: pd.DataFrame({
                'kmax_Till': [0.001, 0.01, 0.1],
                'f_Till': [2, 4, 6],
                'kge_tot': [0.825, 0.1, 0.1],
                'kge_sbsrf': [0.9, 0.9, 0.9],
            }),
        }
        state = {'np': np, 'plt': plt, 'soil_type': 'Till', 'results_by_resolution': frames}
        output = StringIO()
        contours = []
        original_contour = matplotlib.axes.Axes.contour

        def capture_contour(ax, x, y, z, *args, **kwargs):
            contour = original_contour(ax, x, y, z, *args, **kwargs)
            contours.append((ax, x, y, z.copy(), kwargs, contour))
            return contour

        try:
            with redirect_stdout(output), patch.object(matplotlib.axes.Axes, 'contour', capture_contour):
                exec(source, state)
            axes = state['axes']
            self.assertEqual([text.get_text() for text in state['fig'].legends[0].get_texts()], [
                'KGE >= 0.825 (interpolated boundary)',
                'KGE >= 0.825 samples (no contour)',
            ])
            self.assertEqual(len(contours), 2)
            for expected_ax, (ax, x, y, grid, kwargs, contour) in zip((axes[0, 0], axes[0, 2]), contours):
                self.assertIs(ax, expected_ax)
                self.assertEqual(kwargs['levels'], [0.825])
                self.assertGreater(len(contour.allsegs[0]), 0)
                self.assertGreater(grid[0, 0], 0.825)
                self.assertAlmostEqual(grid[25, 40], 0.1)
                self.assertTrue(np.ma.is_masked(grid[-1, -1]))
                self.assertAlmostEqual(x[25, 40], 10 ** -2.8)
                self.assertEqual(y[25, 40], 3)
            self.assertEqual(len(axes[0, 1].collections), 1)
            self.assertEqual(len(axes[1, 0].collections[1].get_offsets()), 1)
            self.assertEqual(len(axes[1, 1].collections[1].get_offsets()), 3)
            self.assertEqual(len(axes[1, 2].collections[1].get_offsets()), 1)
            self.assertIn('20 m Subsurface: no samples meet', output.getvalue())
            self.assertIn('40 m Total: no threshold contour', output.getvalue())
            self.assertIn('40 m Subsurface: no threshold contour', output.getvalue())
        finally:
            plt.close('all')

        with self.assertRaisesRegex(ValueError, 'kge_threshold must be a finite number'):
            exec(source.replace('kge_threshold = 0.825', "kge_threshold = float('nan')"), state)

        duplicate = frames[20].iloc[[0]].copy()
        duplicate['kge_tot'] = 0.3
        state['results_by_resolution'] = {20: pd.concat([frames[20], duplicate])}
        try:
            with self.assertRaisesRegex(ValueError, 'Conflicting KGE values'):
                exec(source, state)
        finally:
            plt.close('all')


if __name__ == '__main__':
    unittest.main()
