"""Regression checks for fresh destinations and generation failure reporting."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'tests')]
import verify_reproduction

spec = importlib.util.spec_from_file_location('study_runner', ROOT / 'run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def test_fresh_figure_destination_has_temp_before_preflight(self):
        (ROOT / 'work').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / 'work') as directory:
            destination = Path(directory) / 'fresh-figures'

            def preflight(mode, env):
                self.assertEqual(mode, 'figures')
                self.assertTrue(Path(env['TEMP']).is_dir())
                self.assertEqual(env['TEMP'], env['TMP'])
                raise RuntimeError('preflight reached with usable temp')

            with mock.patch.dict(os.environ), mock.patch.object(sys, 'argv', [
                    'run.py', 'figures', '--output-dir', str(destination)]), \
                    mock.patch.object(runner, 'check_dependencies', side_effect=preflight):
                with self.assertRaisesRegex(RuntimeError, 'preflight reached with usable temp'):
                    runner.main()

    def test_missing_generated_outputs_fail_overall_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            report = verify_reproduction.verify(ROOT, Path(directory) / 'missing')
        self.assertTrue(report['reference_results_match'])
        self.assertFalse(report['generation_checks_pass'])
        self.assertFalse(report['all_checks_pass'])
        self.assertEqual(sum(not x['match'] for x in report['generation_truth'].values()), 5)
        self.assertEqual(sum(not x['match'] for x in report['generation_images'].values()), 25)

    def test_documented_raster_allowance_is_explicit_and_bounded(self):
        observed = {'match': False, 'changed_pixels': 1, 'max_rgb_difference': 24}
        check = verify_reproduction.generated_image_passes
        self.assertFalse(check('dpi_200.png', observed, False))
        self.assertTrue(check('dpi_200.png', observed, True))
        self.assertFalse(check('base_clean_300dpi.png', observed, True))
        self.assertFalse(check('dpi_200.png', {'match': False, 'reason': 'missing output'}, True))
        self.assertFalse(check('dpi_200.png', {**observed, 'changed_pixels': 2}, True))
        self.assertFalse(check('dpi_200.png', {**observed, 'max_rgb_difference': 25}, True))


if __name__ == '__main__':
    unittest.main()
