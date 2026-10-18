"""Check evidence completeness, sensitive boundary cases, and reference integrity."""
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "scripts/core")]
import import_results
from runtime import run_root


def csv_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


class EvidenceTests(unittest.TestCase):
    def test_no_excluded_material_is_shareable(self):
        forbidden_dirs = {'protocol', 'protocols', 'decisions', 'archive', 'poster_prep', 'legend_removed',
                          'recheck_dpi_75', 'recheck_dpi_90'}
        for p in ROOT.rglob('*'):
            if not p.is_file() or 'work' in p.relative_to(ROOT).parts:
                continue
            self.assertFalse(forbidden_dirs.intersection(p.relative_to(ROOT).parts), str(p))
            self.assertNotIn(p.suffix.lower(), {'.docx', '.pptx', '.log', '.tmp'})
            self.assertNotIn(p.name, {'PROJECT_HANDOFF.md', '.Rhistory', '.RData'})

    def test_all_conditions_runs_and_failures_are_retained(self):
        images = csv_rows(ROOT / "data/truth/image_manifest.csv")
        runs = csv_rows(ROOT / "data/truth/km_gpt_run_sheet.csv")
        status = csv_rows(ROOT / "data/processed/kmgpt_import_run_status.csv")
        self.assertEqual(len(images), 25)
        self.assertEqual(len(runs), 75)
        self.assertEqual(len({r['run_id'] for r in runs}), 75)
        self.assertEqual({r['run_id'] for r in runs}, {r['run_id'] for r in status})
        from collections import Counter
        self.assertEqual(Counter(r['import_status'] for r in status), {
            'accepted': 39, 'single_label_two_curves': 24,
            'unlabeled_two_curves': 6, 'curve_point_count_below_min': 6})
        for image in images:
            self.assertTrue((ROOT / image['image_path']).is_file())
            self.assertEqual({int(r['replicate_id']) for r in runs if r['image_id'] == image['image_id']}, {1, 2, 3})
        for run in runs:
            self.assertTrue((ROOT / run['image_path']).is_file())

    def test_archives_preserve_raw_result_bytes(self):
        batches = csv_rows(ROOT / "data/extractor/batches.csv")
        self.assertEqual({int(b['replicate_id']) for b in batches}, {1, 2, 3})
        hashes = csv_rows(ROOT / "data/processed/digitised_result_hashes.csv")
        self.assertEqual(len(hashes), 75)
        for batch in batches:
            p = ROOT / 'data/extractor' / batch['zip_file']
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), batch['packaged_archive_sha256'])
            with zipfile.ZipFile(p) as z:
                self.assertTrue(all(n == 'batch-summary.csv' or n.startswith(('results/', 'manifests/', 'annotated/'))
                                    for n in z.namelist()))
                summary = list(csv.DictReader(io.StringIO(z.read('batch-summary.csv').decode('utf-8-sig'))))
                self.assertEqual(len(summary), 25)
                results = [n for n in z.namelist() if n.startswith('results/') and n.endswith('.csv')]
                self.assertEqual(len(results), 25)
                expected = {r['result_sha256'] for r in hashes if r['zip_file'] == p.name}
                self.assertEqual({hashlib.sha256(z.read(n)).hexdigest() for n in results}, expected)

    def test_compressed_points_keep_every_observation(self):
        with gzip.open(ROOT / 'data/processed/kmgpt_curve_points.csv.gz', 'rt', encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 91329)
        self.assertEqual(len({r['run_id'] for r in rows}), 75)
        self.assertEqual(len(rows[0]), 11)

    def test_shape_matching_does_not_hide_label_audit(self):
        rows = csv_rows(ROOT / 'results/diagnostics/label_check.csv')
        wrong = {r['run_id'] for r in rows if r['label_correct'].lower() == 'false'}
        self.assertEqual(wrong, {'workflow_screenshot_slide_rep03'})

    def test_manifest_matches_shareable_files(self):
        rows = csv_rows(ROOT / 'MANIFEST.csv')
        expected = {r['path'] for r in rows}
        actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*')
                  if p.is_file() and p.name != 'MANIFEST.csv'
                  and not {'work', '.git', '__pycache__', '.venv', 'venv'}.intersection(p.relative_to(ROOT).parts)}
        self.assertEqual(actual, expected)
        for row in rows:
            p = ROOT / row['path']
            self.assertEqual(p.stat().st_size, int(row['bytes']), row['path'])
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), row['sha256'], row['path'])


class AcceptanceBoundaryTests(unittest.TestCase):
    @staticmethod
    def points(label, n):
        return [{'time_months': i / 10, 'survival': 0.9, 'raw_curve_label': label,
                 'manifest_annotation_label': ''} for i in range(n)]

    def status(self, control='Control', treatment='Treatment', n=150):
        return import_results.classify_curve_groups({
            'curve_1': self.points(control, n), 'curve_2': self.points(treatment, 150)}, 150)[0]

    def test_threshold_and_failure_precedence(self):
        self.assertEqual(self.status(), 'accepted')
        self.assertEqual(self.status(n=149), 'curve_point_count_below_min')
        self.assertEqual(self.status(treatment='Control', n=149), 'single_label_two_curves')
        self.assertEqual(self.status(control='', n=149), 'unlabeled_two_curves')
        self.assertEqual(import_results.classify_curve_groups({}, 150)[0], 'geometric_curve_count_not_two')

    def test_outputs_cannot_overwrite_reference_or_parent(self):
        old = os.environ.get('ISPOR_RUN_ROOT')
        try:
            for forbidden in [ROOT, ROOT.parent, ROOT / 'results', ROOT / 'work']:
                os.environ['ISPOR_RUN_ROOT'] = str(forbidden)
                with self.assertRaises(ValueError):
                    run_root()
            os.environ['ISPOR_RUN_ROOT'] = str(ROOT / 'work/test-run')
            self.assertEqual(run_root(), (ROOT / 'work/test-run').resolve())
        finally:
            if old is None:
                os.environ.pop('ISPOR_RUN_ROOT', None)
            else:
                os.environ['ISPOR_RUN_ROOT'] = old


if __name__ == '__main__':
    unittest.main()
