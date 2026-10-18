"""Compare a completed rerun with all bundled reference tables and PNG figures."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def compare_table(reference: Path, reproduced: Path) -> dict:
    if not reproduced.is_file():
        return {'match': False, 'reason': 'missing output'}
    try:
        a, b = pd.read_csv(reference), pd.read_csv(reproduced)
    except pd.errors.EmptyDataError:
        return {'match': reference.read_bytes().strip() == reproduced.read_bytes().strip(), 'rows': 0}
    if list(a.columns) != list(b.columns) or a.shape != b.shape:
        return {'match': False, 'reason': 'schema/row-count mismatch'}
    differing = []
    for col in a:
        if pd.api.types.is_numeric_dtype(a[col]):
            equal = np.allclose(a[col].to_numpy(dtype=float), b[col].to_numpy(dtype=float),
                                rtol=0, atol=1e-9, equal_nan=True)
        else:
            equal = a[col].fillna('').equals(b[col].fillna(''))
        if not equal:
            differing.append(col)
    return {'match': not differing, 'rows': len(a), 'differing_columns': differing}


def compare_image(reference: Path, reproduced: Path) -> dict:
    if not reproduced.is_file():
        return {'match': False, 'reason': 'missing output'}
    a = np.asarray(Image.open(reference).convert('RGB'))
    b = np.asarray(Image.open(reproduced).convert('RGB'))
    if a.shape != b.shape:
        return {'match': False, 'reason': 'dimensions differ'}
    diff = np.abs(a.astype(int) - b.astype(int))
    changed = int(np.any(diff > 0, axis=2).sum())
    return {'match': changed == 0, 'changed_pixels': changed, 'total_pixels': a.shape[0] * a.shape[1],
            'max_rgb_difference': int(diff.max())}


DOCUMENTED_RASTER_LIMITS = {
    'dpi_200.png': (1, 24),
    'workflow_publisher_raster_export.png': (45, 22),
}


def generated_image_passes(name: str, comparison: dict, allow_documented: bool) -> bool:
    if comparison['match']:
        return True
    if not allow_documented or name not in DOCUMENTED_RASTER_LIMITS:
        return False
    # Missing files and dimension differences have no pixel-difference counts.
    if 'changed_pixels' not in comparison or 'max_rgb_difference' not in comparison:
        return False
    pixels, channel_difference = DOCUMENTED_RASTER_LIMITS[name]
    return (comparison['changed_pixels'] <= pixels
            and comparison['max_rgb_difference'] <= channel_difference)


def verify(run: Path, generated: Path | None = None, allow_documented: bool = False) -> dict:
    report = {'tolerance': 1e-9, 'tables': {}, 'figures': {}}
    for folder in ('data/processed', 'results/tables', 'results/diagnostics', 'results/figures'):
        for ref in sorted((ROOT / folder).glob('*.csv*')):
            rel = ref.relative_to(ROOT)
            report['tables'][rel.as_posix()] = compare_table(ref, run / rel)
    for ref in sorted((ROOT / 'results/figures').glob('*.png')):
        rel = ref.relative_to(ROOT)
        report['figures'][rel.as_posix()] = compare_image(ref, run / rel)
    # This SVG is deterministic and has no timestamp or random element IDs.
    vector_ref = ROOT / 'results/figures/baseline_comparison.svg'
    vector_run = run / vector_ref.relative_to(ROOT)
    report['vector_comparison_match'] = (vector_run.is_file()
                                         and vector_ref.read_bytes() == vector_run.read_bytes())
    scores = pd.read_csv(run / 'results/tables/kmgpt_rmst_error_scores.csv')
    conditions = pd.read_csv(run / 'results/tables/kmgpt_rmst_error_by_condition.csv').set_index('image_id')
    status = pd.read_csv(run / 'data/processed/kmgpt_import_run_status.csv')
    count = status.import_status.value_counts().to_dict()
    report['observations'] = {'runs': len(scores), 'conditions': len(conditions),
                              'scoring': scores.scoring_status.value_counts().to_dict(), 'acceptance': count}
    metric = 'mean_abs_arm_rmst_error_months_mean'
    accepted = conditions[conditions.scored_runs > 0][metric]
    report['headline_months'] = {
        'baseline': round(float(conditions.loc['base_clean_300dpi', metric]), 3),
        'range_min': round(float(accepted.min()), 3), 'range_max': round(float(accepted.max()), 3),
        'heaviest_blur_te': round(float(conditions.loc['blur_1_5', 'rmst_difference_error_months_mean']), 3)}
    report['headline_checks_pass'] = (
        len(scores) == 75 and len(conditions) == 25 and (scores.scoring_status == 'scored').sum() == 39
        and count == {'accepted': 39, 'single_label_two_curves': 24, 'unlabeled_two_curves': 6,
                      'curve_point_count_below_min': 6}
        and report['headline_months'] == {'baseline': .050, 'range_min': .030, 'range_max': .291,
                                         'heaviest_blur_te': .451})
    report['headline_checks_pass'] = bool(report['headline_checks_pass'])
    if generated is not None:
        report['generation_truth'] = {}
        for name in ('base_trial.csv', 'truth_km_curve.csv', 'truth_km_grid.csv', 'truth_risk_table.csv',
                     'truth_summary_by_arm.csv'):
            rel = Path('data/truth') / name
            report['generation_truth'][name] = compare_table(ROOT / rel, generated / rel)
        report['generation_images'] = {ref.name: compare_image(ref, generated / 'data/images' / ref.name)
                                       for ref in sorted((ROOT / 'data/images').glob('*.png'))}
        report['documented_raster_differences_allowed'] = allow_documented
        report['generation_checks_pass'] = (
            all(x['match'] for x in report['generation_truth'].values())
            and all(generated_image_passes(name, result, allow_documented)
                    for name, result in report['generation_images'].items()))
    report['reference_results_match'] = (report['headline_checks_pass']
        and all(x['match'] for x in report['tables'].values())
        and all(x['match'] for x in report['figures'].values())
        and report['vector_comparison_match'])
    report['all_checks_pass'] = (report['reference_results_match']
                                 and report.get('generation_checks_pass', True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('work/reproduced'))
    parser.add_argument('--generated', type=Path)
    parser.add_argument('--allow-documented-raster-differences', action='store_true',
                        help='Allow only the two measured Poppler differences listed in VERIFICATION.md')
    args = parser.parse_args()
    run = (ROOT / args.run).resolve()
    generated = (ROOT / args.generated).resolve() if args.generated else None
    for path in [run] + ([generated] if generated else []):
        if not path.is_relative_to((ROOT / 'work').resolve()):
            parser.error('Verification targets must be inside this package\'s work folder')
    report = verify(run, generated, args.allow_documented_raster_differences)
    destination = run / 'verification.json'
    destination.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Tables:', len(report['tables']), 'matching:', sum(t['match'] for t in report['tables'].values()))
    print('Poster PNGs:', len(report['figures']), 'pixel-identical:', sum(t['match'] for t in report['figures'].values()))
    print('Primary result checks:', report['headline_checks_pass'])
    print('Vector comparison:', report['vector_comparison_match'])
    if generated:
        differences = {k: v for k, v in report['generation_images'].items() if not v['match']}
        print('Synthetic truth tables matching:', sum(t['match'] for t in report['generation_truth'].values()))
        print('Generated image differences:', differences or 'none')
        print('Generation checks:', report['generation_checks_pass'])
    print('Overall checks:', report['all_checks_pass'])
    print('Report:', destination.relative_to(ROOT))
    return 0 if report['all_checks_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
