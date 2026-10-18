"""Export the original baseline at 4.72 by 5.69 inches with its upper-right legend."""
import hashlib
import json
from xml.etree import ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import yaml
import full_baseline
from _common import FOLDER, ROOT, TRUTH

WIDTH, HEIGHT, DPI = 4.72, 5.69, 300
STEM = FOLDER / 'figures/baseline_original_4p72x5p69in'


def main():
    sources = [ROOT / 'config/study.yaml', TRUTH / 'base_trial.csv', TRUTH / 'truth_km_curve.csv']
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    cfg = yaml.safe_load(sources[0].read_text(encoding='utf-8'))
    trial, km = pd.read_csv(sources[1]), pd.read_csv(sources[2])
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 19,
                         'xtick.labelsize': 19, 'ytick.labelsize': 19,
                         'axes.labelsize': 21, 'legend.fontsize': 19,
                         'svg.fonttype': 'none', 'savefig.bbox': None}):
        # Original truth curves and original censor marks only: no extracted points,
        # no enlarged censor marks, no zoom box, and no auxiliary bottom key.
        fig, ax = full_baseline.draw_plot(cfg, trial, km)
        fig.set_size_inches(WIDTH, HEIGHT, forward=True)
        fig.set_dpi(DPI)
        ax.set_position([0.22, 0.16, 0.74, 0.81])
        ax.set_xlabel('Months', fontsize=21, labelpad=8)
        ax.set_ylabel('Overall survival', fontsize=21, labelpad=8)
        ax.tick_params(axis='both', labelsize=19, pad=4)
        ax.legend(loc='upper right', frameon=False, fontsize=19,
                  handlelength=1.5, handletextpad=0.45, borderpad=0.25)
        fig.canvas.draw()
        bounds = ax.get_tightbbox(fig.canvas.get_renderer())
        assert bounds.x0 >= 0 and bounds.y0 >= 0
        assert bounds.x1 <= WIDTH*DPI and bounds.y1 <= HEIGHT*DPI
        assert not ax.collections and not ax.patches and not fig.legends
        assert [text.get_text() for text in ax.get_legend().get_texts()] == ['Control','Treatment']
        fig.savefig(STEM.with_suffix('.svg'), bbox_inches=None, facecolor='white')
        fig.savefig(STEM.with_suffix('.png'), dpi=DPI, bbox_inches=None, facecolor='white')
        with Image.open(STEM.with_suffix('.png')) as image:
            assert image.size == (1416,1707)
        svg = ET.parse(STEM.with_suffix('.svg')).getroot()
        assert svg.attrib['width'] == '339.84pt' and svg.attrib['height'] == '409.68pt'
        text = ' '.join(svg.itertext())
        assert 'Digitised' not in text and 'Censored' not in text
        plt.close(fig)
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p,digest in hashes.items())
    report = {'inches':[WIDTH,HEIGHT], 'png_pixels':[1416,1707], 'dpi':DPI,
              'legend':'Control and Treatment, upper right inside axes',
              'x_ticks':cfg['render']['x_ticks'], 'tick_and_legend_font_pt':19,
              'axis_label_font_pt':21, 'digitised_overlay':False, 'bottom_key':False,
              'zoom_box':False, 'original_censor_marks':True}
    STEM.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
