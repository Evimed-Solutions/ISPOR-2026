"""Export the four-condition comparison with native SVG pixels, traces, arrows and text."""
from collections import defaultdict
import json
import math
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
from PIL import Image, ImageFont
import baseline_comparison as comparison
from _common import FOLDER, IMAGES, OUT, RAW, WORK

base = comparison.base
SVG = 'http://www.w3.org/2000/svg'
ET.register_namespace('', SVG)


def element(parent, name, **attributes):
    return ET.SubElement(parent, '{' + SVG + '}' + name,
                         {k.replace('_', '-'): str(v) for k, v in attributes.items()})


def f(value):
    return f'{value:.4f}'.rstrip('0').rstrip('.')


def clipped_trace(coordinates, width, height):
    """Clip line segments explicitly, including for SVG importers that ignore clipPath."""
    output = []
    previous = None
    for (ax, ay), (bx, by) in zip(coordinates[:-1], coordinates[1:]):
        dx, dy = bx-ax, by-ay
        start, end, keep = 0.0, 1.0, True
        for p, q in ((-dx, ax), (dx, width-ax), (-dy, ay), (dy, height-ay)):
            if p == 0:
                if q < 0:
                    keep = False
                    break
            elif p < 0:
                start = max(start, q/p)
            else:
                end = min(end, q/p)
            if start > end:
                keep = False
                break
        if not keep:
            previous = None
            continue
        a, b = (ax+start*dx, ay+start*dy), (ax+end*dx, ay+end*dy)
        if previous is None or abs(previous[0]-a[0])+abs(previous[1]-a[1]) > 1e-6:
            output.append(f'M{f(a[0])} {f(a[1])}')
        output.append(f'L{f(b[0])} {f(b[1])}')
        previous = b
    return ''.join(output)


def svg_pixels(parent, array, x, y, scale):
    """Run-length encode the actual coloured pixels as vector rectangles, grouped by colour."""
    paths = defaultdict(list)
    for row, values in enumerate(array):
        transitions = np.r_[0, np.where(np.any(values[1:] != values[:-1], axis=1))[0] + 1, len(values)]
        for left, right in zip(transitions[:-1], transitions[1:]):
            color = tuple(int(v) for v in values[left])
            if color == (255, 255, 255):
                continue
            # Tiny overlap prevents antialiasing seams in applications that ignore crispEdges.
            paths[color].append(f'M{f(left-.005)} {f(row-.005)}h{f(right-left+.01)}v1.01h{f(left-right-.01)}z')
    group = element(parent, 'g', transform=f'translate({x} {y}) scale({scale})', shape_rendering='crispEdges')
    for color, segments in paths.items():
        element(group, 'path', d=''.join(segments), fill='#' + ''.join(f'{c:02x}' for c in color))


def draw_tile(parent, definitions, src, frame, scale, position, tile_id, points, callouts):
    x0, y0, x1, y1 = comparison.zoom_box(frame)
    width, height = (x1-x0)*scale, (y1-y0)*scale
    tx, ty = position
    clipped = element(parent, 'g', clip_path=f'url(#{tile_id})')
    clip = element(definitions, 'clipPath', id=tile_id, clipPathUnits='userSpaceOnUse')
    element(clip, 'rect', x=tx, y=ty, width=width, height=height)
    svg_pixels(clipped, src[y0:y1, x0:x1], tx, ty, scale)
    overlay = element(clipped, 'g', transform=f'translate({tx} {ty})')
    to = lambda t, s: ((base.crops.to_px(frame, t=t)-x0+0.5)*scale,
                       (base.crops.to_px(frame, s=s)-y0+0.5)*scale)
    radius = 1.5 * comparison.MM
    if points is not None:
        for arm, group in base.traces(points):
            coordinates = [to(t, s) for t, s in zip(group.time_months, group.survival)]
            path = clipped_trace(coordinates, width, height)
            for color, line_width in ((base.INK, 2.0), (base.WHITE, 1.5), (comparison.TRACE_COLOUR.get(arm, base.INK), .85)):
                element(overlay, 'path', d=path, fill='none', stroke=color,
                        stroke_width=f(round(line_width*comparison.MM*3)/3),
                        stroke_linejoin='round', stroke_linecap='round')
        for arm, group in base.traces(points):
            for t, s in zip(group.time_months, group.survival):
                px, py = to(t, s)
                if -radius <= px <= width+radius and -radius <= py <= height+radius:
                    for color, r in ((base.INK, radius), (base.WHITE, radius-.25*comparison.MM),
                                     (comparison.TRACE_COLOUR.get(arm, base.INK), radius-.6*comparison.MM)):
                        element(overlay, 'circle', cx=f(px), cy=f(py), r=f(r), fill=color)
    font = ImageFont.truetype(base.FONT_DIR + 'arialbd.ttf', round(16/72*300)*3)
    font_size, ascent = font.size/3, font.getmetrics()[0]/3
    head = 2.6*comparison.MM
    for text, (fx, fy), targets in callouts:
        lines = text.split('\n')
        padx, pady, margin = 1.2*comparison.MM, .7*comparison.MM, 1.5*comparison.MM
        bw = max(font.getlength(line)/3 for line in lines) + 2*padx
        bh = len(lines)*font_size*1.2 + 2*pady
        bx, by = min(fx*width, width-bw-margin), min(fy*height, height-bh-margin)
        for who, when in targets:
            targetx, targety = to(*comparison.target_point(points, who, when))
            sx = min(max(targetx, bx), bx+bw)
            sy = by+bh if targety > by+bh else by if targety < by else by+bh/2
            angle = math.atan2(targety-sy, targetx-sx)
            ex, ey = targetx-math.cos(angle)*1.6*radius, targety-math.sin(angle)*1.6*radius
            hx, hy = ex-math.cos(angle)*head, ey-math.sin(angle)*head
            wing = [(hx-math.sin(angle)*head*.5, hy+math.cos(angle)*head*.5),
                    (hx+math.sin(angle)*head*.5, hy-math.cos(angle)*head*.5)]
            for color, line_width, grow in ((base.WHITE, 1.4, .5*comparison.MM), (base.INK, .55, 0)):
                element(overlay, 'path', d=f'M{f(sx)} {f(sy)}L{f(hx)} {f(hy)}', stroke=color,
                        stroke_width=f(round(line_width*comparison.MM*3)/3), fill='none')
                vertices = [(ex+math.cos(angle)*grow, ey+math.sin(angle)*grow)] + [
                    (wx-math.cos(angle)*grow, wy-math.sin(angle)*grow) for wx, wy in wing]
                element(overlay, 'polygon', points=' '.join(f'{f(x)},{f(y)}' for x, y in vertices), fill=color)
        element(overlay, 'rect', x=f(bx), y=f(by), width=f(bw), height=f(bh), rx=f(.8*comparison.MM),
                fill='white', fill_opacity=f(235/255), stroke=base.INK,
                stroke_width=f(round(.25*comparison.MM*3)/3))
        for n, line in enumerate(lines):
            node = element(overlay, 'text', x=f(bx+padx), y=f(by+pady+n*font_size*1.2+ascent),
                           fill=base.INK, font_family='Arial, Helvetica, sans-serif',
                           font_weight='bold', font_size=f(font_size))
            node.text = line
    element(parent, 'rect', x=tx-3, y=ty-3, width=width+5, height=height+5,
            fill='none', stroke=comparison.FRAME, stroke_width=3)


def main():
    source = WORK / 'pixels'
    for name in comparison.ROWS:
        if not (source / (name + '_pixels.csv')).is_file():
            raise FileNotFoundError('Reproduce clustering first using: python run.py figures')
    points = pd.read_csv(RAW / 'kmgpt_curve_points.csv.gz', usecols=lambda c: c != 'notes')
    quality = pd.read_csv(OUT / 'image_quality_metrics.csv').set_index('image_id')
    rows = []
    for name in comparison.ROWS:
        arr = np.asarray(Image.open(IMAGES / f'{name}.png').convert('RGB'))
        reference = quality.loc[name, 'compared_with']
        clean = arr if pd.isna(reference) else np.asarray(Image.open(IMAGES / f'{reference}.png').convert('RGB'))
        frame = base.crops.find_frame(arr)
        x0, y0, x1, y1 = comparison.zoom_box(frame)
        if not rows:
            reference_width = (x1-x0)*comparison.TILE_ZOOM
        scale = max(1, round(reference_width/(x1-x0)))
        pts = points[points.run_id == name+'_rep01']
        groups, stats = base.groups_rgb(base.arm_groups(name, arr.shape[:2]), clean)
        if pd.isna(reference):
            notes = [('Clean image, zoomed\n(no pixel change)', (.5, .04), [])]
        elif name == 'jpeg_95':
            notes = [(comparison.BLOCK_NOTE, (.42,.03), [('at',comparison.block_target(arr,clean,frame))])]
        else:
            notes = []
        rows.append({'name':name,'frame':frame,'scale':scale,'size':(x1-x0)*scale,'points':pts,
                     'left':comparison.change_over_curves(arr,clean,frame),'right':groups,'notes':notes})
    pad, gap, row_gap = [round(n*comparison.MM) for n in (5,4,6)]
    slot = max(row['size'] for row in rows)
    width = 2*pad+2*slot+gap
    height = 2*pad+sum(row['size'] for row in rows)+row_gap*(len(rows)-1)
    svg = ET.Element('{'+SVG+'}svg', {'width':f(width/300*25.4)+'mm',
                                    'height':f(height/300*25.4)+'mm','viewBox':f'0 0 {width} {height}'})
    title = element(svg, 'title'); title.text = 'Baseline, JPEG 95, blur sigma 1.5 and screenshot: pixel change and digitisation'
    description = element(svg, 'desc'); description.text = 'Left: image pixels and changes. Right: algorithmic arm groups and observed replicate 1 traces. Native vector pixels, trace paths, markers, arrows and editable text.'
    definitions = element(svg, 'defs')
    element(svg,'rect',x=0,y=0,width=width,height=height,fill='white')
    y = pad
    for i,row in enumerate(rows):
        edges = []
        for col, side in enumerate(['left','right']):
            x = pad+col*(slot+gap)+(slot-row['size'])//2
            edges.append((x, x+row['size']))
            draw_tile(svg,definitions,row[side],row['frame'],row['scale'],(x,y),f'row{i}-{side}',
                      row['points'] if col else row['points'].iloc[0:0],
                      comparison.CALLOUTS.get(row['name'],()) if col else row['notes'])
        # Explicit gutter masks also keep clipped markers inside their panels in SVG importers.
        for left,right in [(0,edges[0][0]),(edges[0][1],edges[1][0]),(edges[1][1],width)]:
            element(svg,'rect',x=left,y=y,width=right-left,height=row['size'],fill='white')
        element(svg,'rect',x=0,y=y+row['size'],width=width,height=row_gap if i < len(rows)-1 else pad,fill='white')
        for left,right in edges:
            element(svg,'rect',x=left-3,y=y-3,width=right-left+5,height=row['size']+5,
                    fill='none',stroke=comparison.FRAME,stroke_width=3)
        y += row['size']
        if i < len(rows)-1:
            element(svg,'path',d=f'M{pad} {y+row_gap//2}H{width-pad}',stroke=base.RULE,stroke_width=3,fill='none')
            y += row_gap
    output = FOLDER / 'figures/baseline_comparison.svg'
    ET.ElementTree(svg).write(output,encoding='utf-8',xml_declaration=True)
    assert len(svg.findall('.//{'+SVG+'}image')) == 0
    assert len(svg.findall('.//{'+SVG+'}text')) == 14
    report = {'viewBox':[0,0,width,height],'physical_mm':[width/300*25.4,height/300*25.4],
              'embedded_bitmaps':0,'editable_text_lines':len(svg.findall('.//{'+SVG+'}text')),
              'trace_circles':len(svg.findall('.//{'+SVG+'}circle')),'bytes':output.stat().st_size}
    (FOLDER / 'figures/baseline_comparison_svg_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
