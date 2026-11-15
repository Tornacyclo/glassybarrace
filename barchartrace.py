import argparse
import csv
import subprocess
from datetime import date
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import fontManager
from matplotlib.patheffects import Stroke, withStroke

CODECS = {
    '.mov': '-c:v prores_ks -profile:v 4444 -pix_fmt yuva444p10le -vf scale=out_color_matrix=bt709'
            ',setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709',
    '.webm': '-c:v libvpx-vp9 -pix_fmt yuva420p',
    '.webp': '-c:v libwebp_anim -lossless 1 -pix_fmt bgra -loop 0',
}
FONTS = ['Inter', 'Segoe UI', 'DejaVu Sans', 'Noto Sans CJK JP', 'Microsoft YaHei', 'Yu Gothic', 'PingFang SC']
FONTS = [name for name in FONTS if name in {font.name for font in fontManager.ttflist}]
COLORS = ['#61e8b6', '#9cc9ff', '#fa7198', '#ffd080', '#728aff', '#d998ff', '#81e7fc', '#bfe474']


def ink(alpha=1):
    return dict(color='white', alpha=alpha, family=FONTS, parse_math=False,
                path_effects=[withStroke(linewidth=3, foreground='black', alpha=.32 * alpha)])


def calendar(labels):
    try:
        days = [date.fromisoformat(s + '-01-01' if len(s) == 4 else s).toordinal() for s in labels]
    except ValueError:
        return None
    return days if days == sorted(days) else None


def race(src, out='race.mov', title='', top=10, period=1.0, fps=30, size=(1920, 1080)):
    with open(src, newline='', encoding='utf-8-sig') as f:
        (_, *names), *rows = filter(any, csv.reader(f))
    if not names or not rows or any(len(row) != len(names) + 1 for row in rows):
        raise ValueError('Expected a time column and one value per name on each row')
    if min(top, period, fps, *size) <= 0 or len(size) != 2:
        raise ValueError('Top, period, fps and both dimensions must be positive')
    labels, *columns = zip(*rows)
    data = np.nan_to_num([[float(x or 0) for x in col] for col in columns], posinf=0, neginf=0).T
    decimals = min(2, max(len(x.partition('.')[2]) for col in columns for x in col))
    top = min(top, len(names))
    steps = max(1, round(period * fps))
    t = np.minimum(np.arange((len(rows) - 1) * steps + fps + 1) / steps, len(rows) - 1)
    days = calendar(labels)
    dates = np.interp(t, range(len(rows)), days) if days is not None else t
    values = np.array([np.interp(t, range(len(rows)), col) for col in data.T]).T

    dpi = min(size) / 9
    fig = Figure(figsize=np.divide(size, dpi) + 1e-6, dpi=dpi, facecolor='none')
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes((.085, .08, .83, .71))
    bar = min(38, 40 * ax.bbox.height / dpi / (top + .4))
    text_size = np.clip(.54 * bar, 8, 17)
    label_width = .65 * text_size * (max(map(len, names)) + len(f'{data.max():,.{decimals}f}') + 2)
    room = max(.25, 1 - (label_width + bar / 2 + 12) * dpi / 72 / ax.bbox.width)
    if title:
        fig.text(.075, .953, title, size=11, **ink(.72))
    leader = fig.text(.075, .878, '', size=30, weight='semibold', **ink())
    tenure = fig.text(.075, .838, '', size=13, **ink(.8))
    stamp = fig.text(.925, .883, '', ha='right', size=20, **ink(.76))

    rank = np.argsort(-values[0], kind='stable')
    y = rank.argsort().astype(float)
    lead, since = rank[0], dates[0]
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgba',
           '-s', '%dx%d' % size, '-r', str(fps), '-i', '-', *CODECS.get(Path(out).suffix.lower(), '').split(), out]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as ff:
        for tk, day, v in zip(t, dates, values):
            rank = rank[np.argsort(-v[rank], kind='stable')]
            if lead != rank[0]:
                lead = rank[0]
                if days is not None:
                    k = int(np.ceil(tk)) - 1
                    gap = data[k] - data[k, lead]
                    gain = gap - (data[k + 1] - data[k + 1, lead])
                    crossing = max(gap[gap > 0] / gain[gap > 0], default=0)
                    since = days[k] + crossing * (days[k + 1] - days[k])
            y += (rank.argsort() - y) * min(1, 8 / fps)
            order = np.argsort(y)
            gaps = np.r_[1, np.diff(y[order]), 1]
            near = np.minimum(gaps[:-1], gaps[1:])[order.argsort()]
            high, low = v[rank[0]], v[rank[top - 1]]
            x0 = min(low, 0) if low < 0 else max(0, min(2 * low - high, .9 * high))
            ax.clear()
            ax.axis('off')
            ax.set(xlim=(x0, x0 + (high - x0 or 1) / room), ylim=(top - .35, -.65))
            leader.set_text(names[lead])
            leader.set_color(COLORS[lead % len(COLORS)])
            if days is not None:
                elapsed = max(0, int(day - since + 1e-8))
                tenure.set_text(f"{elapsed} {'day' if elapsed == 1 else 'days'} in first")
            label = labels[round(tk)]
            stamp.set_text(date.fromordinal(int(day)).isoformat() if days is not None and len(label) != 4 else label)
            for i, (value, pos, gap) in enumerate(zip(v, y, near)):
                alpha = np.clip(2 * (top - pos) - .5, 0, 1)
                if not alpha:
                    continue
                end = max(value, x0)
                glass = [withStroke(linewidth=bar, foreground='white', alpha=.25 * alpha),
                         Stroke(linewidth=bar / 8, foreground='white', alpha=.44 * alpha, offset=(0, bar / 5))]
                ax.plot([x0, end], [pos, pos], lw=.92 * bar, color=COLORS[i % len(COLORS)],
                        alpha=.56 * alpha, solid_capstyle='round', clip_on=False, path_effects=glass, zorder=value)
                opacity = alpha * np.clip(3 * gap - 1.5, 0, 1)
                ax.annotate(f'{names[i]}  {value:,.{decimals}f}', (end, pos), textcoords='offset points',
                            xytext=(bar / 2 + 10, 0), va='center', size=text_size, zorder=value,
                            **ink(opacity))
            canvas.draw()
            ff.stdin.write(canvas.buffer_rgba())
    if ff.returncode:
        raise subprocess.CalledProcessError(ff.returncode, cmd)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Transparent bar chart races', argument_default=argparse.SUPPRESS)
    p.add_argument('src')
    p.add_argument('out', nargs='?', help='.mov, .webm, .webp or %%04d.png')
    p.add_argument('--title')
    p.add_argument('--top', type=int)
    p.add_argument('--period', type=float, help='seconds per row')
    p.add_argument('--fps', type=int)
    p.add_argument('--size', type=lambda s: tuple(map(int, s.split('x'))), help='e.g. 1080x1920')
    race(**vars(p.parse_args()))
