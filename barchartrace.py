import argparse
import csv
import subprocess
from datetime import date
from functools import cache
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import fontManager
from matplotlib.patches import FancyBboxPatch
from matplotlib.patheffects import Stroke, withStroke
from PIL import Image

CODECS = {
    '.mov': '-c:v prores_ks -profile:v 4444 -pix_fmt yuva444p10le -vf scale=out_color_matrix=bt709'
            ',setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709',
    '.webm': '-c:v libvpx-vp9 -pix_fmt yuva420p',
    '.webp': '-c:v libwebp_anim -lossless 1 -pix_fmt bgra -loop 0',
}
FONTS = ['Inter', 'Segoe UI', 'DejaVu Sans', 'Noto Sans CJK JP', 'Microsoft YaHei', 'Yu Gothic', 'PingFang SC']
FONTS = [name for name in FONTS if name in {font.name for font in fontManager.ttflist}]
COLORS = ['#77d8ff', '#ff9978', '#f9cd74', '#a0ebbc', '#d0adff', '#8ca5ff', '#5edbcf', '#f18dbc']
EMPTY = np.zeros((1, 1, 4), dtype='uint8')


def ink(alpha=1):
    return dict(color='white', alpha=alpha, family=FONTS, parse_math=False,
                path_effects=[withStroke(linewidth=3, foreground='black', alpha=.32 * alpha)])


def calendar(labels):
    try:
        days = [date.fromisoformat(s + '-01-01' if len(s) == 4 else s).toordinal() for s in labels]
    except ValueError:
        return None
    return days if days == sorted(days) else None


def load_profiles(path, names):
    path = Path(path)

    @cache
    def pixels(file):
        with Image.open(path.parent / file) as image:
            return np.array(image.convert('RGBA'))

    def asset(reference, flag=False):
        if not reference:
            return None
        file, _, index = reference.rpartition('#')
        if index.isdecimal() and file:
            image = pixels(file)
            height, width = image.shape[:2]
            tile = round(width * 2 / 3) if flag else width
            start = int(index) * tile
            if not tile or height % tile or start + tile > height:
                raise ValueError(f'Invalid atlas reference: {reference}')
            return image[start:start + tile]
        image = pixels(reference)
        if flag:
            return image
        height, width = image.shape[:2]
        side = min(height, width)
        y, x = (height - side) // 2, (width - side) // 2
        return image[y:y + side, x:x + side]

    with path.open(newline='', encoding='utf-8-sig') as file:
        rows = csv.DictReader(file)
        if 'name' not in (rows.fieldnames or []):
            raise ValueError('Profiles need a name column')
        return {
            row['name']: (asset(row.get('portrait')), asset(row.get('flag'), flag=True), row.get('specs') or '')
            for row in rows if row['name'] in names
        }


def fit(text, size, width, renderer):
    text.set_fontsize(size)
    measured = renderer.get_text_width_height_descent(text.get_text(), text.get_fontproperties(), False)[0]
    if measured > width:
        text.set_fontsize(size * width / measured)


def leader_panel(fig, aspect, profiled, wide):
    text_x, text_y, right = .052, .889, .72
    zoom = min(1, aspect)
    if profiled:
        if wide:
            photo_x, photo_top, photo_w = .703, .889, min(.232, .46 / aspect)
            text_x, text_y, right = photo_x, photo_top - photo_w * aspect - .055, .937
            card = (.689, text_y - .151, .261, .905 - (text_y - .151))
        else:
            photo_x, photo_top, photo_w = .064, .892, min(.226, .146 / aspect)
            text_x, text_y, right = photo_x + photo_w + .035, .867, .929
            card = (.052, .726, .896, .18)
        fig.add_artist(FancyBboxPatch(
            card[:2], *card[2:], boxstyle='round,pad=0.008,rounding_size=.012', transform=fig.transFigure,
            facecolor=(.014, .028, .054, .59), edgecolor=(.72, .84, 1, .28), linewidth=.65, zorder=-1,
        ))
        photo_ax = fig.add_axes((photo_x, photo_top - photo_w * aspect, photo_w, photo_w * aspect))
        photo_ax.axis('off')
        photo = photo_ax.imshow(EMPTY, aspect='auto', interpolation='bilinear')
        flag_ax = fig.add_axes((text_x, text_y - .013 * zoom, .031, .026 * zoom), zorder=3)
        flag_ax.axis('off')
        flag = flag_ax.imshow(EMPTY, aspect='auto', interpolation='bilinear')
    leader = fig.text(text_x, text_y, '', weight='semibold', va='center', **ink())
    tenure = fig.text(text_x, text_y - .056 * zoom, '', size=17 if profiled else 13, va='center', **ink(.9))
    stats = fig.text(text_x, text_y - .103 * zoom, '', va='center', **ink(.72))
    renderer = fig.canvas.get_renderer()

    def update(name, color, profile):
        portrait, country, specs = profile
        flag_w = 0
        if profiled:
            photo.set_data(portrait if portrait is not None else EMPTY)
            flag.set_data(country if country is not None else EMPTY)
            if country is not None:
                flag_w = .026 * zoom * country.shape[1] / (country.shape[0] * aspect)
                flag_ax.set_position((text_x, text_y - .013 * zoom, flag_w, .026 * zoom))
                flag_w += .010
        leader.set_x(text_x + flag_w)
        leader.set_text(name)
        leader.set_color(color)
        fit(leader, 28 if profiled else 25, (right - text_x - flag_w) * fig.bbox.width, renderer)
        stats.set_text(specs)
        fit(stats, 11, (right - text_x) * fig.bbox.width, renderer)

    return update, tenure


def race(src, out='race.mov', title='', top=21, period=1.0, fps=30, size=(1920, 1080), profiles=None):
    with open(src, newline='', encoding='utf-8-sig') as f:
        (_, *names), *rows = filter(any, csv.reader(f))
    if not names or not rows or any(len(row) != len(names) + 1 for row in rows):
        raise ValueError('Expected a time column and one value per name on each row')
    if min(top, period, fps, *size) <= 0 or len(size) != 2:
        raise ValueError('Top, period, fps and both dimensions must be positive')
    roster = load_profiles(profiles, set(names)) if profiles else {}
    profile_rows = [roster.get(name, (None, None, '')) for name in names]
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
    aspect = size[0] / size[1]
    wide = aspect >= 1.32 and bool(roster)
    fig = Figure(figsize=np.divide(size, dpi) + 1e-6, dpi=dpi, facecolor='none')
    canvas = FigureCanvasAgg(fig)
    chart_right = .654 if wide else .948
    chart_top = .899 if wide else .697 if roster else .811
    ax = fig.add_axes((.052, .065, chart_right - .052, chart_top - .065))
    ax.set(xlim=(0, 1), ylim=(top - .35, -.65))
    ax.axis('off')
    row_height = ax.bbox.height / (top + .3)
    bar = min(36, .54 * row_height * 72 / dpi)
    text_size = min(16, .55 * row_height * 72 / dpi)
    flag_height = text_size * dpi / 72
    flag_space = (2 * flag_height + 6) / ax.bbox.width if any(p[1] is not None for p in profile_rows) else 0
    bar_left = max(.19, flag_space + .14)
    specs_x = .763 if any(p[2] for p in profile_rows) else 1
    score_space = min(.17, (.69 * len(f'{data.max():,.{decimals}f}') + 1.6) * flag_height / ax.bbox.width)
    bar_right = specs_x - score_space
    cap = (bar * dpi / 144 + 5) / ax.bbox.width
    update_leader, tenure = leader_panel(fig, aspect, bool(roster), wide)
    if title:
        heading = fig.text(.052, .946, title, weight='semibold', va='center', **ink())
        fit(heading, 21, (chart_right - .32) * size[0], canvas.get_renderer())
    stamp = fig.text(chart_right, .946, '', ha='right', va='center', **ink(.77))

    artists = []
    for i, (name, (_, country, specs)) in enumerate(zip(names, profile_rows)):
        color = COLORS[i % len(COLORS)]
        barline, = ax.plot([], [], lw=.89 * bar, color=color, solid_capstyle='round', clip_on=False)
        name_text = ax.text(flag_space, 0, name, va='center', **ink())
        fit(name_text, text_size, (bar_left - flag_space - cap) * ax.bbox.width, canvas.get_renderer())
        score = ax.text(0, 0, '', size=text_size, va='center', **ink())
        stats = ax.text(specs_x, 0, specs, va='center', **ink())
        fit(stats, .85 * text_size, (1 - specs_x) * ax.bbox.width, canvas.get_renderer())
        flag = (ax.imshow(country, aspect='auto', extent=(0, 1, 0, 1), interpolation='bilinear')
                if country is not None else None)
        flag_w = flag_height * (country.shape[1] / country.shape[0]) / ax.bbox.width if flag else 0
        artists.append((barline, name_text, score, stats, flag, flag_w))

    rank = np.argsort(-values[0], kind='stable')
    y = rank.argsort().astype(float)
    lead, since = rank[0], dates[0]
    update_leader(names[lead], COLORS[lead % len(COLORS)], profile_rows[lead])
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgba',
           '-s', '%dx%d' % size, '-r', str(fps), '-i', '-', *CODECS.get(Path(out).suffix.lower(), '').split(), out]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as ff:
        for tk, day, v in zip(t, dates, values):
            rank = rank[np.argsort(-v[rank], kind='stable')]
            if lead != rank[0]:
                lead = rank[0]
                update_leader(names[lead], COLORS[lead % len(COLORS)], profile_rows[lead])
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
            if days is not None:
                elapsed = max(0, int(day - since + 1e-8))
                tenure.set_text(f"{elapsed} {'day' if elapsed == 1 else 'days'} in first place")
            label = labels[round(tk)]
            stamp.set_text(date.fromordinal(int(day)).isoformat() if days is not None and len(label) != 4 else label)
            fit(stamp, 13, .26 * size[0], canvas.get_renderer())
            for (barline, name_text, score, stats, flag, flag_w), value, pos, gap in zip(artists, v, y, near):
                alpha = np.clip(2 * (top - pos) - .5, 0, 1)
                length = max(0, value - x0) / (high - x0 or 1)
                end = bar_left + (bar_right - bar_left) * length
                barline.set_data([bar_left, end], [pos, pos])
                barline.set_alpha(.47 * alpha)
                barline.set_zorder(1 + length)
                barline.set_path_effects([
                    withStroke(linewidth=bar, foreground='white', alpha=.22 * alpha),
                    Stroke(linewidth=.085 * bar, foreground='white', alpha=.34 * alpha, offset=(0, .21 * bar)),
                ])
                opacity = alpha * np.clip(3 * gap - 1.5, 0, 1)
                score.set_text(f'{value:,.{decimals}f}')
                score.set_x(end + cap)
                for text, weight in ((name_text, 1), (score, .89), (stats, .69)):
                    text.set_y(pos)
                    text.set_alpha(opacity * weight)
                    text.set_visible(bool(opacity))
                barline.set_visible(bool(alpha))
                if flag is not None:
                    flag.set_visible(bool(opacity))
                    flag.set_alpha(opacity)
                    half = .5 * flag_height / row_height
                    flag.set_extent((0, flag_w, pos + half, pos - half))
            canvas.draw()
            ff.stdin.write(canvas.buffer_rgba())
    if ff.returncode:
        raise subprocess.CalledProcessError(ff.returncode, cmd)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Transparent bar chart races', argument_default=argparse.SUPPRESS)
    p.add_argument('src')
    p.add_argument('out', nargs='?', help='.mov, .webm, .webp or %%04d.png')
    p.add_argument('--title')
    p.add_argument('--top', type=int, help='bars to show (default: 21)')
    p.add_argument('--profiles', help='CSV with name, portrait, flag and specs')
    p.add_argument('--period', type=float, help='seconds per row')
    p.add_argument('--fps', type=int)
    p.add_argument('--size', type=lambda s: tuple(map(int, s.split('x'))), help='e.g. 1080x1920')
    race(**vars(p.parse_args()))
