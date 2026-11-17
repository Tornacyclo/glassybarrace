import argparse
import csv
import subprocess
from colorsys import hls_to_rgb
from datetime import date
from functools import cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.font_manager import fontManager
from matplotlib.patches import FancyBboxPatch
from matplotlib.patheffects import Stroke, withStroke
from matplotlib.ticker import MaxNLocator
from PIL import Image, ImageDraw

CODECS = {
    '.mov': '-c:v prores_ks -profile:v 4444 -pix_fmt yuva444p10le -vf scale=out_color_matrix=bt709'
            ',setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709',
    '.webm': '-c:v libvpx-vp9 -pix_fmt yuva420p -row-mt 1',
    '.webp': '-c:v libwebp_anim -lossless 1 -pix_fmt bgra -loop 0',
}
FONTS = ['Inter', 'Segoe UI', 'DejaVu Sans', 'Noto Sans CJK JP', 'Microsoft YaHei', 'Yu Gothic', 'PingFang SC']
FONTS = [name for name in FONTS if name in {font.name for font in fontManager.ttflist}]
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


def resized(image, width, height=None):
    return np.array(Image.fromarray(image).resize((round(width), round(height or width)), Image.LANCZOS))


def badge(image, side, color):
    image = Image.fromarray(image)
    mask = Image.new('L', image.size)
    box = 0, 0, image.width - 1, image.height - 1
    ImageDraw.Draw(mask).ellipse(box, fill=255)
    ImageDraw.Draw(image).ellipse(box, outline=tuple(round(255 * c) for c in color), width=image.width // 14)
    image = Image.composite(image, Image.new('RGBA', image.size), mask)
    return resized(np.array(image), side)


def smooth(series, window):
    return np.convolve(np.pad(series, (len(window) - 1, 0), 'edge'), window / window.sum(), 'valid')


def fit(text, size, width, renderer):
    text.set_fontsize(size)
    measured = renderer.get_text_width_height_descent(text.get_text(), text.get_fontproperties(), False)[0]
    if measured > width:
        text.set_fontsize(size * width / measured)
    return min(measured, width)


def leader_panel(fig, aspect, wide, names, colors, profiles):
    photos = any(p[0] is not None for p in profiles)
    text_x, text_y, right = .052, .86, .72
    zoom = min(1, aspect)
    bottom = text_y - (.103 if any(p[2] for p in profiles) else .056) * zoom - .015
    if photos:
        if wide:
            photo_x, photo_top, photo_w = .703, .874, min(.232, .46 / aspect)
            text_x, text_y, right = photo_x, photo_top - photo_w * aspect - .055, .934
            card = (.689, text_y - .151, .259, .89 - (text_y - .151))
        else:
            photo_x, photo_top, photo_w = .066, .874, min(.226, .146 / aspect)
            text_x, text_y, right = photo_x + photo_w + .035, photo_top - photo_w * aspect / 2 + .05 * zoom, .929
            card = (.052, photo_top - photo_w * aspect - .014 * aspect, .896, photo_w * aspect + .028 * aspect)
        bottom = card[1] - .008
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
    tenure = fig.text(text_x, text_y - .056 * zoom, '', size=17 if photos else 13, va='center', **ink(.9))
    stats = fig.text(text_x, text_y - .103 * zoom, '', va='center', **ink(.72))
    renderer = fig.canvas.get_renderer()

    @cache
    def portrait(i):
        return resized(profiles[i][0], photo_ax.bbox.width) if profiles[i][0] is not None else EMPTY

    def update(i):
        _, country, specs = profiles[i]
        flag_w = 0
        if photos:
            photo.set_data(portrait(i))
        flag.set_data(country if country is not None else EMPTY)
        if country is not None:
            flag_w = .026 * zoom * country.shape[1] / (country.shape[0] * aspect)
            flag_ax.set_position((text_x, text_y - .013 * zoom, flag_w, .026 * zoom))
            flag_w += .010
        leader.set_x(text_x + flag_w)
        leader.set_text(names[i])
        leader.set_color(colors[i])
        fit(leader, 28 if photos else 25, (right - text_x - flag_w) * fig.bbox.width, renderer)
        stats.set_text(specs)
        fit(stats, 11, (right - text_x) * fig.bbox.width, renderer)

    return update, tenure, bottom


def scene(src, title, top, period, fps, size, profiles):
    with open(src, newline='', encoding='utf-8-sig') as f:
        (_, *names), *rows = filter(any, csv.reader(f))
    if not names or not rows or any(len(row) != len(names) + 1 for row in rows):
        raise ValueError('Expected a time column and one value per name on each row')
    if min(top, period, fps, *size) <= 0 or len(size) != 2:
        raise ValueError('Top, period, fps and both dimensions must be positive')
    roster = load_profiles(profiles, set(names)) if profiles else {}
    profiles = [roster.get(name, (None, None, '')) for name in names]
    labels, *columns = zip(*rows)
    data = np.nan_to_num([[float(x or 0) for x in col] for col in columns], posinf=0, neginf=0).T
    decimals = min(2, max(len(x.partition('.')[2]) for col in columns for x in col))
    top = min(top, len(names))
    steps = max(1, round(period * fps))
    t = np.minimum(np.arange((len(rows) - 1) * steps + fps + 1) / steps, len(rows) - 1)
    days = calendar(labels)
    dates = np.interp(t, range(len(rows)), days) if days is not None else t
    values = np.array([np.interp(t, range(len(rows)), col) for col in data.T]).T
    ranks = [np.argsort(-values[0], kind='stable')]
    for v in values[1:]:
        ranks.append(ranks[-1][np.argsort(-v[ranks[-1]], kind='stable')])
    lead = [rank[0] for rank in ranks]
    held, since = [], dates[0]
    for k in range(len(t)):
        if k and lead[k] != lead[k - 1]:
            gap = values[k - 1] - values[k - 1, lead[k]]
            gain = gap - values[k] + values[k, lead[k]]
            since = dates[k - 1] + max(gap[gap > 0] / gain[gap > 0], default=0) * (dates[k] - dates[k - 1])
        held.append(int(dates[k]) - int(since))
    window = np.hanning(fps // 3 + 3)[1:-1]
    y = np.array([smooth(p, window) for p in np.argsort(ranks).T]).T
    high, low = values.max(1), np.sort(values)[:, -top]
    x0 = smooth(np.where(low < 0, low, np.clip(np.minimum(2 * low - high, .9 * high), 0, None)), window)
    span = np.where(high > x0, high - x0, 1)
    colors = [hls_to_rgb((.55 + .382 * k) % 1, .66, .8) for k in np.argsort(np.argsort(-data.mean(0)))]

    dpi = min(size) / 9
    aspect = size[0] / size[1]
    wide = aspect >= 1.32 and any(p[0] is not None for p in profiles)
    fig = Figure(figsize=np.divide(size, dpi) + 1e-6, dpi=dpi, facecolor='none')
    canvas = FigureCanvasAgg(fig)
    renderer = canvas.get_renderer()
    update_leader, tenure, bottom = leader_panel(fig, aspect, wide, names, colors, profiles)
    chart_right = .654 if wide else .948
    chart_top = .875 if wide else bottom - .4 * dpi / size[1]
    ax = fig.add_axes((.052, .065, chart_right - .052, chart_top - .065))
    ax.set(xlim=(0, 1), ylim=(top - .35, -.65))
    ax.axis('off')
    px = 1 / ax.bbox.width
    row = ax.bbox.height / (top + .3)
    bar = min(36, .54 * row * 72 / dpi)
    text_size = min(16, .55 * row * 72 / dpi)
    flag_h = text_size * dpi / 72
    head = min(.9 * row, 1.7 * bar * dpi / 72)
    flags = [resized(p[1], flag_h * p[1].shape[1] / p[1].shape[0], flag_h) if p[1] is not None else None
             for p in profiles]
    gap = .7 * flag_h * px
    flag_x = max((f.shape[1] * px + gap / 2 for f in flags if f is not None), default=0)
    tags = [ax.text(flag_x, 0, name, va='center', **ink()) for name in names]
    stats = [ax.text(0, 0, p[2], va='center', **ink()) for p in profiles]
    scores = [ax.text(0, 0, f'{m:,.{decimals}f}', va='center', **ink()) for m in data.max(0)]
    bar_left = flag_x + max(fit(tag, text_size, .3 * ax.bbox.width, renderer) for tag in tags) * px + gap
    stats_x = 1 - max(fit(text, .85 * text_size, .32 * ax.bbox.width, renderer) for text in stats) * px
    bar_right = stats_x - max(fit(score, text_size, ax.bbox.width, renderer) for score in scores) * px - 3 * gap
    for text in stats:
        text.set_x(stats_x)
    cap = bar * dpi / 144 * px
    if title:
        heading = fig.text(.052, .93, title, weight='semibold', **ink())
        fit(heading, 21, .45 * size[0], renderer)
    stamp = fig.text(.948, .93, '', ha='right', weight='light', **ink(.94))
    grid = ax.add_collection(LineCollection([], colors='white', alpha=.12, linewidths=.8, zorder=0))
    bins = 1 + int((bar_right - bar_left) / px / dpi // 1.25)
    locator = MaxNLocator(bins, steps=[1, 2, 2.5, 5, 10])
    ticks = [ax.text(0, -.65 - dpi / 20 / row, '', size=.72 * text_size, ha='center', **ink(.6))
             for _ in range(bins + 1)]
    artists = [(
        ax.plot([], [], lw=.89 * bar, color=color, solid_capstyle='round', clip_on=False)[0], tag, score, text,
        ax.imshow(flag, aspect='auto', interpolation='bilinear') if flag is not None else None,
        ax.imshow(badge(p[0], head, color), aspect='auto', interpolation='bilinear') if p[0] is not None else None,
    ) for color, tag, score, text, flag, p in zip(colors, tags, scores, stats, flags, profiles)]

    def draw(k):
        v, pos, base, scale = values[k], y[k], x0[k], span[k]
        update_leader(lead[k])
        label = labels[round(t[k])]
        if days is not None:
            n, day = held[k], date.fromordinal(int(dates[k]))
            tenure.set_text(f"Leading for {n:,} {'day' if n == 1 else 'days'}" if n else 'Leading')
            label = label if len(label) == 4 else f'{day.day} {day:%B %Y}'
        stamp.set_text(label)
        fit(stamp, 30, .45 * size[0], renderer)
        marks = [m for m in locator.tick_values(base, base + scale) if base <= m <= base + scale]
        xs = bar_left + (np.array(marks) - base) / scale * (bar_right - bar_left)
        grid.set_segments([[(x, -.65), (x, top - .35)] for x in xs])
        for text in ticks[len(marks):]:
            text.set_visible(False)
        for text, mark, x in zip(ticks, marks, xs):
            text.set(x=x, text=f'{mark:,.15g}', visible=True)
        alpha = np.clip(2 * (top - pos) - .5, 0, 1)
        near = np.abs(pos[:, None] - pos).min(1, initial=1, where=v > v[:, None])
        opacity = alpha * np.clip(4 * near - 1.4, 0, 1)
        for (line, tag, score, specs, flag, face), value, p, a, o in zip(artists, v, pos, alpha, opacity):
            length = np.clip((value - base) / scale, 0, 1)
            end = bar_left + length * (bar_right - bar_left)
            tip = max(end, bar_left + (2 * cap + px if face is None else head * px))
            line.set(visible=bool(a), data=([bar_left + cap, tip - cap], [p, p]),
                     alpha=.47 * a, zorder=1 + length, path_effects=[
                         withStroke(linewidth=bar, foreground='white', alpha=.22 * a),
                         Stroke(linewidth=.085 * bar, foreground='white', alpha=.34 * a, offset=(0, .21 * bar))])
            score.set(x=tip + gap, text=f'{value:,.{decimals}f}')
            for text, weight in ((tag, 1), (score, .89), (specs, .69)):
                text.set(y=p, alpha=o * weight, visible=bool(o))
            if flag is not None:
                flag_w = flag.get_array().shape[1] * px
                flag.set(visible=bool(o), alpha=o, extent=(0, flag_w, p + flag_h / row / 2, p - flag_h / row / 2))
            if face is not None:
                face.set(visible=bool(a), alpha=a, zorder=2 + length,
                         extent=(tip - head * px, tip, p + head / row / 2, p - head / row / 2))
        canvas.draw()
        return bytes(canvas.buffer_rgba())

    return draw, len(t)


def start(*args):
    global draw
    draw = scene(*args)[0]


def frame(k):
    return draw(k)


def race(src, out='race.mov', title='', top=21, period=1.0, fps=30, size=(1920, 1080), profiles=None):
    args = src, title, top, period, fps, size, profiles
    count = scene(*args)[1]
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgba',
           '-s', '%dx%d' % size, '-r', str(fps), '-i', '-', *CODECS.get(Path(out).suffix.lower(), '').split(), out]
    with Pool(initializer=start, initargs=args) as pool, subprocess.Popen(cmd, stdin=subprocess.PIPE) as ff:
        for chunk in range(0, count, 64):
            ff.stdin.writelines(pool.imap(frame, range(chunk, min(chunk + 64, count))))
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
