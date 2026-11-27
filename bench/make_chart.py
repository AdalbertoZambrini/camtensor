"""Renders the README benchmark chart (light and dark SVG) from measured numbers.

    python bench/make_chart.py

Numbers come from docs/benchmarks.md (16 cameras, 1080p25 H.264).
"""

import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "docs", "assets")

ROWS = [  # (label, images/s, cpu %)
    ("OpenCV + NumPy", 156, 762),
    ("camtensor · CPU decode", 389, 454),
    ("camtensor · NVDEC", 400, 25),
]
REALTIME = 400  # 16 cameras x 25 fps

THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", text2="#52514e", muted="#8a8984",
                  grid="#e6e5e0", series=["#2a78d6", "#eb6834", "#1baf7a"]),
    "dark": dict(surface="#1a1a19", text="#f0efec", text2="#c3c2b7", muted="#8f8e88",
                 grid="#33332f", series=["#3987e5", "#d95926", "#199e70"]),
}

W, H = 880, 280
LABEL_W = 200
PANEL_W = 300
GAP = 40
TOP = 84
ROW_H = 44
BAR_H = 22
FONT = "system-ui,-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif"


def bar(x, y, w, h, r, color):
    """Bar anchored at x with a rounded data end (right side only)."""
    r = min(r, w / 2, h / 2)
    return (f'<path d="M{x},{y} H{x + w - r} Q{x + w},{y} {x + w},{y + r} V{y + h - r} '
            f'Q{x + w},{y + h} {x + w - r},{y + h} H{x} Z" fill="{color}"/>')


def panel(x0, title, subtitle, values, vmax, ticks, fmt, t, ref=None):
    s = [f'<text x="{x0}" y="{TOP - 34}" font-size="14" font-weight="600" fill="{t["text"]}">{title}</text>',
         f'<text x="{x0}" y="{TOP - 16}" font-size="12" fill="{t["text2"]}">{subtitle}</text>']
    bottom = TOP + ROW_H * len(values)
    vals = []
    for v in ticks:
        gx = x0 + PANEL_W * v / vmax
        s.append(f'<line x1="{gx:.1f}" y1="{TOP - 4}" x2="{gx:.1f}" y2="{bottom}" stroke="{t["grid"]}" stroke-width="1"/>')
        s.append(f'<text x="{gx:.1f}" y="{bottom + 16}" font-size="11" fill="{t["muted"]}" text-anchor="middle">{fmt(v)}</text>')
    for i, v in enumerate(values):
        y = TOP + i * ROW_H + (ROW_H - BAR_H) / 2
        w = max(PANEL_W * v / vmax, 3)
        s.append(bar(x0, y, w, BAR_H, 4, t["series"][i]))
        vals.append(f'<text x="{x0 + w + 6:.1f}" y="{y + BAR_H / 2 + 4.5:.1f}" font-size="13" font-weight="600" '
                    f'fill="{t["text"]}" stroke="{t["surface"]}" stroke-width="4" paint-order="stroke" '
                    f'stroke-linejoin="round">{fmt(v)}</text>')
    if ref is not None:
        rx = x0 + PANEL_W * ref[0] / vmax
        s.append(f'<line x1="{rx:.1f}" y1="{TOP - 4}" x2="{rx:.1f}" y2="{bottom}" stroke="{t["text2"]}" '
                 f'stroke-width="1.5" stroke-dasharray="4 3"/>')
        s.append(f'<text x="{rx + 4:.1f}" y="{TOP - 2}" font-size="11" fill="{t["text2"]}">{ref[1]}</text>')
    return s + vals


def render(name, t):
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         f'font-family="{FONT}" role="img" aria-labelledby="ttl desc">',
         '<title id="ttl">camtensor vs OpenCV, 16 cameras 1080p25 H.264</title>',
         '<desc id="desc">' + "; ".join(f"{l}: {a} images/s, {c}% CPU" for l, a, c in ROWS) + "</desc>",
         f'<rect width="{W}" height="{H}" rx="8" fill="{t["surface"]}"/>']
    for i, (label, _, _) in enumerate(ROWS):
        y = TOP + i * ROW_H + ROW_H / 2
        s.append(f'<rect x="16" y="{y - 5:.1f}" width="10" height="10" rx="2" fill="{t["series"][i]}"/>')
        s.append(f'<text x="32" y="{y + 4.5:.1f}" font-size="13" fill="{t["text"]}">{label}</text>')
    x1 = LABEL_W
    s += panel(x1, "Images per second", "higher is better",
               [r[1] for r in ROWS], 520, [0, 100, 200, 300, 400, 500], lambda v: f"{v:g}", t,
               ref=(REALTIME, "real time (16 × 25 fps)"))
    x2 = LABEL_W + PANEL_W + GAP + 20
    s += panel(x2, "CPU (100% = one core)", "lower is better",
               [r[2] for r in ROWS], 1000, [0, 250, 500, 750, 1000], lambda v: f"{v:g}%", t)
    s.append(f'<text x="16" y="{H - 14}" font-size="11" fill="{t["muted"]}">16 RTSP cameras · 1080p25 H.264 · '
             'output 640×640 f16 batches · i9-14900HX + RTX 5070 Laptop · details in docs/benchmarks.md</text>')
    s.append("</svg>")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"bench-{name}.svg"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(s) + "\n")


for name, theme in THEMES.items():
    render(name, theme)
print("wrote", OUT)
