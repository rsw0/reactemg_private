"""
Animated “scanning mask” diagram
--------------------------------
• 8 squares in one horizontal row.
• Squares 2, 3, 5, and 8 start masked (gray + “M”).
• An orange 50 %-opaque scanner sweeps left → right.
  – When it moves past a masked square, that square reverts to its true colour.
• Compatible with Pillow 9.x and 10.x+ (uses textbbox fallback).
• Change SCALE to export higher resolutions.
"""

from PIL import Image, ImageDraw, ImageFont, ImagePalette
import os

# ── appearance / scaling ──────────────────────────────────────────────
SCALE = 10                  # 1 = 100 px squares, 2 = 200 px squares, etc.

square_size = 100 * SCALE
gap         = 10  * SCALE
margin      = 20  * SCALE
font_px     = 48  * SCALE

num_squares = 8

pink          = "#ff8080"
blue          = "#6fb7ff"
gray          = "#bfbfbf"
scanner_color = (255, 165, 0, 128)        # RGBA orange @ 50 % opacity

# ── initial colours & mask metadata ───────────────────────────────────
original_colors = [pink]*4 + [blue]*4
masked_indices  = {1, 2, 4, 7}            # 0-based (2nd, 3rd, 5th, 8th)

current_colors = [
    gray if i in masked_indices else original_colors[i]
    for i in range(num_squares)
]
is_masked = [i in masked_indices for i in range(num_squares)]

# ── canvas size (horizontal row) ──────────────────────────────────────
img_width  = num_squares * square_size + (num_squares - 1) * gap + 2 * margin
img_height = square_size + 2 * margin

# ── font ──────────────────────────────────────────────────────────────
try:
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_px)
except IOError:
    font = ImageFont.load_default()

# ── helper: width/height of text, Pillow-10-safe ──────────────────────
def text_wh(draw, text, font):
    """Return (w, h) using Pillow 10's textbbox if available."""
    if hasattr(draw, "textbbox"):
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return r - l, b - t
    # Pillow <10 fallback
    return draw.textsize(text, font=font)

# ── frame renderer ────────────────────────────────────────────────────
def draw_frame(scanner_idx=None):
    img = Image.new("RGBA", (img_width, img_height), "white")
    draw = ImageDraw.Draw(img)

    # Draw squares left → right
    for i in range(num_squares):
        x0 = margin + i * (square_size + gap)
        y0 = margin
        x1 = x0 + square_size
        y1 = y0 + square_size
        draw.rectangle([x0, y0, x1, y1], fill=current_colors[i], outline="black")

        if is_masked[i]:
            w, h = text_wh(draw, "M", font=font)
            draw.text(
                (x0 + (square_size - w) / 2, y0 + (square_size - h) / 2),
                "M",
                fill="black",
                font=font,
            )

    # Scanner overlay
    if scanner_idx is not None:
        x0 = margin + scanner_idx * (square_size + gap)
        y0 = margin
        x1 = x0 + square_size
        y1 = y0 + square_size
        overlay = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 0))
        odraw = ImageDraw.Draw(overlay)
        odraw.rectangle([x0, y0, x1, y1], fill=scanner_color)
        img = Image.alpha_composite(img, overlay)

    # Convert to palette mode for smaller GIFs (handles Pillow 9/10)
    try:
        return img.convert("P", palette=Image.Palette.ADAPTIVE)
    except AttributeError:                      # Pillow <10
        return img.convert("P", palette=Image.ADAPTIVE)

# ── build animation ───────────────────────────────────────────────────
frames = []
frame_hold     = 3            # number of identical frames at each stop
scanner_indices = list(range(num_squares))  # 0..7

for idx in scanner_indices:
    frames.extend([draw_frame(scanner_idx=idx)] * frame_hold)
    if is_masked[idx]:
        current_colors[idx] = original_colors[idx]
        is_masked[idx] = False

# final pause with no scanner
frames.extend([draw_frame(scanner_idx=None)] * frame_hold)

# ── save GIF ──────────────────────────────────────────────────────────
outfile = "scanning_mask_horizontal.gif"
frames[0].save(
    outfile,
    save_all=True,
    append_images=frames[1:],
    duration=200,   # ms per frame
    loop=0,
    disposal=2,
)
print(f"GIF written to {outfile}")
