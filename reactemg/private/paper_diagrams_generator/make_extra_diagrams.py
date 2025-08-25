"""
Generate three transparent PNGs:

1. squares_labeled.png – four gray squares, labeled 0–3 (perfectly centred)
2. rhombus_blue.png    – four #6fb7ff diamonds
3. rhombus_pink.png    – four #ff8080 diamonds
"""

from PIL import Image, ImageDraw, ImageFont, ImagePalette
import os

# ── appearance / scaling ──────────────────────────────────────────────
SCALE = 50                  # 1 = 100-px shapes, 2 = 200-px, etc.

square_size = 100 * SCALE
gap         = 10  * SCALE
margin      = 20  * SCALE
font_px     = 48  * SCALE

num_shapes  = 4            # four items in a row

# colours
gray  = "#bfbfbf"
blue  = "#6fb7ff"
pink  = "#ff8080"

# ── canvas size (horizontal row) ──────────────────────────────────────
img_width  = num_shapes * square_size + (num_shapes - 1) * gap + 2 * margin
img_height = square_size + 2 * margin

# ── font ──────────────────────────────────────────────────────────────
try:
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_px)
except IOError:
    font = ImageFont.load_default()

# ── helper: bounding box metrics (Pillow-10-safe) ─────────────────────
def text_bbox(draw, text, font):
    if hasattr(draw, "textbbox"):                # Pillow ≥10
        return draw.textbbox((0, 0), text, font=font)
    # Pillow <10 fallback
    w, h = draw.textsize(text, font=font)
    return (0, 0, w, h)                          # l, t, r, b

# ── drawing functions ────────────────────────────────────────────────
def draw_squares_labeled(filename):
    img = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    for i in range(num_shapes):
        x0 = margin + i * (square_size + gap)
        y0 = margin
        x1 = x0 + square_size
        y1 = y0 + square_size
        draw.rectangle([x0, y0, x1, y1], fill=gray, outline="black")

        # centred label (0-based)
        label = str(i)
        l, t, r, b = text_bbox(draw, label, font=font)
        w = r - l
        h = b - t
        # subtract l and t so that the *bounding box* centre aligns to square centre
        draw.text(
            (
                x0 + (square_size - w) / 2 - l,
                y0 + (square_size - h) / 2 - t,
            ),
            label,
            fill="black",
            font=font,
        )

    img.save(filename, "PNG")
    print(f"Wrote {filename}")

def draw_rhombi(filename, colour):
    img = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    for i in range(num_shapes):
        cx = margin + i * (square_size + gap) + square_size / 2
        cy = margin + square_size / 2
        r  = square_size / 2
        pts = [
            (cx, cy - r),  # top
            (cx + r, cy),  # right
            (cx, cy + r),  # bottom
            (cx - r, cy),  # left
        ]
        draw.polygon(pts, fill=colour, outline="black")

    img.save(filename, "PNG")
    print(f"Wrote {filename}")

# ── create all three images ───────────────────────────────────────────
draw_squares_labeled("squares_labeled.png")
draw_rhombi("rhombus_blue.png",  blue)
draw_rhombi("rhombus_pink.png",  pink)
