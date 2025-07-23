"""
Render the *initial* state of the horizontal “scanning mask” diagram—no scanner.

• 8 squares (4 pink, 4 blue) in one row
• Squares 2, 3, 5, 8 are masked (gray + 'M')
• No orange scanner overlay
"""

from PIL import Image, ImageDraw, ImageFont

# ── appearance / scaling ──────────────────────────────────────────────
SCALE = 1                  # 1 = 100-px squares, 2 = 200-px, etc.

square_size = 100 * SCALE
gap         = 10  * SCALE
margin      = 20  * SCALE
font_px     = 48  * SCALE

num_squares = 8

# colours
pink = "#ff8080"
blue = "#6fb7ff"
gray = "#bfbfbf"

# ── masking metadata ──────────────────────────────────────────────────
original_colors = [pink]*4 + [blue]*4
masked_indices  = {1, 2, 4, 7}            # 0-based: squares 2,3,5,8

current_colors = [
    gray if i in masked_indices else original_colors[i]
    for i in range(num_squares)
]
is_masked = [i in masked_indices for i in range(num_squares)]

# ── canvas size ───────────────────────────────────────────────────────
img_width  = num_squares * square_size + (num_squares - 1) * gap + 2 * margin
img_height = square_size + 2 * margin

# ── font ──────────────────────────────────────────────────────────────
try:
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_px)
except IOError:
    font = ImageFont.load_default()

# ── helper for bounding-box metrics (Pillow-10-safe) ──────────────────
def text_bbox(draw, text, font):
    if hasattr(draw, "textbbox"):              # Pillow ≥10
        return draw.textbbox((0, 0), text, font=font)
    w, h = draw.textsize(text, font=font)      # Pillow <10
    return (0, 0, w, h)

# ── draw still image ──────────────────────────────────────────────────
img  = Image.new("RGBA", (img_width, img_height), "white")
draw = ImageDraw.Draw(img)

for i in range(num_squares):
    x0 = margin + i * (square_size + gap)
    y0 = margin
    x1 = x0 + square_size
    y1 = y0 + square_size
    draw.rectangle([x0, y0, x1, y1], fill=current_colors[i], outline="black")

    if is_masked[i]:
        l, t, r, b = text_bbox(draw, "M", font=font)
        w = r - l
        h = b - t
        draw.text(
            (x0 + (square_size - w) / 2 - l,
             y0 + (square_size - h) / 2 - t),
            "M",
            fill="black",
            font=font,
        )

img.save("masked_frame0_no_scanner.png", "PNG")
print("Wrote masked_frame0_no_scanner.png")
