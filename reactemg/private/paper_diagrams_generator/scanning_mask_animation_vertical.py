from PIL import Image, ImageDraw, ImageFont, ImagePalette   # ⬅︎ added ImagePalette
import os

# ── constants and colour setup (unchanged) ─────────────────────────────
SCALE = 10                # 1 = 100 px squares, 2 = 200 px squares, etc.

square_size = 100 * SCALE
gap         = 10  * SCALE
margin      = 20  * SCALE
font_px     = 48  * SCALE
num_squares = 8





pink = "#ff8080"
blue = "#6fb7ff"
gray = "#bfbfbf"
scanner_color = (255, 165, 0, 128)       # 50 %‑opaque orange

original_colors = [pink]*4 + [blue]*4
masked_indices = {1, 2, 4, 7}
current_colors = [gray if i in masked_indices else original_colors[i] for i in range(num_squares)]
is_masked = [i in masked_indices for i in range(num_squares)]

img_width = square_size + 2*margin
img_height = num_squares*square_size + (num_squares-1)*gap + 2*margin






try:
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_px)
except IOError:
    font = ImageFont.load_default()

frames = []
frame_hold = 3                 # identical frames to “pause” at each step
scanner_indices = list(range(num_squares))

# ── helper: width/height of text, Pillow‑10‑safe ───────────────────────
def text_wh(draw, text, font):
    """Return (width, height) using textbbox() if available."""
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top
    else:                       # Pillow <10 fallback
        return draw.textsize(text, font=font)

# ── frame renderer ─────────────────────────────────────────────────────
def draw_frame(scanner_idx=None):
    img = Image.new("RGBA", (img_width, img_height), "white")
    draw = ImageDraw.Draw(img)

    for i in range(num_squares):
        x0 = margin
        y0 = margin + i*(square_size + gap)
        x1 = x0 + square_size
        y1 = y0 + square_size
        draw.rectangle([x0, y0, x1, y1], fill=current_colors[i], outline="black")

        if is_masked[i]:
            w, h = text_wh(draw, "M", font=font)          # ⬅︎ changed
            draw.text(
                (x0 + (square_size - w)/2, y0 + (square_size - h)/2),
                "M",
                fill="black",
                font=font,
            )

    if scanner_idx is not None:
        x0 = margin
        y0 = margin + scanner_idx*(square_size + gap)
        x1 = x0 + square_size
        y1 = y0 + square_size
        overlay = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 0))
        odraw = ImageDraw.Draw(overlay)
        odraw.rectangle([x0, y0, x1, y1], fill=scanner_color)
        img = Image.alpha_composite(img, overlay)

    # Pillow 10 changed the Palette enum; guard for both cases
    try:
        return img.convert("P", palette=Image.Palette.ADAPTIVE)
    except AttributeError:
        return img.convert("P", palette=Image.ADAPTIVE)

# ── build animation ────────────────────────────────────────────────────
for idx in scanner_indices:
    frames.extend([draw_frame(scanner_idx=idx)]*frame_hold)
    if is_masked[idx]:
        current_colors[idx] = original_colors[idx]
        is_masked[idx] = False

frames.extend([draw_frame(scanner_idx=None)]*frame_hold)

out = "scanning_mask.gif"
frames[0].save(
    out,
    save_all=True,
    append_images=frames[1:],
    duration=200,
    loop=0,
    disposal=2,
)
print(f"GIF written to {out}")
