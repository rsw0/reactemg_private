import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.legend_handler import HandlerPatch
import matplotlib.transforms as transforms

###############################################################################
# 1) Custom legend handler
###############################################################################
class NarrowTallRectHandler(HandlerPatch):
    """Forces a narrow, tall rectangle patch in the legend."""

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        w = width * 0.2    # 20% of legend's patch width
        h = height * 0.9   # 90% of legend's patch height
        x = xdescent + (width - w) / 2
        y = ydescent + (height - h) / 2

        patch = Rectangle(
            (x, y),
            w,
            h,
            facecolor=orig_handle.get_facecolor(),
            edgecolor=orig_handle.get_edgecolor(),
        )
        self.update_prop(patch, orig_handle, legend)
        patch.set_transform(trans)  # Keep transform to place it correctly
        return [patch]

###############################################################################
# 2) Main plotting code
###############################################################################
# Parameters
window_size = 600
lookahead   = 50
delta_pred  = 20
t0          = 0

major_ends  = list(range(t0, t0 + lookahead + 1, 10))  # 0..50 step 10
n_major     = len(major_ends)

row_y_window = [i * 2 for i in range(n_major)]
row_y_dots   = [i * 2 + 1 for i in range(n_major - 1)]

# Horizontal positions
base_start = -50  # draw start of top window
indent     = 6    # shift each window right
x_left     = base_start - 22  # extend a bit for labels
x_right    = t0 + lookahead   # stop at t+50

fig, ax = plt.subplots(figsize=(10, 5))

# Make fonts bigger overall
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize']   = 12
plt.rcParams['axes.titlesize']   = 12
plt.rcParams['xtick.labelsize']  = 12
plt.rcParams['ytick.labelsize']  = 12
plt.rcParams['legend.fontsize']  = 12
plt.rcParams['figure.titlesize'] = 12

# Shaded regions
ax.axvspan(t0, t0 + delta_pred, facecolor="lightskyblue", alpha=0.35)
ax.axvspan(t0, t0 + lookahead,  facecolor="navajowhite",  alpha=0.35)

# Draw each window (blue bar)
for idx, (end, y) in enumerate(zip(major_ends, row_y_window[::-1])):  # earliest on top
    start_true = end - window_size + 1
    start_draw = base_start + idx * indent
    width_draw = end - start_draw

    # Bar for the window
    ax.add_patch(
        Rectangle(
            (start_draw, y),
            width_draw,
            0.8,
            facecolor="steelblue",
            edgecolor="k",
            alpha=0.30,
        )
    )

    # Horizontal ellipsis ~15% into bar
    ell_x = start_draw + 0.15 * width_draw
    ax.text(
        ell_x,
        y + 0.4,
        "⋯",
        ha="center",
        va="center",
        fontsize=16,  # match vertical ellipsis
        color="dimgray",
    )

    # Label left of bars
    label_x = base_start - 22
    ax.text(
        label_x,
        y + 0.55,
        f"Window {idx * 10}",
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        label_x,
        y + 0.05,
        f"Range [t{start_true:+d}, t{end:+d}]",
        ha="left",
        va="bottom",
        fontsize=12,
    )

    # Per-timestep logits INSIDE each window, from t0 up to `end`
    if end >= t0:
        for s in range(t0, end + 1):
            ax.add_patch(
                Rectangle(
                    (s - 0.2, y + 0.2),  # bottom-left corner
                    0.4,                # width
                    0.4,                # height
                    facecolor="dimgray"
                )
            )

# Vertical ellipsis rows aligned near t
dots_x = t0 - 5
for y in row_y_dots:
    ax.text(
        dots_x, y + 0.4, "⋮", fontsize=16, ha="center", va="center", color="dimgray"
    )

# Patches for legend
logit_patch = Patch(facecolor="dimgray", label="per‑timestep logit")
hold_patch  = Patch(facecolor="lightskyblue", alpha=0.35, label="hold Δ=20")
look_patch  = Patch(facecolor="navajowhite",  alpha=0.35, label="look‑ahead Δ=50")

# Prediction marker
ax.axvline(t0, linestyle="--", color="black")

###############################################################################
# Customize the x-axis ticks
###############################################################################
# We add -50 for the left side of the top-most window => "t-599"
xticks = [-50, t0] + list(range(t0 + 10, x_right + 1, 10))
xlabels = ["t-599", "t"] + [f"+{v}" for v in range(10, x_right - t0 + 1, 10)]

ax.set_xticks(xticks)
ax.set_xticklabels(xlabels)

# Axis limits and style
ax.set_xlim(x_left, x_right)
ax.set_ylim(-1.5, row_y_window[-1] + 1.5)
ax.yaxis.set_visible(False)
for spine_side in ["left", "right", "top"]:
    ax.spines[spine_side].set_visible(False)

# Legend
ax.legend(
    handles=[hold_patch, look_patch, logit_patch],
    handler_map={logit_patch: NarrowTallRectHandler()},  # custom handler for thin/tall patch
    loc="upper right",
    fontsize=12,
    framealpha=0.9,
)

###############################################################################
# Place the horizontal ellipsis ("⋯") below the x-axis, aligned with tick labels
###############################################################################
mid_x = ( -50 + 0 ) / 2  # -25
text_offset = transforms.ScaledTranslation(0, -8/72, fig.dpi_scale_trans)

# put at x = -25, y=0 (the axis line in 'get_xaxis_transform()' space),
# then shift down by ~8 points so it aligns with typical tick labels
ax.text(
    mid_x,
    0,
    "⋯",
    ha="center",
    va="baseline",
    fontsize=12,
    color="dimgray",
    transform=ax.get_xaxis_transform() + text_offset,
    clip_on=False
)

plt.tight_layout()
plt.savefig('smoothing.pdf', format='pdf', dpi=300)
plt.show()
