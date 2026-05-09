import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patheffects as pe
from io import BytesIO
from PIL import Image


# -------------------- Crisp rendering helpers --------------------
def _enable_crisp_rendering():
    """Set rcParams to favor crisp, anti-aliased lines and text, with transparent bg."""
    rc = {
        "figure.figsize": (8.8, 8.4),
        "axes.linewidth": 1.25,
        "font.size": 12,
        "figure.facecolor": "none",        # transparent figure bg
        "axes.facecolor": "none",          # transparent axes bg (2D)
        "savefig.dpi": 300,
        "savefig.transparent": True,       # ensure transparent exports by default
        "lines.antialiased": True,
        "patch.antialiased": True,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "path.simplify": True,
        "path.simplify_threshold": 0.0,
    }
    for k, v in rc.items():
        if k in mpl.rcParams:
            mpl.rcParams[k] = v

def _stroke_all_text(fig, lw=0.8, fg="white"):
    """Add a subtle outline to every text object for extra clarity."""
    stroke = [pe.withStroke(linewidth=lw, foreground=fg)]
    for txt in fig.findobj(match=mpl.text.Text):
        # Skip empty strings (e.g., placeholders)
        if txt.get_text():
            txt.set_path_effects(stroke)

def _save_png_supersampled(fig, basename, scale=2):
    """
    Supersample PNG export with transparency:
      1) render at (scale × dpi) with transparent=True
      2) LANCZOS downsample (keeps alpha)
    """
    if scale <= 1:
        fig.savefig(f"{basename}.png", bbox_inches="tight", transparent=True)
        return

    buf = BytesIO()
    dpi_hi = mpl.rcParams.get("savefig.dpi", 300) * scale
    fig.savefig(buf, format="png", dpi=dpi_hi, bbox_inches="tight", transparent=True)
    buf.seek(0)

    im = Image.open(buf).convert("RGBA")  # keep alpha
    target = (max(1, im.width // scale), max(1, im.height // scale))
    im = im.resize(target, Image.LANCZOS)
    im.save(f"{basename}.png")

def _blend_axes_to_paper(ax,
                         grid_color=(0.88, 0.88, 0.88, 0.9),  # ≈ #E0E0E0
                         axis_color=(0.40, 0.40, 0.40, 1.0),  # ≈ #666
                         tick_color=(0.40, 0.40, 0.40, 1.0),  # ≈ #666
                         label_color=(0.15, 0.15, 0.15, 1.0), # ≈ #262626
                         grid_lw=0.8, axis_lw=0.8):
    """
    Make 3D grid/axes blend with white paper while keeping transparency.
    Uses mplot3d's private _axinfo for precise grid styling.
    """
    # Transparent panes (walls) already set elsewhere; keep them invisible
    # for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    for axis in [getattr(ax, f'{a}axis') for a in 'xyz' if hasattr(ax, f'{a}axis')]:        
        try:
            # Grid lines (subtle, paper-like)
            axis._axinfo["grid"]["color"] = grid_color
            axis._axinfo["grid"]["linewidth"] = grid_lw
            axis._axinfo["grid"]["linestyle"] = "-"

            # Axis line (frame) softened
            axis.line.set_color(axis_color)
            axis.line.set_linewidth(axis_lw)

            # Labels color
            axis.label.set_color(label_color)
        except Exception:
            pass

    try: ax.tick_params(axis="x", colors=tick_color, width=0.6, length=3)
    except Exception: ...
    try: ax.tick_params(axis="y", colors=tick_color, width=0.6, length=3)
    except Exception: ...
    try: ax.tick_params(axis="z", colors=tick_color, width=0.6, length=3)
    except Exception: ...
    
    # Light, subtle grid. (mplot3d respects this partially; _axinfo above is authoritative)
    ax.grid(True, linewidth=grid_lw, alpha=grid_color[-1])

    # Optional: lighten legend frame so it blends too
    leg = ax.get_legend()
    if leg:
        leg.get_frame().set_edgecolor((0.85, 0.85, 0.85, 1.0))  # ~#D9D9D9
        leg.get_frame().set_facecolor((1, 1, 1, 0.85))

