import io
import base64

import marimo as mo
from matplotlib.figure import Figure


def marimo_figure(
    fig: Figure,
    filename: str = "figure",
    download_format: str = "pdf",
    dpi: int = 150,
    transparent: bool = False,
) -> mo.Html:
    """Render a matplotlib figure with download link and drag-to-save support.

    Usage (as last expression in a marimo cell):
        marimo_figure(_fig)
        marimo_figure(_fig, "my_plot", download_format="png")
    """
    # PNG for display
    png_buf = io.BytesIO()
    fig.savefig(png_buf, format="png", dpi=dpi, bbox_inches="tight", transparent=transparent)
    png_b64 = base64.b64encode(png_buf.getvalue()).decode()

    # Download file (may be different format)
    if download_format == "png":
        dl_b64 = png_b64
        dl_mime = "image/png"
    else:
        dl_buf = io.BytesIO()
        fig.savefig(dl_buf, format=download_format, dpi=dpi, bbox_inches="tight", transparent=transparent)
        dl_b64 = base64.b64encode(dl_buf.getvalue()).decode()
        dl_mime = "application/pdf" if download_format == "pdf" else f"image/{download_format}"

    uid = f"mf_{id(fig)}"
    dl_uri = f"data:{dl_mime};base64,{dl_b64}"

    return mo.Html(
        f'<div style="display:inline-block;">'
        f'<a href="{dl_uri}" download="{filename}.{download_format}" draggable="false">'
        f'<img id="{uid}" src="data:image/png;base64,{png_b64}" '
        f'style="max-width:100%; display:block; cursor:grab;" '
        f'draggable="true" alt="{filename}" />'
        f'</a>'
        f'<div style="margin-top:4px;">'
        f'<a href="{dl_uri}" download="{filename}.{download_format}" '
        f'style="font-size:13px; color:#666; text-decoration:none;" '
        f'onmouseover="this.style.textDecoration=\'underline\'" '
        f'onmouseout="this.style.textDecoration=\'none\'"'
        f'>Download {download_format.upper()}</a>'
        f'</div>'
        f'</div>'
    )
