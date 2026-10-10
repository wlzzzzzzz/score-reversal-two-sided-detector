"""Manuscript-sized plots, with an optional LaTeX font stack."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text

HERE = Path(__file__).resolve().parent
BLUE, ORANGE, GREEN, GRAY = '#2166AC', '#C65D26', '#23836B', '#5D6470'

def configure(usetex=False):
    plt.rcParams.update({
        'text.usetex': usetex,
        'font.family': 'serif', 'font.serif': ['DejaVu Serif'] if not usetex else ['serif'],
        'font.sans-serif': ['Helvetica'], 'font.monospace': ['Courier'],
        'text.latex.preamble': r'\usepackage[T1]{fontenc}\usepackage{times}\usepackage{amsmath,amssymb}',
        'font.size': 10, 'axes.titlesize': 10, 'axes.labelsize': 10,
        'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
        'figure.titlesize': 10, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .6, 'axes.labelpad': 4,
        'xtick.major.width': .6, 'ytick.major.width': .6,
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'savefig.facecolor': 'white', 'savefig.bbox': None,
    })

def save_figure(fig, name, output_dir=None, dpi=600, *, allowed_font_sizes=(10,)):
    """Save fixed-width PDF, PNG and SVG using Matplotlib."""
    target = Path(output_dir) if output_dir is not None else HERE / 'figures'
    target.mkdir(parents=True, exist_ok=True)
    if abs(fig.get_figwidth() - 5.5) > 1e-9:
        raise ValueError('Figure width must equal manuscript text width (5.5 in).')
    for text in fig.findobj(Text):
        if text.get_text() and text.get_visible():
            if text.get_fontsize() not in allowed_font_sizes:
                raise ValueError(f'Expected text at {allowed_font_sizes} pt: ' + text.get_text())
    pdf = target / f'{name}.pdf'
    fig.savefig(pdf, metadata={'CreationDate': None, 'ModDate': None,
                              'Creator': 'Paper figure builder'})
    fig.savefig(target / f'{name}.png', dpi=dpi)
    fig.savefig(target / f'{name}.svg', metadata={'Date': None})
    plt.close(fig)
    print(pdf)
