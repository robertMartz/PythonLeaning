from unfolding import *
import os
import numpy as np
import matplotlib.pyplot as plt

def analyze_regions(regions: dict, deg: int = 5, base: str = "hermite", L_values: np.ndarray = None,
                    num_windows: int = 250, circular_N: int = None) -> dict:
    """Unfold every region and compute P(s), P(r~), Delta_3(L) and Sigma^2(L).

    Parameters:
    regions (dict): output of separate_spectrum_regions.
    deg, base: degree and type of polynomial used in the unfolding.
    L_values (np.ndarray): lengths for Delta_3 and Sigma^2 (default: linspace(0.5, 50, 100)).
    num_windows (int): windows per L.
    circular_N (int): if given, the "eigenvalues" are phases in (-pi, pi] of an N = circular_N circular spectrum
        (CUE/COE-like). Then NO polynomial unfolding is done: the mean density is exactly N / (2 pi), so the
        unfolded levels are simply phase * N / (2 pi).

    Returns:
    dict: results[region] = {eigenvalues, unfolded, spacings, ratios, mean_spacing, mean_ratio, L_values, delta3,
                             sigma2}
    """
    if L_values is None:
        L_values = np.linspace(0.5, 50.0, 100)

    results = {}
    for name, reg in regions.items():
        if circular_N is None:
            eps = unfold_spectrum(reg["eigenvalues"], reg["eta"], deg=deg, base=base)["unfolded_eigenvalues"]
        else:
            eps = reg["eigenvalues"] * circular_N / (2.0 * np.pi)
        spacings = get_spacing_distribution(eps)
        ratios = get_ratio_distribution(spacings)
        results[name] = {
            "eigenvalues": reg["eigenvalues"],
            "unfolded": eps,
            "spacings": spacings,
            "ratios": ratios,
            "mean_spacing": float(np.mean(spacings)),
            "mean_ratio": float(np.mean(ratios)),
            "L_values": L_values,
            "delta3": evaluate_delta3_statistic(eps, L_values, num_windows=num_windows),
            "sigma2": evaluate_number_variance(eps, L_values, num_windows=num_windows),
        }
        print(f"[{name:6s}] levels = {len(eps):5d}  <s> = {results[name]['mean_spacing']:.4f}  "
              f"<r~> = {results[name]['mean_ratio']:.4f}")
    return results


REGION_STYLE = {
    "left":   {"color": "tab:blue",  "label": "Región 1 (Inf. 33%)"},
    "middle": {"color": "tab:green", "label": "Región 2 (Centro 33%)"},
    "right":  {"color": "tab:red",   "label": "Región 3 (Sup. 33%)"},
}
# Theory: GOE dashed black, GUE solid black, Poisson dotted gray
THEORY_STYLE = {
    1: {"color": "black", "ls": "--", "lw": 2.0},
    2: {"color": "black", "ls": "-",  "lw": 2.0},
    0: {"color": "gray",  "ls": ":",  "lw": 2.0},
}

def set_style() -> None:
    """Light grid, frame in the legends, default (sans-serif) font as in the reference figure."""
    plt.rcParams.update({
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "legend.framealpha": 0.9,
    })

# ----------------------------------------------------------------------------------------------------------------------
# Figure 1: grid of statistics
# ----------------------------------------------------------------------------------------------------------------------
def _finish(fig, save_file, show):
    fig.tight_layout()
    if save_file is not None:
        folder = os.path.dirname(save_file)
        if folder:
            os.makedirs(folder, exist_ok=True)
        fig.savefig(save_file, bbox_inches="tight", dpi=200)   # save BEFORE show()
    if show:
        plt.show()


def plot_statistics_grid(results_goe: dict, results_gue: dict, N: int, exact_theory: bool = True,
                         save_file: str = "statistics_regions.pdf", show: bool = True,
                         labels: tuple = ("GOE", "GUE"), s_bins: int = 40, r_bins: int = 40) -> plt.Figure:
    """4 x 2 grid. Columns: GOE | GUE. Rows: P(s), P(r~), Delta_3(L), Sigma^2(L).
    In every panel: the three regions + RMT prediction (GOE dashed, GUE solid) + Poisson (dotted)."""
    set_style()
    fig, axes = plt.subplots(4, 2, figsize=(14, 17.5))
    columns = [(labels[0], results_goe), (labels[1], results_gue)]  # labels only change the titles

    s = np.linspace(0.0, 3.5, 600)
    r = np.linspace(0.0, 1.0, 600)
    L_max = max(np.nanmax(res["L_values"]) for _, results in columns for res in results.values())
    L_th = np.linspace(0.05, L_max, 800)

    mean_r_th = {b: mean_folded_ratio_theory(b) for b in (0, 1, 2)}

    for col, (ens, results) in enumerate(columns):
        # ---------------- Row 0: P(s) ----------------
        ax = axes[0, col]
        for region, st in REGION_STYLE.items():
            ax.hist(results[region]["spacings"], bins=s_bins, density=True, histtype="step", lw=2,
                    color=st["color"], label=st["label"])
        ax.plot(s, wigner_surmise_spacing_distribution(s, beta=1), label="Wigner GOE (β=1)", **THEORY_STYLE[1])
        ax.plot(s, wigner_surmise_spacing_distribution(s, beta=2), label="Wigner GUE (β=2)", **THEORY_STYLE[2])
        ax.plot(s, wigner_surmise_spacing_distribution(s, beta=0), label="Poisson", **THEORY_STYLE[0])
        ax.set(title=f"{ens} ({N}x{N}) — P(s) por Región", xlabel="Espaciamiento desdoblado s", ylabel="P(s)",
               xlim=(0, 3.5), ylim=(0, 1.07))
        ax.legend(loc="upper right")

        # ---------------- Row 1: P(r~) ----------------
        ax = axes[1, col]
        for region, st in REGION_STYLE.items():
            ax.hist(results[region]["ratios"], bins=r_bins, density=True, histtype="step", lw=2,
                    color=st["color"], label=f"{st['label']} (<r̃>={results[region]['mean_ratio']:.3f})")
        ax.plot(r, folded_ratio_distribution(r, beta=1), label=f"Teoría GOE (β=1, <r̃>≈{mean_r_th[1]:.3f})",
                **THEORY_STYLE[1])
        ax.plot(r, folded_ratio_distribution(r, beta=2), label=f"Teoría GUE (β=2, <r̃>≈{mean_r_th[2]:.3f})",
                **THEORY_STYLE[2])
        ax.plot(r, folded_ratio_distribution(r, beta=0), label=f"Poisson (<r̃>≈{mean_r_th[0]:.3f})",
                **THEORY_STYLE[0])
        ax.set(title=f"{ens} — Cociente de Espaciamientos P(r̃) por Región",
               xlabel="Cociente de espaciamientos próximos r̃ = min(s_n, s_{n-1}) / max(s_n, s_{n-1})",
               ylabel="P(r̃)", xlim=(0, 1), ylim=(0, 2.1))
        ax.xaxis.label.set_fontsize(10)  # long label: avoid clipping
        ax.legend(loc="upper left")

        # ---------------- Row 2: Delta_3(L) ----------------
        ax = axes[2, col]
        for region, st in REGION_STYLE.items():
            ax.plot(results[region]["L_values"], results[region]["delta3"], "o-", ms=4, color=st["color"],
                    label=st["label"])
        ax.plot(L_th, theory_delta3(L_th, 1, exact_theory), label="Dyson-Mehta GOE (β=1)", **THEORY_STYLE[1])
        ax.plot(L_th, theory_delta3(L_th, 2, exact_theory), label="Dyson-Mehta GUE (β=2)", **THEORY_STYLE[2])
        ax.plot(L_th, theory_delta3(L_th, 0), label="Poisson (L/15)", **THEORY_STYLE[0])
        ax.set(title=f"{ens} — Estadística Δ₃(L) por Región",
               xlabel="Longitud del intervalo L (en unidades de espaciamiento medio)", ylabel="Δ₃(L)",
               xlim=(0, L_max), ylim=(0, 0.55))
        ax.legend(loc="upper left")

        # ---------------- Row 3: Sigma^2(L) ----------------
        ax = axes[3, col]
        for region, st in REGION_STYLE.items():
            ax.plot(results[region]["L_values"], results[region]["sigma2"], "o-", ms=4, color=st["color"],
                    label=st["label"])
        ax.plot(L_th, theory_number_variance(L_th, 1, exact_theory), label="GOE (β=1)", **THEORY_STYLE[1])
        ax.plot(L_th, theory_number_variance(L_th, 2, exact_theory), label="GUE (β=2)", **THEORY_STYLE[2])
        ax.plot(L_th, theory_number_variance(L_th, 0), label="Poisson (L)", **THEORY_STYLE[0])
        ax.set(title=f"{ens} — Varianza del número Σ²(L) por Región",
               xlabel="Longitud del intervalo L (en unidades de espaciamiento medio)", ylabel="Σ²(L)",
               xlim=(0, L_max), ylim=(0, 1.6))
        ax.legend(loc="lower right")  # the upper-left corner is occupied by the Poisson line

    _finish(fig, save_file, show)
    return fig

# ----------------------------------------------------------------------------------------------------------------------
# Figure 2 and 3: spectrum of each region before / after the unfolding
# ----------------------------------------------------------------------------------------------------------------------
def plot_spectra_regions(results_goe: dict, results_gue: dict, N: int, kind: str = "raw", nbins: int = 40,
                         normalized: bool = False, save_file: str = None, show: bool = True) -> plt.Figure:
    """Histograms (histtype="step") of the three regions for GOE and GUE.

    kind = "raw"      : eigenvalues before the unfolding (+ semicircle for reference).
    kind = "unfolded" : eigenvalues after the unfolding (+ the expected constant density 1).

    The vertical axis is the number of levels per unit of energy (counts / bin width), NOT a histogram normalized
    to unit area per region. In this way the three regions are directly comparable and:
        - before the unfolding the histogram follows the semicircle rho(E) (different in every region),
        - after the unfolding it must be flat and equal to 1 (the unfolded spectrum has mean spacing 1).
    nbins is the number of bins PER REGION. normalized=True if the matrices were divided by sqrt(N).
    """
    if kind not in ("raw", "unfolded"):
        raise ValueError("kind must be 'raw' or 'unfolded'.")
    set_style()
    key = "eigenvalues" if kind == "raw" else "unfolded"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, (ens, results) in zip(axes, [("GOE", results_goe), ("GUE", results_gue)]):
        for region, st in REGION_STYLE.items():
            x = results[region][key]
            edges = np.linspace(x.min(), x.max(), nbins + 1)
            width = edges[1] - edges[0]
            ax.hist(x, bins=edges, weights=np.full(x.size, 1.0 / width), histtype="step", lw=2,
                    color=st["color"], label=st["label"])

        if kind == "raw":
            lo = min(results[reg]["eigenvalues"].min() for reg in REGION_STYLE)
            hi = max(results[reg]["eigenvalues"].max() for reg in REGION_STYLE)
            E = np.linspace(lo, hi, 600)
            ax.plot(E, semicircle_density(E, N, normalized), "k--", lw=2, label="Semicírculo de Wigner")
            ax.set(title=f"{ens} ({N}x{N}) — Antes del desdoblamiento",
                   xlabel="Autovalores E", ylabel="Niveles por unidad de energía  ρ(E)")
        else:
            ax.axhline(1.0, color="black", ls="--", lw=2, label="Densidad esperada = 1")
            ax.set(title=f"{ens} ({N}x{N}) — Después del desdoblamiento",
                   xlabel="Autovalores desdoblados", ylabel="Niveles por unidad de energía desdoblada",
                   ylim=(0, 1.6))
        ax.legend(loc="upper right" if kind == "unfolded" else "best")

    _finish(fig, save_file, show)
    return fig


if __name__ == "__main__":
    N = 5000  # matrix dimension
    seed = 42
    normalize = False  # if True, the matrices are divided by sqrt(N)
    c = 20  # levels discarded at the edges (see separate_spectrum_regions)
    deg, base = 5, "hermite"  # unfolding polynomial
    data_file = f"data_GOE_GUE_N{N}.npz"

    if os.path.exists(data_file):
        data = np.load(data_file)  # only the eigenvalues are read
        goe_eigenvalues, gue_eigenvalues = data["evals_goe"], data["evals_gue"]
        print(f"Data recovered from {data_file}")
    else:
        rng = np.random.default_rng(seed)
        goe_eigenvalues, _ = get_spectrum(get_goe_matrix(N, normalize=normalize, rng=rng), name="GOE")
        gue_eigenvalues, _ = get_spectrum(get_gue_matrix(N, normalize=normalize, rng=rng), name="GUE")
        np.savez(data_file, evals_goe=goe_eigenvalues, evals_gue=gue_eigenvalues)

    print("GOE");
    goe_res = analyze_regions(separate_spectrum_regions(goe_eigenvalues, c=c), deg=deg, base=base)
    print("GUE");
    gue_res = analyze_regions(separate_spectrum_regions(gue_eigenvalues, c=c), deg=deg, base=base)

    plot_spectra_regions(goe_res, gue_res, N, "raw", normalized=normalize, save_file="joint_figures/raw_spectra_regions.pdf")
    plot_spectra_regions(goe_res, gue_res, N, "unfolded", save_file="joint_figures/unfolded_spectra_regions.pdf")
    plot_statistics_grid(goe_res, gue_res, N=N, save_file="joint_figures/statistics_regions.pdf")
