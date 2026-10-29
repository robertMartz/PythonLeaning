import os
import numpy as np
import matplotlib.pyplot as plt
from unfolding import (get_goe_matrix, get_gue_matrix, get_spectrum, separate_spectrum_regions,
                       wigner_surmise_spacing_distribution, folded_ratio_distribution)
from plots import (analyze_regions, plot_statistics_grid, REGION_STYLE, THEORY_STYLE,
                           set_style, theory_delta3, theory_number_variance, mean_folded_ratio_theory)

# ----------------------------------------------------------------------------------------------------------------------
# eigenvector matrix -> quasi-energies -> phases
# ----------------------------------------------------------------------------------------------------------------------
def phase_spectrum(eigenvectors: np.ndarray, gauge: str = "lapack", rng: np.random.Generator = None) -> dict:
    """Eigenvalues (quasi-energies) of the eigenvector matrix and their phases.

    Parameters:
    eigenvectors (np.ndarray): matrix whose columns are the eigenvectors (orthogonal for GOE, unitary for GUE).
    gauge (str): "lapack" (as returned by eigh) or "random" (random column phases / signs, see module docstring).
    rng: random generator for the "random" gauge.

    Returns:
    dict: lam (eigenvalues sorted by phase), theta (phases in (-pi, pi], ascending), N.
    """
    M = np.asarray(eigenvectors)
    N = M.shape[0]
    if gauge == "random":
        rng = rng or np.random.default_rng()
        if np.iscomplexobj(M):
            M = M * np.exp(2j * np.pi * rng.random(N))[None, :]
        else:
            M = M * rng.choice([-1.0, 1.0], size=N)[None, :]
    elif gauge != "lapack":
        raise ValueError("gauge must be 'lapack' or 'random'.")

    lam = np.linalg.eigvals(M)  # general (non-Hermitian) eigenproblem: slow for large N
    order = np.argsort(np.angle(lam))
    lam = lam[order]
    return {"lam": lam, "theta": np.angle(lam), "N": N}

# TODO: Check the following claim about the phase spectrum of quasi-energies: unit circle, uniform density, no unfolding
#  needed

def check_phase_spectrum(spec: dict, name: str, real_matrix: bool) -> None:
    """Numerical checks of unit circle, uniform density (no unfolding needed)."""
    lam, theta, N = spec["lam"], spec["theta"], spec["N"]
    print(f"\n[{name}] checks on the phase spectrum (N = {N})")
    print(f"  max | |lambda| - 1 |            = {np.max(np.abs(np.abs(lam) - 1.0)):.2e}   (unit circle)")
    nb = 30
    counts, _ = np.histogram(theta, bins=nb, range=(-np.pi, np.pi))
    expected = N / nb
    print(f"  phase density: counts per bin   = {counts.min()} ... {counts.max()}  (uniform: {expected:.1f}); "
          f"max rel. deviation = {np.max(np.abs(counts - expected)) / expected * 100:.2f} %")
    if real_matrix:
        # real matrix -> complex-conjugate pairs (theta, -theta); a real eigenvalue (+1 or -1) has no partner
        is_real = np.abs(lam.imag) < 1e-9
        t_c = theta[~is_real]
        print(f"  real eigenvalues (+1 / -1)       = {int(is_real.sum())}")
        print(f"  max | theta_j + theta_(partner) | = {np.max(np.abs(t_c + t_c[::-1])):.2e}   "
              f"(real matrix: complex-conjugate pairs, spectrum symmetric under theta -> -theta)")


# ----------------------------------------------------------------------------------------------------------------------
# Control: genuine COE / CUE matrices
# ----------------------------------------------------------------------------------------------------------------------
def cue_matrix(N: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-distributed unitary matrix (QR of a Ginibre matrix with the phases of diag(R) fixed, Mezzadri 2007)."""
    Z = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))) / np.sqrt(2.0)
    Q, R = np.linalg.qr(Z)
    d = np.diag(R)
    return Q * (d / np.abs(d))[None, :]


def coe_matrix(N: int, rng: np.random.Generator) -> np.ndarray:
    """COE matrix: U = V^T V with V from the CUE (unitary AND symmetric)."""
    V = cue_matrix(N, rng)
    return V.T @ V

# ----------------------------------------------------------------------------------------------------------------------
# Complex plane and histogram of the phases
# ----------------------------------------------------------------------------------------------------------------------

def plot_phase_spectrum(spec_goe: dict, spec_gue: dict, nbins_region: int = 20,
                        save_file: str = None, show: bool = True) -> plt.Figure:
    """Row 1 (GOE) and row 2 (GUE). Left: eigenvalues in the complex plane (the 3 regions in colour).
    Right: histogram (step) of the phases of each region: number of levels per unit phase; the horizontal dashed line
    is the uniform density N / (2 pi) expected for a circular ensemble."""
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(13, 11.5))
    rows = [("Matriz de autovectores O (GOE)", spec_goe), ("Matriz de autovectores U (GUE)", spec_gue)]

    for row, (title, spec) in enumerate(rows):
        lam, theta, N = spec["lam"], spec["theta"], spec["N"]
        regions = separate_spectrum_regions(theta, c=0)  # three arcs of 33% of the levels

        ax = axes[row, 0]
        phi = np.linspace(0.0, 2.0 * np.pi, 400)
        ax.plot(np.cos(phi), np.sin(phi), color="gray", lw=1.0, ls=":", label="círculo unidad")
        for name, st in REGION_STYLE.items():
            z = lam[regions[name]["slice"]]
            ax.scatter(z.real, z.imag, s=4, color=st["color"], label=st["label"])
        ax.set(title=f"{title}: autovalores en el plano complejo", xlabel="Re(λ)", ylabel="Im(λ)")
        ax.set_aspect("equal")
        ax.legend(loc="center", fontsize=9)

        ax = axes[row, 1]
        for name, st in REGION_STYLE.items():
            x = regions[name]["eigenvalues"]
            edges = np.linspace(x.min(), x.max(), nbins_region + 1)
            ax.hist(x, bins=edges, weights=np.full(x.size, 1.0 / (edges[1] - edges[0])), histtype="step", lw=2,
                    color=st["color"], label=st["label"])
        ax.axhline(N / (2.0 * np.pi), color="black", ls="--", lw=2, label="N / 2π (densidad uniforme)")
        ax.set(title=f"{title}: histograma de fases", xlabel="Fase θ", ylabel="Niveles por unidad de fase",
               xlim=(-np.pi, np.pi), ylim=(0, 1.6 * N / (2.0 * np.pi)))
        ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    if save_file:
        os.makedirs(os.path.dirname(save_file) or ".", exist_ok=True)
        fig.savefig(save_file, bbox_inches="tight", dpi=200)
    if show:
        plt.show()
    return fig


# ----------------------------------------------------------------------------------------------------------------------
# Numerical summary
# ----------------------------------------------------------------------------------------------------------------------

# Large-N reference values (numerical RMT results) used in the summary table
VAR_S_SURMISE = {0: 1.0, 1: 4.0 / np.pi - 1.0, 2: 3.0 * np.pi / 8.0 - 1.0}  # variance of P(s)
MEAN_R_LARGE_N = {0: 0.3863, 1: 0.5307, 2: 0.5996}  # <r~>

def summary_table(results: dict, title: str) -> None:
    """Per region and pooled: <s>, Var(s), <r~>, Sigma^2(10), Delta_3(10); followed by the theoretical references."""
    L_ref = 10.0
    print(f"\n{title}")
    print(f"{'':22s} {'<s>':>7s} {'Var(s)':>8s} {'<r~>':>8s} {'Sigma2(10)':>11s} {'Delta3(10)':>11s}")
    pooled_s, pooled_r, pooled_S2, pooled_D3 = [], [], [], []
    for name, st in REGION_STYLE.items():
        res = results[name]
        S2 = np.interp(L_ref, res["L_values"], res["sigma2"])
        D3 = np.interp(L_ref, res["L_values"], res["delta3"])
        print(f"{st['label']:22s} {res['spacings'].mean():7.3f} {res['spacings'].var():8.3f} "
              f"{res['ratios'].mean():8.4f} {S2:11.3f} {D3:11.3f}")
        pooled_s.append(res["spacings"]); pooled_r.append(res["ratios"]); pooled_S2.append(S2); pooled_D3.append(D3)
    s_all, r_all = np.concatenate(pooled_s), np.concatenate(pooled_r)
    print(f"{'pooled':22s} {s_all.mean():7.3f} {s_all.var():8.3f} {r_all.mean():8.4f} "
          f"{np.mean(pooled_S2):11.3f} {np.mean(pooled_D3):11.3f}")
    for beta, nm in ((1, "theory GOE/COE (β=1)"), (2, "theory GUE/CUE (β=2)"), (0, "Poisson")):
        print(f"{nm:22s} {1.0:7.3f} {VAR_S_SURMISE[beta]:8.3f} {MEAN_R_LARGE_N[beta]:8.4f} "
              f"{float(theory_number_variance(np.asarray(L_ref), beta)):11.3f} {float(theory_delta3(np.asarray(L_ref), beta)):11.3f}")
    print("  (Var(s) theory = Wigner surmise; <r~> theory = large-N values; Sigma2 and Delta3 = exact RMT curves)")


# ----------------------------------------------------------------------------------------------------------------------
# Comparison original spectrum  vs  phase spectrum  (vs genuine COE/CUE control)
# ----------------------------------------------------------------------------------------------------------------------
def _collect(results: dict, region: str = "all"):
    names = list(REGION_STYLE) if region == "all" else [region]
    spacings = np.concatenate([results[n]["spacings"] for n in names])
    ratios = np.concatenate([results[n]["ratios"] for n in names])
    L = results[names[0]]["L_values"]
    delta3 = np.nanmean([results[n]["delta3"] for n in names], axis=0)
    sigma2 = np.nanmean([results[n]["sigma2"] for n in names], axis=0)
    return spacings, ratios, L, delta3, sigma2


def plot_comparison(res_original: dict, res_phase: dict, name: str, N: int, region: str = "all",
                    res_control: dict = None, control_label: str = None,
                    save_file: str = None, show: bool = True) -> plt.Figure:
    """2 x 2 figure: P(s), P(r~), Delta_3(L), Sigma^2(L) for the spectrum of the original matrix (unfolded with the
    polynomial) and for the phase spectrum of its eigenvector matrix (no unfolding), both with the three regions pooled
    (region="all") or for a single region ("left", "middle", "right"). Optionally a control (genuine COE/CUE)."""
    set_style()
    series = [(res_original, f"Espectro original ({name})", "tab:blue", "o"),
              (res_phase, "Fases de la matriz de autovectores", "tab:orange", "s")]
    if res_control is not None:
        series.append((res_control, control_label, "gray", "^"))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10.5))
    s_th = np.linspace(0.0, 3.5, 600)
    r_th = np.linspace(0.0, 1.0, 600)
    mean_r_th = {b: mean_folded_ratio_theory(b) for b in (0, 1, 2)}

    # P(s) and P(r~): histograms
    ax_s, ax_r, ax_d, ax_v = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    for res, label, color, marker in series:
        spacings, ratios, L, delta3, sigma2 = _collect(res, region)
        ax_s.hist(spacings, bins=40, density=True, histtype="step", lw=2, color=color, label=label)
        ax_r.hist(ratios, bins=40, density=True, histtype="step", lw=2, color=color,
                  label=f"{label} (<r̃>={ratios.mean():.3f})")
        ax_d.plot(L, delta3, marker + "-", ms=3, color=color, label=label)
        ax_v.plot(L, sigma2, marker + "-", ms=3, color=color, label=label)

    ax_s.plot(s_th, wigner_surmise_spacing_distribution(s_th, 1), label="Wigner GOE/COE (β=1)", **THEORY_STYLE[1])
    ax_s.plot(s_th, wigner_surmise_spacing_distribution(s_th, 2), label="Wigner GUE/CUE (β=2)", **THEORY_STYLE[2])
    ax_s.plot(s_th, wigner_surmise_spacing_distribution(s_th, 0), label="Poisson", **THEORY_STYLE[0])
    ax_s.set(title="P(s)", xlabel="Espaciamiento desdoblado s", ylabel="P(s)", xlim=(0, 3.5), ylim=(0, 1.07))
    ax_s.legend(loc="upper right", fontsize=9)

    ax_r.plot(r_th, folded_ratio_distribution(r_th, 1), label=f"Teoría GOE/COE (<r̃>≈{mean_r_th[1]:.3f})",
              **THEORY_STYLE[1])
    ax_r.plot(r_th, folded_ratio_distribution(r_th, 2), label=f"Teoría GUE/CUE (<r̃>≈{mean_r_th[2]:.3f})",
              **THEORY_STYLE[2])
    ax_r.plot(r_th, folded_ratio_distribution(r_th, 0), label=f"Poisson (<r̃>≈{mean_r_th[0]:.3f})", **THEORY_STYLE[0])
    ax_r.set(title="P(r̃)", xlabel="r̃ = min(s_n, s_{n-1}) / max(s_n, s_{n-1})", ylabel="P(r̃)", xlim=(0, 1),
             ylim=(0, 2.6))
    ax_r.legend(loc="upper left", fontsize=8)

    L_max = float(np.nanmax(_collect(res_original, region)[2]))
    L_th = np.linspace(0.05, L_max, 800)
    for ax, fn, ylab, ylim, lab_p in ((ax_d, theory_delta3, "Δ₃(L)", (0, 0.55), "Poisson (L/15)"),
                                      (ax_v, theory_number_variance, "Σ²(L)", (0, 1.6), "Poisson (L)")):
        ax.plot(L_th, fn(L_th, 1), label="GOE/COE (β=1)", **THEORY_STYLE[1])
        ax.plot(L_th, fn(L_th, 2), label="GUE/CUE (β=2)", **THEORY_STYLE[2])
        ax.plot(L_th, fn(L_th, 0), label=lab_p, **THEORY_STYLE[0])
        ax.set(title=ylab, xlabel="Longitud del intervalo L (en unidades de espaciamiento medio)", ylabel=ylab,
               xlim=(0, L_max), ylim=ylim)
    ax_d.legend(loc="upper left", fontsize=9)
    ax_v.legend(loc="lower right", fontsize=9)

    where = "3 regiones" if region == "all" else REGION_STYLE[region]["label"]
    fig.suptitle(f"{name}: espectro original vs fases de la matriz de autovectores ({N}x{N}, {where})", fontsize=14)
    fig.tight_layout()
    if save_file:
        os.makedirs(os.path.dirname(save_file) or ".", exist_ok=True)
        fig.savefig(save_file, bbox_inches="tight", dpi=200)
    if show:
        plt.show()
    return fig

if __name__ == "__main__":
    # ------------------------------ configuration ------------------------------
    N = 3000                       # matrix dimension (used only if the data file does not exist)
    seed = 42
    gauge = "random"               # "lapack" (as returned by eigh) or "random" (random phases/signs of the columns)
    c = 10                         # levels discarded at the edges for the ORIGINAL spectrum (as before)
    deg, base = 5, "hermite"       # unfolding polynomial for the ORIGINAL spectrum
    run_control = True             # also evaluate a genuine COE and CUE matrix (smaller size N_control)
    N_control = 1500
    show = True
    out = f"figures_phases_{gauge}"
    data_file = f"data_GOE_GUE_N{N}.npz"
    L_values = np.linspace(0.5, 50.0, 100)
    rng = np.random.default_rng(seed)

    # ------------------------------ step 1-2: matrices and eigenvector matrices ------------------------------

    if os.path.exists(data_file):
        data = np.load(data_file)
        goe_ev, gue_ev = data["evals_goe"], data["evals_gue"]
        O_goe, U_gue = data["evecs_goe"], data["evecs_gue"]
        N = len(goe_ev)
        print(f"Data recovered from {data_file} (N = {N})")
    else:
        gen = np.random.default_rng(seed)
        goe_ev, O_goe = get_spectrum(get_goe_matrix(N, rng=gen), name="GOE")
        gue_ev, U_gue = get_spectrum(get_gue_matrix(N, rng=gen), name="GUE")
        np.savez(data_file, evals_goe=goe_ev, evecs_goe=O_goe, evals_gue=gue_ev, evecs_gue=U_gue)

    # ------------------------------ step 3-4: phase spectrum  ------------------------------

    cache = f"phases_{gauge}_N{N}.npz"
    if os.path.exists(cache):
        z = np.load(cache)
        spec_goe = {"lam": z["lam_goe"], "theta": z["theta_goe"], "N": N}
        spec_gue = {"lam": z["lam_gue"], "theta": z["theta_gue"], "N": N}
        print(f"Phase spectra recovered from {cache}")
    else:
        print(f"Diagonalizing the eigenvector matrices (gauge = {gauge}) ...")
        spec_goe = phase_spectrum(O_goe, gauge=gauge, rng=rng)
        spec_gue = phase_spectrum(U_gue, gauge=gauge, rng=rng)
        np.savez(cache, lam_goe=spec_goe["lam"], theta_goe=spec_goe["theta"],
                 lam_gue=spec_gue["lam"], theta_gue=spec_gue["theta"])

    check_phase_spectrum(spec_goe, "GOE: eigenvalues of O", real_matrix=True)
    check_phase_spectrum(spec_gue, "GUE: eigenvalues of U", real_matrix=False)
    plot_phase_spectrum(spec_goe, spec_gue, save_file=f"{out}/phase_spectrum.pdf", show=show)

    # ------------------------------ step 5: three regions, all statistics, NO unfolding ------------------------------
    phase_goe = analyze_regions(separate_spectrum_regions(spec_goe["theta"], c=0), L_values=L_values, circular_N=N)
    phase_gue = analyze_regions(separate_spectrum_regions(spec_gue["theta"], c=0), L_values=L_values, circular_N=N)
    plot_statistics_grid(phase_goe, phase_gue, N=N, labels=("Fases de O (GOE)", "Fases de U (GUE)"),
                         save_file=f"{out}/statistics_phase_regions.pdf", show=show)

    # ------------------------------ step 6: comparison with the original spectrum ------------------------------
    orig_goe = analyze_regions(separate_spectrum_regions(goe_ev, c=c), deg=deg, base=base, L_values=L_values)
    orig_gue = analyze_regions(separate_spectrum_regions(gue_ev, c=c), deg=deg, base=base, L_values=L_values)

    ctrl_goe = ctrl_gue = None
    if run_control:
        print(f"\nControl: genuine COE / CUE matrices of size {N_control}")
        coe_th = np.sort(np.angle(np.linalg.eigvals(coe_matrix(N_control, rng))))
        cue_th = np.sort(np.angle(np.linalg.eigvals(cue_matrix(N_control, rng))))
        ctrl_goe = analyze_regions(separate_spectrum_regions(coe_th, c=0), L_values=L_values, circular_N=N_control)
        ctrl_gue = analyze_regions(separate_spectrum_regions(cue_th, c=0), L_values=L_values, circular_N=N_control)

    summary_table(orig_goe, "GOE: spectrum of the original matrix (polynomial unfolding)")
    summary_table(phase_goe, "GOE: phases of the eigenvalues of O (no unfolding)")
    if run_control:
        summary_table(ctrl_goe, f"Control: genuine COE ({N_control}x{N_control}), phases (no unfolding)")
    summary_table(orig_gue, "GUE: spectrum of the original matrix (polynomial unfolding)")
    summary_table(phase_gue, "GUE: phases of the eigenvalues of U (no unfolding)")
    if run_control:
        summary_table(ctrl_gue, f"Control: genuine CUE ({N_control}x{N_control}), phases (no unfolding)")

    plot_comparison(orig_goe, phase_goe, "GOE", N, res_control=ctrl_goe, control_label=f"COE genuina (N={N_control})",
                    save_file=f"{out}/comparison_GOE.pdf", show=show)
    plot_comparison(orig_gue, phase_gue, "GUE", N, res_control=ctrl_gue, control_label=f"CUE genuina (N={N_control})",
                    save_file=f"{out}/comparison_GUE.pdf", show=show)

