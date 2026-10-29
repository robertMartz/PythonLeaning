"""
GOE -> GUE crossover,   H = H_GOE + (lambda / sqrt(N)) H_GUE

Only the imaginary antisymmetric part of H_GUE breaks time-reversal symmetry; with the factor 1/sqrt(N) its matrix
elements are of the order of the mean level spacing, so lambda is the N-INDEPENDENT crossover parameter
(scaling collapse). Observables, on the central 60% of the spectrum:

    <r~>       mean spacing ratio                           (GOE 0.531 -> GUE 0.600)
    Var(s)     variance of the unfolded spacing distribution (GOE 0.286 -> GUE 0.180)
    Delta_3(10) spectral rigidity at L = 10                  (GOE 0.236 -> GUE 0.176)
    <y^2>      second moment of the Porter-Thomas variable   (GOE 3 -> GUE 2)   [y = N |c|^2]
    <rho>      phase rigidity of the eigenvectors            (1 -> ~0)     [rho = |sum psi^2| / sum |psi|^2]

Uses unfolding.py and plots.py. Results are cached in mixed_goe_gue.pkl.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import i0e

from unfolding import get_goe_matrix, get_gue_matrix, unfold_spectrum, evaluate_delta3_statistic
from plots import set_style, wigner_surmise_spacing_distribution, folded_ratio_distribution, THEORY_STYLE

R_GOE, R_GUE = 0.5307, 0.5996          # large-N <r~>
Y_EDGES = np.linspace(0.0, 12.0, 61)   # bins for P(y)


def mixed_matrix(N: int, lam: float, rng: np.random.Generator) -> np.ndarray:
    """H = H_GOE + lam/sqrt(N) * H_GUE  (complex Hermitian)."""
    return get_goe_matrix(N, rng=rng) + lam / np.sqrt(N) * get_gue_matrix(N, rng=rng)


def measure(H: np.ndarray, trim: float = 0.2, deg: int = 5) -> dict:
    """All observables of ONE matrix (central part of the spectrum / of the eigenstates)."""
    E, V = np.linalg.eigh(H)
    N = len(E)
    lo, hi = int(trim * N), int((1 - trim) * N)
    Ec, W = E[lo:hi], V[:, lo:hi]

    s_raw = np.diff(Ec)
    ratios = np.minimum(s_raw[1:], s_raw[:-1]) / np.maximum(s_raw[1:], s_raw[:-1])
    eps = unfold_spectrum(Ec, np.arange(1, len(Ec) + 1) - 0.5, deg=deg, base="hermite")["unfolded_eigenvalues"]
    spacings = np.diff(eps)

    Y = N * np.abs(W) ** 2     # intensities y = N |c|^2
    rho = np.abs(np.sum(W ** 2, axis=0)) / np.sum(np.abs(W) ** 2, axis=0)
    return {"ratios": ratios, "spacings": spacings,
            "r": ratios.mean(), "var_s": spacings.var(),
            "d3": evaluate_delta3_statistic(eps, np.array([10.0]), num_windows=100)[0],
            "y2": np.mean(Y ** 2), "rho": rho.mean(), "rho2": np.mean(rho ** 2),
            "rho_all": rho, "ycounts": np.histogram(Y.ravel(), bins=Y_EDGES)[0]}


def run(N_list, n_real, lam_values, seed=5):
    """results[N][lam] = dict with per-matrix scalars (arrays) and pooled arrays."""
    rng = np.random.default_rng(seed)
    results = {}
    for N, nr in zip(N_list, n_real):
        results[N] = {}
        for lam in lam_values:
            ms = [measure(mixed_matrix(N, lam, rng)) for _ in range(nr)]
            res = {k: np.array([m[k] for m in ms]) for k in ("r", "var_s", "d3", "y2", "rho", "rho2")}
            res["spacings"] = np.concatenate([m["spacings"] for m in ms])
            res["ratios"] = np.concatenate([m["ratios"] for m in ms])
            res["rho_all"] = np.concatenate([m["rho_all"] for m in ms])
            res["ycounts"] = np.sum([m["ycounts"] for m in ms], axis=0)
            results[N][lam] = res
            print(f"  N = {N:5d}  lambda = {lam:5.2f}  <r~> = {res['r'].mean():.4f}  <y^2> = {res['y2'].mean():.3f}  "
                  f"<rho> = {res['rho'].mean():.3f}", flush=True)
    return results


def mean_sem(results, N, lam_values, key):
    a = np.array([results[N][l][key] for l in lam_values])
    return a.mean(axis=1), a.std(axis=1) / np.sqrt(a.shape[1])


def py_given_rho(y_fine: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """P(y | rho) averaged over the measured rho of the eigenstates (mixture)."""
    rho = np.clip(rho, 0.0, 1.0 - 1e-9)[:, None]
    y = y_fine[None, :]
    z = rho * y / (1 - rho ** 2)
    return np.mean(np.exp(-y / (1 + rho)) * i0e(z) / np.sqrt(1 - rho ** 2), axis=0)


if __name__ == "__main__":
    N_list = [250, 500, 1000]
    n_real = [20, 10, 6]
    lam_values = [0, 0.25, 0.5, 0.75, 1, 1.5, 2.0, 2.5,3.0, 3.5, 5, 7, 10]
    lam_show = [0, 0.25, 0.5, 1, 2, 5]
    cache = "mixed_goe_gue.pkl"

    if os.path.exists(cache):
        results = pickle.load(open(cache, "rb"))
        print(f"Results recovered from {cache}")
    else:
        results = run(N_list, n_real, lam_values)
        pickle.dump(results, open(cache, "wb"))
    N_max = max(N_list)
    set_style()
    colors = {n: c for n, c in zip(N_list, ("tab:blue", "tab:orange", "tab:green"))}

    # ------------------------------------------------------------------ table and collapse quality
    print("\nCrossover indicator chi = (<r~> - GOE) / (GUE - GOE)   (0 = GOE, 1 = GUE)")
    print("lambda " + " ".join(f"N={n:<6d}" for n in N_list) + "  max spread")
    chi = {n: (mean_sem(results, n, lam_values, "r")[0] - R_GOE) / (R_GUE - R_GOE) for n in N_list}
    for i, lam in enumerate(lam_values):
        vals = [chi[n][i] for n in N_list]
        print(f"{lam:5.2f}  " + " ".join(f"{v:8.2f}" for v in vals) + f"   {max(vals) - min(vals):.2f}")
    for n in N_list:
        i = np.argmax(chi[n] >= 0.5)
        l_half = np.interp(0.5, chi[n][i - 1:i + 1], np.array(lam_values)[i - 1:i + 1])
        print(f"N = {n:5d}: lambda_half (chi = 1/2) = {l_half:.2f}")

    print("\n<y^2> versus the prediction 2 + <rho^2>  (N = %d)" % N_max)
    y2, _ = mean_sem(results, N_max, lam_values, "y2")
    r2, _ = mean_sem(results, N_max, lam_values, "rho2")
    for lam, a, b in zip(lam_values, y2, r2):
        print(f"  lambda = {lam:5.2f}   <y^2> = {a:.3f}   2 + <rho^2> = {2 + b:.3f}")

    # ------------------------------------------------------------------ Figure 1: scaling collapse
    fig, axes = plt.subplots(3, 2, figsize=(13, 13))
    lam = np.array(lam_values, dtype=float)
    refs = {"r": (R_GOE, R_GUE), "var_s": (0.286, 0.180), "d3": (0.236, 0.176), "y2": (3.0, 2.0), "rho": (1.0, 0.0)}

    for N in N_list:                                             # (a) NO collapse in g = lambda/sqrt(N)
        m, e = mean_sem(results, N, lam_values, "r")
        axes[0, 0].errorbar(lam / np.sqrt(N), m, yerr=e, fmt="o-", ms=4, color=colors[N], label=f"N = {N}")
    axes[0, 0].set(xlabel="g = λ/√N  (coeficiente de H_GUE)", ylabel="⟨r̃⟩", title="(a) Sin colapso en g = λ/√N")
    axes[0, 0].set_xscale("symlog", linthresh=0.01)
    axes[0, 0].set_xlim(0, 0.8)
    panels = [(axes[0, 1], "r", "⟨r̃⟩", "(b) Colapso en λ: ⟨r̃⟩"),
              (axes[1, 0], "var_s", "Var(s)", "(c) Varianza de P(s)"),
              (axes[1, 1], "d3", "Δ₃(10)", "(d) Rigidez Δ₃(L=10)"),
              (axes[2, 0], "y2", "⟨y²⟩", "(e) Segundo momento de Porter–Thomas"),
              (axes[2, 1], "rho", "⟨ρ⟩", "(f) Rigidez de fase de los autovectores")]
    for ax, key, ylab, title in panels:
        for N in N_list:
            m, e = mean_sem(results, N, lam_values, key)
            ax.errorbar(lam, m, yerr=e, fmt="o-", ms=4, color=colors[N], label=f"N = {N}")
        ax.set(xlabel="λ", ylabel=ylab, title=title)
        ax.set_xscale("symlog", linthresh=0.25)         # linear near 0 (includes lambda = 0), logarithmic above
        ax.set_xlim(0, 12)
    for ax, key in ((axes[0, 0], "r"), (axes[0, 1], "r"), (axes[1, 0], "var_s"), (axes[1, 1], "d3"),
                    (axes[2, 0], "y2"), (axes[2, 1], "rho")):
        ax.axhline(refs[key][0], color="k", ls="--", lw=1.5, label="GOE")
        ax.axhline(refs[key][1], color="k", ls="-", lw=1.5, label="GUE")
    for N in (N_max,):                                           # prediction <y^2> = 2 + <rho^2>
        m, _ = mean_sem(results, N, lam_values, "rho2")
        axes[2, 0].plot(lam, 2 + m, "r:", lw=2.5, label="2 + ⟨ρ²⟩")
    for ax in axes.ravel():
        ax.legend(fontsize=8)
    fig.suptitle("H = H_GOE + (λ/√N) H_GUE : transición GOE → GUE", fontsize=14)
    fig.tight_layout()
    fig.savefig("mixed_collapse.png", dpi=200, bbox_inches="tight")

    # ------------------------------------------------------------------ Figure 2: P(s) and P(r~) for several lambda
    fig2, ax2 = plt.subplots(2, len(lam_show), figsize=(3.6 * len(lam_show), 7))
    s, r = np.linspace(0, 3.5, 400), np.linspace(0, 1, 400)
    for j, l in enumerate(lam_show):
        d = results[N_max][l]
        a = ax2[0, j]
        a.hist(d["spacings"], bins=np.linspace(0, 3.5, 31), density=True, histtype="step", lw=2, color="tab:purple")
        a.plot(s, wigner_surmise_spacing_distribution(s, 1), label="GOE", **THEORY_STYLE[1])
        a.plot(s, wigner_surmise_spacing_distribution(s, 2), label="GUE", **THEORY_STYLE[2])
        a.set(title=f"λ = {l}", xlim=(0, 3.5), ylim=(0, 1.1), xlabel="s", ylabel="P(s)" if j == 0 else "")
        b = ax2[1, j]
        b.hist(d["ratios"], bins=np.linspace(0, 1, 21), density=True, histtype="step", lw=2, color="tab:purple")
        b.plot(r, folded_ratio_distribution(r, 1), **THEORY_STYLE[1])
        b.plot(r, folded_ratio_distribution(r, 2), **THEORY_STYLE[2])
        b.set(xlim=(0, 1), ylim=(0, 2.1), xlabel="r̃", ylabel="P(r̃)" if j == 0 else "")
        if j == 0:
            a.legend(fontsize=8)
    fig2.suptitle(f"P(s) y P(r̃) al variar λ (N = {N_max})", fontsize=13)
    fig2.tight_layout()
    fig2.savefig("mixed_distributions.png", dpi=200, bbox_inches="tight")

    # ------------------------------------------------------------------ Figure 3: Porter-Thomas in the crossover
    fig3, ax3 = plt.subplots(1, len(lam_show), figsize=(3.6 * len(lam_show), 4.2))
    centers = 0.5 * (Y_EDGES[1:] + Y_EDGES[:-1])
    y_fine = (np.arange(12 * 500) + 0.5) / 500                    # 100 sub-points per bin -> bin-averaged theory
    for j, l in enumerate(lam_show):
        d = results[N_max][l]
        dens = d["ycounts"] / d["ycounts"].sum() / np.diff(Y_EDGES)
        a = ax3[j]
        a.step(centers, dens, where="mid", color="tab:purple", lw=2, label="datos")
        a.plot(centers, np.diff(stats.chi2(1).cdf(Y_EDGES)) / np.diff(Y_EDGES), **THEORY_STYLE[1], label="PT real (GOE)")
        a.plot(centers, np.diff(stats.expon().cdf(Y_EDGES)) / np.diff(Y_EDGES), **THEORY_STYLE[2], label="PT complejo (GUE)")
        f = py_given_rho(y_fine, d["rho_all"]).reshape(-1, 100).mean(axis=1)
        a.plot(centers, f, "r:", lw=2.5, label="P(y|ρ) con ρ medida")
        a.set(yscale="log", ylim=(1e-5, 5), xlim=(0, 12), xlabel="y = N|c|²", ylabel="P(y)" if j == 0 else "",
              title=f"λ = {l}  (⟨ρ⟩ = {d['rho_all'].mean():.2f})")
        if j == 0:
            a.legend(fontsize=7)
    fig3.suptitle(f"Porter–Thomas en la transición (N = {N_max})", fontsize=13)
    fig3.tight_layout()
    fig3.savefig("mixed_porter_thomas.png", dpi=200, bbox_inches="tight")
    plt.show()