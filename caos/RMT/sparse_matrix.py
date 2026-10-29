"""
Question : how many zeros can a GOE / GUE matrix have before its spectrum stops looking GOE / GUE?

MODEL
    H = A o M       (Hadamard product)
    A : GOE / GUE matrix
    M : symmetric 0/1 mask, M_ij = M_ji, which keeps the matrix in its symmetry class
        (real symmetric -> GOE-like, Hermitian -> GUE-like)
    k : mean number of NONZERO off-diagonal elements per row ("connectivity" = mean degree of the graph).
        Fraction of zeros per row = 1 - k/(N-1).

    scheme = "bernoulli" : every pair (i<j) is kept with probability p = k/(N-1)  (Erdos-Renyi graph, the usual
                           sparse ensemble; the number of nonzero elements per row fluctuates, Poisson(k))
    scheme = "regular"   : EXACTLY k nonzero off-diagonal elements in every row (random k-regular graph)
    keep_diagonal = True : the diagonal (on-site Gaussian energies) is not diluted. If it were diluted, isolated
                           vertices would give exactly degenerate zero eigenvalues.

OBSERVABLES (all computed on the central 60% of the levels, to avoid the edges)
    <r~>      : mean spacing ratio (needs no unfolding): Poisson 0.386, GOE 0.531, GUE 0.600
    eta       : normalized class indicator (<r~> - 0.386) / (<r~>_RMT - 0.386):  0 = Poisson, 1 = RMT
    beta_eff  : effective Dyson index obtained by inverting the ratio-surmise curve <r~>(beta)
    Var(s)    : variance of the unfolded spacings: Poisson 1, GOE 0.286, GUE 0.180 (large N)
    Delta_3(L), Sigma^2(L): long-range rigidity (L = 10, 20), polynomial unfolding
    N*IPR     : mean normalized inverse participation ratio of the eigenvectors: 3 (GOE), 2 (GUE) if
                delocalized; ~N if localized on a single site.

Uses  unfolding.py  and  plots.py  .
"""

import os
import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt

from unfolding import (get_goe_matrix, get_gue_matrix, unfold_spectrum, evaluate_delta3_statistic,
                       evaluate_number_variance, wigner_surmise_spacing_distribution, folded_ratio_distribution)
from plots import set_style, theory_delta3, theory_number_variance, THEORY_STYLE

# Large-N reference values
R_POISSON = 2.0 * np.log(2.0) - 1.0  # 0.3863 (exact)
R_REF = {"GOE": 0.5307, "GUE": 0.5996}  # numerical large-N values (Atas et al. 2013)
VAR_S_REF = {"POISSON": 1.0, "GOE": 0.286, "GUE": 0.180}  # large-N variance of the spacing distribution
BETA = {"GOE": 1, "GUE": 2}


# ----------------------------------------------------------------------------------------------------------------------
# 1. Masks
# ----------------------------------------------------------------------------------------------------------------------
def bernoulli_mask(N: int, k: float, rng: np.random.Generator) -> np.ndarray:
    """Symmetric Erdos-Renyi mask: each pair i<j is kept with probability p = k/(N-1). Diagonal = False."""
    p = min(1.0, k / (N - 1))
    upper = np.triu(rng.random((N, N)) < p, 1)
    return upper | upper.T


def regular_mask(N: int, k: int, rng: np.random.Generator, swaps_per_edge: int = 10) -> np.ndarray:
    """Symmetric mask with EXACTLY k True elements in every row (random k-regular graph, diagonal False).
    Construction: circulant graph + random double-edge swaps (a-b, c-d) -> (a-d, c-b), which preserve all the degrees.
    For k > (N-1)/2 the complement of a (N-1-k)-regular graph is used. Requires N*k even."""
    k = int(round(k))
    if k >= N - 1:
        return ~np.eye(N, dtype=bool)
    if (N * k) % 2 != 0:
        raise ValueError("N*k must be even for a k-regular graph.")
    if k > (N - 1) // 2:
        return ~regular_mask(N, N - 1 - k, rng, swaps_per_edge) & ~np.eye(N, dtype=bool)

    A = np.zeros((N, N), dtype=bool)
    idx = np.arange(N)
    for d in range(1, k // 2 + 1):
        A[idx, (idx + d) % N] = True
        A[(idx + d) % N, idx] = True
    if k % 2 == 1:  # N even here (N*k even)
        h = np.arange(N // 2)
        A[h, h + N // 2] = True
        A[h + N // 2, h] = True

    edges = [tuple(e) for e in np.argwhere(np.triu(A, 1)).tolist()]
    E = len(edges)
    n_sw = swaps_per_edge * E
    ie, je, fl = rng.integers(0, E, n_sw), rng.integers(0, E, n_sw), rng.integers(0, 2, n_sw)
    for t in range(n_sw):
        i, j = ie[t], je[t]
        if i == j:
            continue
        a, b = edges[i]
        c, d = edges[j]
        if fl[t]:
            c, d = d, c
        if a == d or c == b or A[a, d] or A[c, b]:
            continue
        A[a, b] = A[b, a] = A[c, d] = A[d, c] = False
        A[a, d] = A[d, a] = A[c, b] = A[b, c] = True
        edges[i], edges[j] = (a, d), (c, b)
    return A


def make_mask(N: int, k: float, rng: np.random.Generator, scheme: str = "bernoulli") -> np.ndarray:
    if scheme == "bernoulli":
        return bernoulli_mask(N, k, rng)
    if scheme == "regular":
        return regular_mask(N, k, rng)
    raise ValueError("scheme must be 'bernoulli' or 'regular'.")


def sparse_matrix(ensemble: str, N: int, k: float, rng: np.random.Generator, scheme: str = "bernoulli",
                  keep_diagonal: bool = True) -> np.ndarray:
    """GOE/GUE matrix with a symmetric mask applied: H = A o M. The result stays real symmetric / Hermitian."""
    A = get_goe_matrix(N, rng=rng) if ensemble == "GOE" else get_gue_matrix(N, rng=rng)
    M = make_mask(N, k, rng, scheme)
    if keep_diagonal:
        np.fill_diagonal(M, True)
    return np.where(M, A, 0.0)


# ----------------------------------------------------------------------------------------------------------------------
# 2. Measures on one spectrum
# ----------------------------------------------------------------------------------------------------------------------
def _beta_curve():
    """<r~> as a function of beta for the surmise P(r) ~ (r + r^2)^beta / (1 + r + r^2)^(1 + 3 beta / 2)."""
    r = np.linspace(0.0, 1.0, 4001)
    betas = np.linspace(0.0, 3.0, 301)
    integrate = getattr(np, "trapezoid", None) or np.trapz  # name changed in numpy 2.0
    means = []
    for b in betas:
        P = (r + r ** 2) ** b / (1.0 + r + r ** 2) ** (1.0 + 1.5 * b)
        means.append(integrate(r * P, r) / integrate(P, r))
    return betas, np.array(means)


_BETAS, _MEANS = _beta_curve()


def beta_eff_from_r(mean_r: float) -> float:
    """Effective Dyson index: beta such that the surmise has the measured <r~> (0 = Poisson, 1 = GOE, 2 = GUE).
    The surmise gives 0.536 (GOE) and 0.603 (GUE), slightly above the large-N values, so beta_eff of a true GOE is
    about 0.97."""
    return float(np.interp(mean_r, _MEANS, _BETAS))


def eta_indicator(mean_r, ensemble: str):
    """0 = Poisson, 1 = RMT of the given class."""
    return (np.asarray(mean_r) - R_POISSON) / (R_REF[ensemble] - R_POISSON)


def ratios_of(E: np.ndarray) -> np.ndarray:
    """r~_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1}) from the (raw, not unfolded) levels."""
    s = np.diff(E)
    a, b = s[:-1], s[1:]
    mx = np.maximum(a, b)
    ok = mx > 0
    return np.minimum(a, b)[ok] / mx[ok]


def measure_spectrum(E: np.ndarray, V: np.ndarray = None, trim: float = 0.2, deg: int = 7,
                     L_values=(10.0, 20.0), num_windows: int = 100) -> dict:
    """Measures on the central part of one spectrum. V (optional): eigenvectors as columns, for the IPR."""
    n = len(E)
    lo, hi = int(trim * n), int((1.0 - trim) * n)
    Ec = E[lo:hi]
    out = {"ratios": ratios_of(Ec)}
    out["r_mean"] = float(np.mean(out["ratios"]))

    # unfolding by polynomial fit of the staircase (central part only)
    out.update({"spacings": np.array([]), "var_s": np.nan, "delta3": np.full(len(L_values), np.nan),
                "sigma2": np.full(len(L_values), np.nan)})
    try:
        eta = np.arange(1, len(Ec) + 1) - 0.5
        eps = unfold_spectrum(Ec, eta, deg=deg, base="hermite")["unfolded_eigenvalues"]
        s = np.diff(eps)
        out["spacings"] = s
        out["var_s"] = float(np.var(s))
        Lv = np.asarray(L_values, dtype=float)
        out["delta3"] = evaluate_delta3_statistic(eps, Lv, num_windows=num_windows)
        out["sigma2"] = evaluate_number_variance(eps, Lv, num_windows=num_windows)
    except ValueError:
        pass  # polynomial not monotonic: leave NaN

    if V is not None:
        ipr = np.sum(np.abs(V[:, lo:hi]) ** 4, axis=0)  # sum_i |psi_i|^4 for each central state
        out["nipr"] = float(len(E) * np.mean(ipr))
    return out


# ----------------------------------------------------------------------------------------------------------------------
# 3. Sweep over k
# ----------------------------------------------------------------------------------------------------------------------
def sweep(ensemble: str, N: int, k_values, n_real: int = 10, scheme: str = "bernoulli", keep_diagonal: bool = True,
          seed: int = 0, compute_ipr: bool = True, L_values=(10.0, 20.0), store_k=None, verbose: bool = True) -> dict:
    """Mean (and standard deviation over realizations) of every measure for each value of k.
    store_k: values of k (subset of k_values) for which the pooled spacings / ratios / one spectrum are kept for the
    distribution plots."""
    rng = np.random.default_rng(seed)
    k_values = list(k_values)
    store_k = set(store_k or [])
    nk = len(k_values)
    names = ["r_mean", "var_s", "nipr"]
    data = {nm: np.full((nk, n_real), np.nan) for nm in names}
    data["delta3"] = np.full((nk, n_real, len(L_values)), np.nan)
    data["sigma2"] = np.full((nk, n_real, len(L_values)), np.nan)
    stored = {}

    for ik, k in enumerate(k_values):
        pooled_s, pooled_r = [], []
        for ir in range(n_real):
            H = sparse_matrix(ensemble, N, k, rng, scheme, keep_diagonal)
            if compute_ipr:
                E, V = np.linalg.eigh(H)
            else:
                E, V = np.linalg.eigvalsh(H), None
            m = measure_spectrum(E, V, L_values=L_values)
            data["r_mean"][ik, ir] = m["r_mean"]
            data["var_s"][ik, ir] = m["var_s"]
            data["delta3"][ik, ir] = m["delta3"]
            data["sigma2"][ik, ir] = m["sigma2"]
            if compute_ipr:
                data["nipr"][ik, ir] = m["nipr"]
            if k in store_k:
                pooled_s.append(m["spacings"]);
                pooled_r.append(m["ratios"])
                if ir == 0:
                    stored.setdefault(k, {})["eigenvalues"] = E
        if k in store_k:
            stored[k]["spacings"] = np.concatenate(pooled_s)
            stored[k]["ratios"] = np.concatenate(pooled_r)
        if verbose:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                print(f"  [{ensemble} N={N} {scheme}] k = {k:7.2f}  <r~> = {np.nanmean(data['r_mean'][ik]):.4f}  "
                      f"Var(s) = {np.nanmean(data['var_s'][ik]):.3f}", flush=True)

    res = {"ensemble": ensemble, "N": N, "k": np.array(k_values, dtype=float), "L_values": np.array(L_values),
           "scheme": scheme, "n_real": n_real, "raw": data, "stored": stored}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # nipr is all-NaN when compute_ipr=False
        for nm in ("r_mean", "var_s", "nipr"):
            res[nm] = np.nanmean(data[nm], axis=1)
            res[nm + "_std"] = np.nanstd(data[nm], axis=1)
        res["delta3"] = np.nanmean(data["delta3"], axis=1)  # (nk, nL)
        res["sigma2"] = np.nanmean(data["sigma2"], axis=1)
    res["eta"] = eta_indicator(res["r_mean"], ensemble)
    res["beta_eff"] = np.array([beta_eff_from_r(r) for r in res["r_mean"]])
    return res


# ----------------------------------------------------------------------------------------------------------------------
# 4. Formal thresholds
# ----------------------------------------------------------------------------------------------------------------------
def k_crossing(k: np.ndarray, y: np.ndarray, level: float) -> float:
    """Value of k (interpolated linearly in log k) where the curve y(k) first reaches `level`. To be robust against
    noise at large k, y is replaced by its running maximum (y is expected to increase with k). NaN if never reached."""
    k, y = np.asarray(k, float), np.maximum.accumulate(np.asarray(y, float))
    if y[-1] < level:
        return np.nan
    i = int(np.argmax(y >= level))
    if i == 0:
        return float(k[0])
    t = (level - y[i - 1]) / (y[i] - y[i - 1])
    return float(10 ** (np.log10(k[i - 1]) + t * (np.log10(k[i]) - np.log10(k[i - 1]))))


def thresholds(res: dict, nsigma: float = 2.0) -> dict:
    """Operational definitions (single-matrix resolution, nsigma standard deviations):
        k_onset : <r~> exceeds the Poisson value by nsigma*sigma_P  -> repulsion becomes detectable
        k_half  : eta = 1/2 (middle of the crossover)
        k_rmt   : <r~> gets within nsigma*sigma_RMT of the dense value -> indistinguishable from the full ensemble
        k_90    : eta = 0.9 (robust alternative to k_rmt)
    sigma_P and sigma_RMT are the standard deviations of <r~> over single matrices at the smallest and largest k.
    k_rmt depends on the ESTIMATE of sigma_RMT, which is noisy with few matrices (use >= 20 per k); k_half and
    k_90 are much more stable."""
    k, r = res["k"], res["r_mean"]
    sigma_P, sigma_R = res["r_mean_std"][0], res["r_mean_std"][-1]
    r_dense = r[-1]
    return {
        "k_onset": k_crossing(k, r, r[0] + nsigma * sigma_P),
        "k_half": k_crossing(k, r, 0.5 * (r[0] + r_dense)),
        "k_rmt": k_crossing(k, r, r_dense - nsigma * sigma_R),
        "k_90": k_crossing(k, r, r[0] + 0.9 * (r_dense - r[0])),
        "sigma_P": sigma_P, "sigma_RMT": sigma_R, "r_poisson_end": r[0], "r_dense": r_dense,
    }


# ----------------------------------------------------------------------------------------------------------------------
# 5. Plots
# ----------------------------------------------------------------------------------------------------------------------
COLORS = {"GOE": "tab:blue", "GUE": "tab:red"}


def _vlines(ax, N):
    ax.axvline(1.0, color="gray", ls=":", lw=1)  # percolation threshold (Erdos-Renyi): k = 1
    ax.axvline(np.log(N), color="gray", ls="--", lw=1)  # connectivity threshold: k = ln N
    ymax = ax.get_ylim()[1]
    ax.text(1.0, ymax, " k=1", va="top", ha="left", fontsize=8, color="gray")
    ax.text(np.log(N), ymax, " k=ln N", va="top", ha="left", fontsize=8, color="gray")


def plot_sweep(results: dict, N: int, save_file: str = None, show: bool = True) -> plt.Figure:
    """2x2: <r~>, Var(s), Delta_3(10), <N IPR> versus k for GOE and GUE (results = {"GOE": res, "GUE": res})."""
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_r, ax_v, ax_d, ax_i = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    for ens, res in results.items():
        c = COLORS[ens]
        n_real = res["n_real"]
        ax_r.errorbar(res["k"], res["r_mean"], yerr=res["r_mean_std"] / np.sqrt(n_real), fmt="o-", color=c,
                      ms=4, capsize=2, label=f"{ens} diluida")
        ax_v.errorbar(res["k"], res["var_s"], yerr=res["var_s_std"] / np.sqrt(n_real), fmt="o-", color=c,
                      ms=4, capsize=2, label=f"{ens} diluida")
        iL = list(res["L_values"]).index(10.0)
        ax_d.plot(res["k"], res["delta3"][:, iL], "o-", color=c, ms=4, label=f"{ens} diluida")
        if not np.all(np.isnan(res["nipr"])):
            ax_i.errorbar(res["k"], res["nipr"], yerr=res["nipr_std"] / np.sqrt(n_real), fmt="o-", color=c,
                          ms=4, capsize=2, label=f"{ens} diluida")
        for ax, val in ((ax_r, R_REF[ens]), (ax_v, VAR_S_REF[ens]), (ax_d, float(theory_delta3(10.0, BETA[ens]))),
                        (ax_i, 3.0 if ens == "GOE" else 2.0)):
            ax.axhline(val, color=c, ls="--", lw=1.5, label=f"{ens} densa (teoría)")

    ax_r.axhline(R_POISSON, color="gray", ls=":", lw=2, label="Poisson")
    ax_v.axhline(1.0, color="gray", ls=":", lw=2, label="Poisson")
    ax_d.axhline(10.0 / 15.0, color="gray", ls=":", lw=2, label="Poisson (L/15)")
    ax_r.set(title="Cociente de espaciamientos ⟨r̃⟩", ylabel="⟨r̃⟩", ylim=(0.35, 0.65))
    ax_v.set(title="Varianza de P(s) (espectro desdoblado)", ylabel="Var(s)")
    ax_d.set(title="Rigidez espectral Δ₃(L=10)", ylabel="Δ₃(10)")
    ax_i.set(title="IPR normalizado ⟨N·IPR⟩ de los autovectores", ylabel="N·IPR", yscale="log")
    for ax in (ax_r, ax_v, ax_d, ax_i):
        ax.set_xscale("log")
        ax.set_xlabel("k = nº medio de elementos no nulos fuera de la diagonal por fila")
        ax.legend(fontsize=8, loc="best")
    for ax in (ax_r, ax_v, ax_d):
        _vlines(ax, N)
    fig.suptitle(f"Matrices GOE/GUE diluidas, N = {N}", fontsize=14)
    fig.tight_layout()
    if save_file:
        os.makedirs(os.path.dirname(save_file) or ".", exist_ok=True)
        fig.savefig(save_file, bbox_inches="tight", dpi=200)
    if show:
        plt.show()
    return fig


def plot_distributions_vs_k(res: dict, k_list, save_file: str = None, show: bool = True) -> plt.Figure:
    """P(s) (row 1) and P(r~) (row 2) for several k (columns). k_list must be a subset of store_k."""
    set_style()
    ens, N = res["ensemble"], res["N"]
    nc = len(k_list)
    fig, axes = plt.subplots(2, nc, figsize=(3.6 * nc, 7), squeeze=False)
    s = np.linspace(0, 3.5, 400)
    r = np.linspace(0, 1, 400)
    for j, k in enumerate(k_list):
        st = res["stored"][k]
        ik = list(res["k"]).index(k)
        a = axes[0, j]
        a.hist(st["spacings"], bins=np.linspace(0, 3.5, 31), density=True, histtype="step", lw=2, color=COLORS[ens],
               label="diluida")
        a.plot(s, wigner_surmise_spacing_distribution(s, 1), label="GOE", **THEORY_STYLE[1])
        a.plot(s, wigner_surmise_spacing_distribution(s, 2), label="GUE", **THEORY_STYLE[2])
        a.plot(s, wigner_surmise_spacing_distribution(s, 0), label="Poisson", **THEORY_STYLE[0])
        a.set(title=f"k = {k:g}   (η = {res['eta'][ik]:.2f})", xlim=(0, 3.5), ylim=(0, 1.1), xlabel="s",
              ylabel="P(s)" if j == 0 else "")
        b = axes[1, j]
        b.hist(st["ratios"], bins=np.linspace(0, 1, 21), density=True, histtype="step", lw=2, color=COLORS[ens])
        b.plot(r, folded_ratio_distribution(r, 1), **THEORY_STYLE[1])
        b.plot(r, folded_ratio_distribution(r, 2), **THEORY_STYLE[2])
        b.plot(r, folded_ratio_distribution(r, 0), **THEORY_STYLE[0])
        b.set(xlim=(0, 1), ylim=(0, 2.1), xlabel="r̃", ylabel="P(r̃)" if j == 0 else "")
        if j == 0:
            a.legend(fontsize=8, loc="upper right")
    fig.suptitle(f"{ens} diluida (N = {N}): P(s) y P(r̃) al variar k", fontsize=13)
    fig.tight_layout()
    if save_file:
        os.makedirs(os.path.dirname(save_file) or ".", exist_ok=True)
        fig.savefig(save_file, bbox_inches="tight", dpi=200)
    if show:
        plt.show()
    return fig


def plot_dos(res: dict, k_list, save_file: str = None, show: bool = True) -> plt.Figure:
    """Density of states of one matrix for several k, rescaled to unit variance (so shapes are comparable),
    with the semicircle of the same variance (dashed)."""
    set_style()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    cmap = plt.get_cmap("viridis")
    for j, k in enumerate(k_list):
        E = res["stored"][k]["eigenvalues"]
        x = (E - E.mean()) / E.std()
        ax.hist(x, bins=70, density=True, histtype="step", lw=1.8, color=cmap(j / max(1, len(k_list) - 1)),
                label=f"k = {k:g}")
    xs = np.linspace(-2, 2, 400)
    ax.plot(xs, np.sqrt(np.clip(4 - xs ** 2, 0, None)) / (2 * np.pi), "k--", lw=2, label="semicírculo")
    ax.set(title=f"{res['ensemble']} diluida (N = {res['N']}): densidad de estados", xlabel="E / σ_E",
           ylabel="ρ(E)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save_file:
        os.makedirs(os.path.dirname(save_file) or ".", exist_ok=True)
        fig.savefig(save_file, bbox_inches="tight", dpi=200)
    if show:
        plt.show()
    return fig


def plot_finite_size(by_N: dict, save_file: str = None, show: bool = True) -> plt.Figure:
    """by_N[ensemble][N] = sweep result. Left: eta(k) for several N. Right: thresholds k_onset, k_half, k_rmt vs N,
    compared with ln N."""
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for row, ens in enumerate(("GOE", "GUE")):
        Ns = sorted(by_N[ens])
        cmap = plt.get_cmap("plasma")
        ax = axes[row, 0]
        for j, N in enumerate(Ns):
            res = by_N[ens][N]
            ax.plot(res["k"], res["eta"], "o-", ms=4, color=cmap(j / max(1, len(Ns) - 1)), label=f"N = {N}")
        ax.axhline(1.0, color="k", ls="--", lw=1.5)
        ax.axhline(0.0, color="gray", ls=":", lw=2)
        ax.set(xscale="log", xlabel="k", ylabel="η = (⟨r̃⟩ − r_P)/(r_RMT − r_P)", title=f"{ens}: indicador de clase",
               ylim=(-0.15, 1.3))
        ax.legend(fontsize=8)

        ax = axes[row, 1]
        th = {N: thresholds(by_N[ens][N]) for N in Ns}
        for key, mk, lab in (("k_onset", "v", "k_onset (aparece repulsión)"), ("k_half", "o", "k_half (η = 1/2)"),
                             ("k_90", "s", "k_90 (η = 0.9)"),
                             ("k_rmt", "^", "k_rmt (≈ ensemble denso, ruidoso)")):
            ax.plot(Ns, [th[N][key] for N in Ns], mk + "-", label=lab)
        Nf = np.array(Ns, float)
        ax.plot(Nf, np.log(Nf), "k--", lw=1.2, label="ln N")
        ax.axhline(1.0, color="gray", ls=":", lw=1, label="k = 1 (percolación)")
        ax.set(xscale="log", yscale="log", xlabel="N", ylabel="k", title=f"{ens}: umbrales vs N")
        ax.legend(fontsize=8)
    fig.tight_layout()
    if save_file:
        os.makedirs(os.path.dirname(save_file) or ".", exist_ok=True)
        fig.savefig(save_file, bbox_inches="tight", dpi=200)
    if show:
        plt.show()
    return fig


def print_table(res: dict) -> None:
    ens, N = res["ensemble"], res["N"]
    iL = list(res["L_values"]).index(10.0)
    print(f"\n{ens}, N = {N}, scheme = {res['scheme']}, {res['n_real']} matrices per k")
    print(f"{'k':>7s} {'zeros %':>8s} {'<r~>':>7s} {'eta':>6s} {'beta_eff':>8s} {'Var(s)':>7s} {'Delta3(10)':>10s} "
          f"{'N*IPR':>8s}")
    for i, k in enumerate(res["k"]):
        print(f"{k:7.2f} {100 * (1 - k / (N - 1)):8.2f} {res['r_mean'][i]:7.4f} {res['eta'][i]:6.2f} "
              f"{res['beta_eff'][i]:8.2f} {res['var_s'][i]:7.3f} {res['delta3'][i, iL]:10.3f} {res['nipr'][i]:8.1f}")
    th = thresholds(res)
    print(f"  thresholds: k_onset = {th['k_onset']:.2f}, k_half = {th['k_half']:.2f}, k_90 = {th['k_90']:.2f}, "
          f"k_rmt = {th['k_rmt']:.2f}   "
          f"(sigma_P = {th['sigma_P']:.4f}, sigma_RMT = {th['sigma_RMT']:.4f}, ln N = {np.log(N):.2f})")
    print(f"  references: Delta3(10): Poisson {10 / 15:.3f}, GOE {float(theory_delta3(10.0, 1)):.3f}, "
          f"GUE {float(theory_delta3(10.0, 2)):.3f}; Var(s): GOE 0.286, GUE 0.180")


def print_scheme_comparison(res_bern: dict, res_reg: dict) -> None:
    """Bernoulli (Poisson-distributed degrees) vs regular (exactly k nonzero per row) at the same k."""
    print(f"\nBernoulli vs regular, {res_bern['ensemble']} N = {res_bern['N']}")
    print(f"{'k':>5s} | {'<r~> bern':>9s} {'<r~> reg':>9s} | {'N*IPR bern':>10s} {'N*IPR reg':>10s} | "
          f"{'Var(s) bern':>11s} {'Var(s) reg':>10s}")
    for i, k in enumerate(res_bern["k"]):
        print(f"{k:5.0f} | {res_bern['r_mean'][i]:9.4f} {res_reg['r_mean'][i]:9.4f} | {res_bern['nipr'][i]:10.1f} "
              f"{res_reg['nipr'][i]:10.1f} | {res_bern['var_s'][i]:11.3f} {res_reg['var_s'][i]:10.3f}")


# ----------------------------------------------------------------------------------------------------------------------
# 6. Main: experiments
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # ------------------------------ configuration ------------------------------
    N = 1000
    n_real = 12
    scheme = "bernoulli"  # "bernoulli" or "regular"
    keep_diagonal = True
    seed = 7
    k_values = [k for k in [0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128, 256] if k < N - 1] + [N - 1]
    k_show = [1.5, 3, 6, 12, 32, N - 1]  # for the distribution plots (must be in k_values)
    run_finite_size = False
    N_list = [250, 500, 1000, 2000]
    n_real_fs = 8
    k_values_fs = [1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64]
    run_scheme_comparison = False
    k_values_sch = [2, 4, 6, 8, 12, 24]  # integers, with N even (regular graph needs N*k even)
    show = True
    out = "figures_sparse"

    # ------------------------------ experiment 1: sweep in k at fixed N ------------------------------
    cache = f"sparse_N{N}_{scheme}.pkl"
    if os.path.exists(cache):
        results = pickle.load(open(cache, "rb"))
        print(f"Results recovered from {cache}")
    else:
        results = {ens: sweep(ens, N, k_values, n_real, scheme, keep_diagonal, seed, compute_ipr=True,
                              store_k=k_show) for ens in ("GOE", "GUE")}
        pickle.dump(results, open(cache, "wb"))

    for ens in ("GOE", "GUE"):
        print_table(results[ens])
    plot_sweep(results, N, save_file=f"{out}/sweep_N{N}.pdf", show=show)
    for ens in ("GOE", "GUE"):
        plot_distributions_vs_k(results[ens], k_show, save_file=f"{out}/distributions_{ens}_N{N}.pdf", show=show)
        plot_dos(results[ens], k_show, save_file=f"{out}/dos_{ens}_N{N}.pdf", show=show)

    # ------------------------------ experiment 2: finite-size dependence ------------------------------
    if run_finite_size:
        cache_fs = f"sparse_finite_size_{scheme}.pkl"
        if os.path.exists(cache_fs):
            by_N = pickle.load(open(cache_fs, "rb"))
        else:
            by_N = {ens: {n: sweep(ens, n, k_values_fs + [n - 1], n_real_fs, scheme, keep_diagonal, seed,
                                   compute_ipr=False) for n in N_list} for ens in ("GOE", "GUE")}
            pickle.dump(by_N, open(cache_fs, "wb"))
        plot_finite_size(by_N, save_file=f"{out}/finite_size.pdf", show=show)
        for ens in ("GOE", "GUE"):
            for n in N_list:
                th = thresholds(by_N[ens][n])
                print(f"{ens} N = {n:5d}: k_onset = {th['k_onset']:.2f}, k_half = {th['k_half']:.2f}, "
                      f"k_90 = {th['k_90']:.2f}, k_rmt = {th['k_rmt']:.2f}   (ln N = {np.log(n):.2f})")

    # ------------------------------ experiment 3: Bernoulli (fluctuating degree) vs exactly k per row ------------------------------
    if run_scheme_comparison:
        res_b = sweep("GOE", N, k_values_sch, n_real, "bernoulli", keep_diagonal, seed, compute_ipr=True)
        res_r = sweep("GOE", N, k_values_sch, n_real, "regular", keep_diagonal, seed, compute_ipr=True)
        print_scheme_comparison(res_b, res_r)
