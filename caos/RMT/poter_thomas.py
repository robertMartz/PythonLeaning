"""
porter_thomas.py : Porter-Thomas (PT) statistics of the eigenvector components of GOE, GUE, COE and CUE matrices.

For a Haar-random normalized vector the components are (almost) independent Gaussians of variance 1/N, so the
intensities y = N |c_i|^2 follow
    real vectors    (GOE, COE):  P(y) = exp(-y/2) / sqrt(2 pi y)   (chi^2 with 1 degree of freedom)   <y^2> = 3
    complex vectors (GUE, CUE):  P(y) = exp(-y)                    (chi^2 with 2 degrees of freedom)  <y^2> = 2
No unfolding is needed: the variance 1/N of the components is the same for every state (invariance of the ensemble).

We also compute the "phase rigidity" of every eigenvector,
    rho_j = | sum_i psi_i^2 | / sum_i |psi_i|^2   (invariant under a global phase of the eigenvector),
which is 1 for a real vector (up to a global phase) and ~ sqrt(pi/(2N)) for a random complex vector.
It is the fingerprint of time-reversal symmetry in the eigenvectors.

Uses unfolding.py (goe_matrix, gue_matrix), eigenvector_phases.py (cue_matrix, coe_matrix), plots_regions.py (style).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.linalg import schur

from unfolding import get_goe_matrix, get_gue_matrix
from eigenvector_phases import cue_matrix, coe_matrix
from plots import REGION_STYLE, set_style


# ----------------------------------------------------------------------------------------------------------------------
# Eigenvectors
# ----------------------------------------------------------------------------------------------------------------------
def eigenvectors_hermitian(H: np.ndarray) -> np.ndarray:
    """Columns = eigenvectors of H (GOE / GUE), sorted by increasing energy."""
    return np.linalg.eigh(H)[1]


def eigenvectors_unitary(U: np.ndarray) -> np.ndarray:
    """Columns = orthonormal eigenvectors of a unitary matrix (COE / CUE), sorted by the phase of the eigenvalue.
    The complex Schur decomposition U = Z T Z^dagger gives T diagonal for a normal matrix, and Z is exactly unitary
    (more robust than np.linalg.eig). The phase of each column is arbitrary, which does not affect |c|^2 or rho."""
    T, Z = schur(U, output="complex")
    order = np.argsort(np.angle(np.diag(T)))
    return Z[:, order]


def phase_rigidity(V: np.ndarray) -> np.ndarray:
    """rho_j for every column of V."""
    return np.abs(np.sum(V ** 2, axis=0)) / np.sum(np.abs(V) ** 2, axis=0)


# ----------------------------------------------------------------------------------------------------------------------
# PT theory
# ----------------------------------------------------------------------------------------------------------------------
PT = {  # name: (scipy distribution of y, moments <y^m> for m = 1..4)
    "real": (stats.chi2(1), [1, 3, 15, 105]),                 # chi^2_1 : P(y) = exp(-y/2) / sqrt(2 pi y)
    "complex": (stats.expon(), [1, 2, 6, 24]),                # exp(-y)
}


def binned_density(dist, edges: np.ndarray) -> np.ndarray:
    """Theoretical density averaged over each bin (exact, also in the first bin where the real PT diverges)."""
    return np.diff(dist.cdf(edges)) / np.diff(edges)


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    N = 1000
    seed = 11
    rng = np.random.default_rng(seed)

    # (name, class, matrix/eigenvector generator)
    ensembles = {
        "GOE": ("real", lambda: eigenvectors_hermitian(get_goe_matrix(N, rng=rng))),
        "GUE": ("complex", lambda: eigenvectors_hermitian(get_gue_matrix(N, rng=rng))),
        "COE": ("real", lambda: eigenvectors_unitary(coe_matrix(N, rng))),
        "CUE": ("complex", lambda: eigenvectors_unitary(cue_matrix(N, rng))),
    }
    thirds = {"left": slice(0, N // 3), "middle": slice(N // 3, 2 * N // 3), "right": slice(2 * N // 3, N)}
    edges = np.linspace(0.0, 12.0, 61)
    centers = 0.5 * (edges[1:] + edges[:-1])

    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    layout = {"GOE": axes[0, 0], "GUE": axes[0, 1], "COE": axes[1, 0], "CUE": axes[1, 1]}
    rho_all = {}

    for name, (kind, gen) in ensembles.items():
        V = gen()
        print(f"{name}: orthonormality error = {np.max(np.abs(V.conj().T @ V - np.eye(N))):.1e}")
        Y = N * np.abs(V) ** 2                       # Y[i, j] = intensity of component i of eigenstate j
        rho = phase_rigidity(V)
        rho_all[name] = rho
        dist, mom_th = PT[kind]
        ax = layout[name]

        print(f"  {'region':8s} {'<y>':>6s} {'<y^2>':>7s} {'<y^3>':>7s} {'<y^4>':>8s} {'KS dist':>8s} {'<rho>':>7s}"
              f"    (theory: {mom_th})")
        for region, sl in thirds.items():
            y = Y[:, sl].ravel()
            m = [np.mean(y ** p) for p in (1, 2, 3, 4)]
            ks = stats.kstest(y, dist.cdf).statistic          # distance only: p-values are meaningless here (y not independent)
            print(f"  {region:8s} {m[0]:6.3f} {m[1]:7.3f} {m[2]:7.2f} {m[3]:8.1f} {ks:8.4f} {np.mean(rho[sl]):7.3f}")
            ax.hist(y, bins=edges, density=True, histtype="step", lw=2, color=REGION_STYLE[region]["color"],
                    label=f"{REGION_STYLE[region]['label']} (⟨y²⟩={m[1]:.2f})")
        ax.plot(centers, binned_density(dist, edges), "k--", lw=2,
                label="Porter–Thomas " + ("χ²₁" if kind == "real" else "exp(−y)"))
        ax.set(yscale="log", xlim=(0, 12), ylim=(1e-5, 5), xlabel="y = N |c|²", ylabel="P(y)",
               title=f"{name} (N = {N})")
        ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig("porter_thomas.png", dpi=200, bbox_inches="tight")

    # ---- phase rigidity of the eigenvectors
    fig2, ax2 = plt.subplots(1, 2, figsize=(11, 4.5))
    for name in ("GOE", "COE"):
        print(f"{name}: rho = {rho_all[name].min():.6f} ... {rho_all[name].max():.6f}  (real eigenvectors: rho = 1)")
    for name, color in (("GUE", "tab:red"), ("CUE", "tab:gray")):
        ax2[0].hist(rho_all[name], bins=40, density=True, histtype="step", lw=2, color=color, label=name)
        print(f"{name}: <rho> = {rho_all[name].mean():.4f}   (random complex vector: sqrt(pi/(2N)) = "
              f"{np.sqrt(np.pi / (2 * N)):.4f})")
    ax2[0].set(title="Rigidez de fase ρ (clases unitarias)", xlabel="ρ", ylabel="densidad")
    ax2[0].legend()
    ax2[1].axis("off")
    ax2[1].text(0.0, 0.7, "GOE y COE: ρ = 1 para todos los autovectores\n(autovectores reales salvo una fase global)\n\n"
                          "GUE y CUE: ρ ~ 1/√N  (autovectores complejos genéricos)", fontsize=12, va="center")
    fig2.tight_layout()
    fig2.savefig("phase_rigidity_dense.png", dpi=200, bbox_inches="tight")
    plt.show()