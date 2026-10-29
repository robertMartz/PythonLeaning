import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sici
from scipy.integrate import cumulative_trapezoid

EULER_GAMMA = 0.5772156649015329

# TODO: Create a random NxN matrix of the GOE and GUE ensembles and obtain its energy spectrum

def get_goe_matrix(N: int, normalize: bool = False, rng: np.random.Generator = None) -> np.ndarray:
    """
    Generate a Gaussian Orthogonal Ensemble (GOE, beta = 1) random matrix of size N x N.

    Parameters:
    N (int): The dimension of the matrix.
    normalize (bool): Whether to normalize the matrix.
    rng (np.random.Generator): The random number generator to use.

    Returns:
    np.ndarray: The generated GOE matrix.
    """
    if rng is None:
        rng = np.random.default_rng()

    A = rng.standard_normal((N, N), dtype=np.float64)

    # Symmetrize the matrix to make it GOE
    H_goe = (A + A.T) / np.sqrt(2)

    if normalize:
        H_goe /= np.sqrt(N)

    return H_goe

def get_gue_matrix(N: int, normalize: bool = False, rng: np.random.Generator = None
) -> np.ndarray:
    """Generate a Gaussian Unitary Ensemble (GUE, beta = 2) random matrix of size n x n.

    Parameters:
    N (int): The dimension of the matrix.
    normalize (bool): Whether to normalize the matrix.
    rng (np.random.Generator): The random number generator to use.

    Returns:
    np.ndarray: The generated GUE matrix.
    """
    if rng is None:
        rng = np.random.default_rng()

    A_real = rng.standard_normal((N, N), dtype=np.float64)
    A_imag = rng.standard_normal((N, N), dtype=np.float64)

    # Create a complex matrix
    A_complex = ( A_real + 1j * A_imag ) / np.sqrt(2)  # To ensure the variance of the complex entries is 1

    # Symmetrize the matrix to make it GUE
    H_gue = (A_complex + A_complex.conj().T) / np.sqrt(2)

    if normalize:
        H_gue /= np.sqrt(N)

    return H_gue

def get_spectrum(matrix: np.ndarray, name = "Ensamble") -> tuple[np.ndarray, np.ndarray]:
    """Compute and return the eigenvalues (in ascending order) and eigenvectors of a given matrix.

    Parameters:
    matrix (np.ndarray): The input random matrix.

    Returns:
    eigenvalues: np.ndarray with the real eigenvalues of the matrix in ascending order.
    eigenvectors: np.ndarray with the corresponding eigenvectors of the matrix.
    """
    t0 = time.perf_counter()
    # np.linalg.eigh exploits the fact that the matrix is Hermitian (or real symmetric) to compute the eigenvalues and
    # eigenvectors more efficiently.
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    t1 = time.perf_counter()

    print(f"[{name}] Diagonalization completed. Time taken to compute eigenvalues and eigenvectors: "
          f"{t1 - t0:.6f} seconds.")

    return eigenvalues, eigenvectors

# TODO: Divide in three regions the spectrum using 33% and plot the histogram of the eigenvalues in each region. For
#  the two regions that include the edges of the spectrum, discard c eigenvalues in each edge to avoid the
#  tails of the semicircle.

def separate_spectrum_regions(eigenvalues: np.ndarray, c: int = 10) -> dict:
    """Separate the eigenvalues into three regions of 33%: left, middle, and right discarding c of the eigenvalues
    in each edge for the left and right regions.

    Parameters:
    eigenvalues (np.ndarray): The eigenvalues to separate.
    c (int): The number of eigenvalues to discard at each edge.

    Returns: dict with a dict for each region containing the corresponding slice, eigenvalues and the accumulative
    step function values

    """
    n = len(eigenvalues)

    # Accumulative step function : eta_ = i - 0.5 for i = 1, 2, ..., n
    eta_global = np.arange(1, n + 1, dtype=np.float64) - 0.5

    # Cut points for the three regions
    cut_point1 = n // 3
    cut_point2 = 2 * n // 3

    regions = {
        "left": {
            "slice": slice(c, cut_point1),
            "eigenvalues": eigenvalues[c:cut_point1],
            "eta": eta_global[c:cut_point1]
        },
        "middle": {
            "slice": slice(cut_point1, cut_point2),
            "eigenvalues": eigenvalues[cut_point1:cut_point2],
            "eta": eta_global[cut_point1:cut_point2]
        },
        "right": {
            "slice": slice(cut_point2, n - c),
            "eigenvalues": eigenvalues[cut_point2:n - c],
            "eta": eta_global[cut_point2:n - c]
        }
        # "Espectro Completo": {
        #       "slice": slice(recorte_borde, N - recorte_borde),
        #       "E": eigenvalores[recorte_borde:N - recorte_borde],
        #       "eta": eta_global[recorte_borde:N - recorte_borde],
        #   },
    }

    return regions

def plot_spectrum(eigenvalues: np.ndarray, title: str = "Eigenvalue Spectrum",
                  bins: int = 30, color: str = "blue", save_file="hist.pdf") -> None:
    """Plot the histogram of eigenvalues.

    Parameters:
    eigenvalues (np.ndarray): The eigenvalues to plot.
    title (str): The title of the plot.
    bins (int): The number of bins for the histogram.
    bars (bool): Whether to plot the histogram as a bar plot.
    color (str): The color of the histogram.
    save_file (str): The name of the file to save the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        eigenvalues,
        bins=bins,  # Regla de Freedman–Diaconis
        density=True,  # Normaliza el área a uno
        color=color,
        edgecolor="black",
        histtype = "bar",
        linewidth=0.8,
        alpha=0.95
    )

    ax.set_xlabel("Eigenvalues")
    ax.set_ylabel("Density")
    ax.set_title(title)

    # Cuadrícula sutil
    ax.grid(axis="y", alpha=0.2, linewidth=0.7)
    ax.set_axisbelow(True)

    # Limpiar bordes innecesarios
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    fig.savefig(save_file, bbox_inches="tight")

    plt.show()

# TODO: Implement the unfolding of each region of the spectrum independently and plot the histogram of the unfolded
#  eigenvalues in each region. Use some polinomial fitting to unfold the spectrum (it can be a "typical" polinomial
#  function or some special polinomial function like a Hermite polynomial).

def unfold_spectrum(eigenvalues: np.ndarray, eta: np.ndarray, deg: int = 5, base: str = "hermite") -> dict:
    """Unfold the eigenvalue spectrum using a polynomial fit to the cumulative distribution function and verified that
    the fitted polynomial is monotonic.

    Parameters:
    eigenvalues (np.ndarray): The eigenvalues in ascending order to unfold.
    eta (np.ndarray): The eigenvalues to unfold.
    deg (int): The degree of the polynomial to unfold.
    base (str): The type of polynomial to use for unfolding. Options are "hermite", "legendre", "laguerre", "chebyshev",
    or "Monomial".

    Returns:
    dict: A dictionary containing the unfolded eigenvalues and the fitted polynomial coefficients.
    """
    basis = {
        "hermite": np.polynomial.hermite.Hermite,
        "legendre": np.polynomial.legendre.Legendre,
        "laguerre": np.polynomial.laguerre.Laguerre,
        "chebyshev": np.polynomial.chebyshev.Chebyshev,
        "monomial": np.polynomial.Polynomial
    }

    if base.lower() not in basis:
        raise ValueError(f"Invalid base '{base}'. Choose from {list(basis.keys())}.")

    poly_class = basis[base.lower()]

    # Fit a polynomial to the cumulative distribution function
    poly_fit = poly_class.fit(eigenvalues, eta, deg=deg)  #  window=[-1.0, 1.0], domain is rescaled internally

    # Smooth density rho(E) = d eta / dE. It must be strictly positive on the whole range of eigenvalues.

    # Derive the polynomial to check for monotonicity
    poly_derivative = poly_fit.deriv(1)

    # Check if the derivative is non-negative over the range of eigenvalues
    eigenvalue_range = np.linspace(eigenvalues[0], eigenvalues[-1], 5000)
    derivative_values = poly_derivative(eigenvalue_range)
    min_derivative = float(np.min(derivative_values))

    # Evaluate the polynomial at the eigenvalues to get the unfolded eigenvalues
    unfolded_eigenvalues = poly_fit(eigenvalues)

    # Evaluate the spacings as a second check for monotonicity
    spacings = np.diff(unfolded_eigenvalues)

    if min_derivative < 0 or np.any(spacings <= 0):
        raise ValueError(f"The fitted polynomial is not monotonic in the range [{eigenvalues[0]:.2f}, {eigenvalues[-1]:.2f}], "
                         f"minimum derivative value is: {min_derivative:.4e} \t minimum spacing is: {np.min(spacings):.4e} "
                         f"\nConsider using a different degree or base "
                         f"for the polynomial.")

    return {
        "polynomial_fit": poly_fit,
        "coefficients": poly_fit.convert().coef, # monomial coefficients in the original variable E
        "unfolded_eigenvalues": unfolded_eigenvalues,
        "mean_spacing": float(np.mean(spacings)),  # should be very close to 1
        "min_derivative": min_derivative,
        "residuals": eta - unfolded_eigenvalues  # N(E_i) - N_smooth(E_i): should oscillate around 0, no trend
    }

# TODO: Using the unfolded spectrum of each region, create P(s) and P(r): the distribution of the spacings and ratios of
#  consecutive level spacings. Plot the histogram of P(s) and P(r) for each region and compare it with the theoretical
#  distributions

# Theoretical distributions for P(s) and P(r) for different ensembles (GOE, GUE, GSE) can be computed using the Wigner
# surmise and the spacing ratio distribution.

def wigner_surmise_spacing_distribution(s: np.ndarray, beta: int = 1) -> np.ndarray:
    """Compute the Wigner surmise for the spacing distribution P(s) for a given beta.

    Parameters:
    s (np.ndarray): The spacings to evaluate the distribution at.
    beta (int): The Dyson index (0 for Poisson, 1 for GOE, 2 for GUE, 4 for GSE).

    Returns:
    np.ndarray: The values of the Wigner surmise distribution at the given spacings.
    """
    if beta == 0:
        return np.exp(-s)
    elif beta == 1:
        return (np.pi / 2.0) * s * np.exp(- (np.pi / 4.0) * s**2)
    elif beta == 2:
        return (32.0 / np.pi**2) * s**2 * np.exp(- (4.0 / np.pi) * s**2)
    elif beta == 4:
        return (2**18 / (3**6 * np.pi**3)) * s**4 * np.exp(- (64.0 / (9.0 * np.pi)) * s**2)
    else:
        raise ValueError("Invalid beta value. Choose from {0, 1, 2, 4}.")

def spacing_ratio_distribution(r: np.ndarray, beta: int = 1) -> np.ndarray:
    """Compute the distribution of the ratio of consecutive level spacings P(r) for a given beta in the interval [0, infinity]
    (Atas, Bogomolny, Giraud & Roux, PRL 110, 084101, 2013):

    Parameters:
    r (np.ndarray): The ratios to evaluate the distribution at.
    beta (int): The Dyson index (0 for Poisson, 1 for GOE, 2 for GUE, 4 for GSE).

    Returns:
    np.ndarray: The values of the spacing ratio distribution at the given ratios.
    """
    if beta == 0:
        return 1.0 / ((1.0 + r) ** 2)
    elif beta == 1:
        z_beta = 8.0 / 27.0
    elif beta == 2:
        z_beta = (4.0 * np.pi) / (81.0 * np.sqrt(3.0))
    elif beta == 4:
        z_beta = (64.0 * np.pi) / (729.0 * np.sqrt(3.0))
    else:
        raise ValueError("Invalid beta value. Choose from {0, 1, 2, 4}.")
    return (1.0 / z_beta) * ((r + r**2) ** beta) / ((1.0 + r + r**2) ** (1.0 + 1.5 * beta))

def folded_ratio_distribution(r: np.ndarray, beta: int = 1) -> np.ndarray:
    """Compute the folded ratio distribution P(tilde{r}) for a given beta in the interval [0, 1]
    r_tilde_n = min(s_{n+1}, s_n) / max(s_{n+1}, s_n) (Oganesyan & Huse, PRB 75, 155111, 2007):
        P_tilde(r_tilde) = 2 * P(r_tilde) for r_tilde in [0, 1].

    Parameters:
    r (np.ndarray): The ratios to evaluate the distribution at.
    beta (int): The Dyson index (0 for Poisson, 1 for GOE, 2 for GUE, 4 for GSE).

    Returns:
    np.ndarray: The values of the folded ratio distribution at the given ratios.
    """
    return 2.0 * spacing_ratio_distribution(r, beta=beta)

# Numerical distributions

def get_spacing_distribution(eigenvalues: np.ndarray) -> np.ndarray:
    """Compute the normalized spacings between consecutive eigenvalues.

    Parameters:
    eigenvalues (np.ndarray): The eigenvalues in ascending order.

    Returns:
    np.ndarray: The normalized spacings between consecutive eigenvalues.
    """
    spacings = np.diff(eigenvalues)
    #spacings / np.mean(spacings)   # It is not necessary cause the unfolding, check that mean value of spacing is 1
    return spacings

def get_ratio_distribution(spacings: np.ndarray) -> np.ndarray:
    """Compute the folded ratios of consecutive spacings between eigenvalues r_tilde = min(s_{n+1}, s_n) / max(s_{n+1}, s_n).

    Parameters:
    spacings (np.ndarray): The spacings between consecutive eigenvalues.

    Returns:
    np.ndarray: The folded ratios of consecutive spacings between eigenvalues.
    """

    ratios = spacings[1:] / spacings[:-1] # Interval [0, infinity]
    # Compute the folded ratios of consecutive spacings r_tilde = min(s_{n+1}, s_n) / max(s_{n+1}, s_n)
    folded_ratios = np.minimum(ratios, 1.0 / ratios)  # Interval [0, 1]

    return folded_ratios

def plot_Ps_distribution(data: np.ndarray, title: str, bins: int = 50,
                      color: str = "blue", beta: int = 1, save_file = "ps.pdf") -> None:
    """Plot the histogram of the data and overlay the theoretical distribution.

    Parameters:
    data (np.ndarray): The data to plot.
    theoretical_func (callable): The theoretical distribution function to overlay.
    title (str): The title of the plot.
    bins (int): The number of bins for the histogram.
    color (str): The color of the histogram.
    beta (int): The Dyson index for the theoretical distribution.
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        data,
        bins=bins,  # Regla de Freedman–Diaconis
        density=True,  # Normaliza el área a uno
        color=color,
        edgecolor="black",
        histtype = "step",
        linewidth=0.8,
        alpha=0.9,
        label="Numerical",
    )

    ax.set_xlabel("Spacings")
    ax.set_ylabel("P(s)")
    ax.set_title(title)

    # Cuadrícula sutil
    ax.grid(axis="y", alpha=0.2, linewidth=0.7)
    ax.set_axisbelow(True)

    # Limpiar bordes innecesarios
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    # Generate points for the theoretical distribution
    x = np.linspace(np.min(data), np.max(data), 1000)
    y = wigner_surmise_spacing_distribution(x, beta=beta)
    # Comparison with Poisson distribution (integrable)
    z = wigner_surmise_spacing_distribution(x, beta=0)

    ax.plot(x, y, 'b-', lw=2, label="Theoretical")
    ax.plot(x, z, 'g-', lw=2, label="Poisson")
    ax.legend()

    fig.savefig(save_file, bbox_inches="tight")

    plt.show()

def plot_Pr_distribution(data: np.ndarray, title: str, bins: int = 50,
                      color: str = "blue", beta: int = 1, save_file = "pr.pdf") -> None:
    """Plot the histogram of the data and overlay the theoretical distribution.

        Parameters:
        data (np.ndarray): The data to plot.
        theoretical_func (callable): The theoretical distribution function to overlay.
        title (str): The title of the plot.
        bins (int): The number of bins for the histogram.
        color (str): The color of the histogram.
        beta (int): The Dyson index for the theoretical distribution.
        """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        data,
        bins=bins,  # Regla de Freedman–Diaconis
        density=True,  # Normaliza el área a uno
        color=color,
        edgecolor="black",
        histtype="step",
        linewidth=0.8,
        alpha=0.9,
        label="Numerical",
    )

    ax.set_xlabel("Folded Ratios")
    ax.set_ylabel("P(r)")
    ax.set_title(title)

    # Cuadrícula sutil
    ax.grid(axis="y", alpha=0.2, linewidth=0.7)
    ax.set_axisbelow(True)

    # Limpiar bordes innecesarios
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    # Generate points for the theoretical distribution
    x = np.linspace(np.min(data), np.max(data), 1000)
    y = folded_ratio_distribution(x, beta=beta)
    # Comparison with Poisson distribution (integrable)
    z = folded_ratio_distribution(x, beta=0)

    ax.plot(x, y, 'b-', lw=2, label="Theoretical")
    ax.plot(x, z, 'g-', lw=2, label="Poisson")
    ax.legend()

    fig.savefig(save_file, bbox_inches="tight")

    plt.show()


# TODO: Evaluate the Delta_3 statistic for each unfolded region and plot it. Compare the results with the theoretical
#  predictions for GOE, GUE, and Poisson.

def theoretical_delta3(L_values: np.ndarray, beta: int = 1) -> np.ndarray:
    """Asymptotic (L >> 1) theoretical Delta_3(L) for a given Dyson index (Mehta, Random Matrices):

       beta = 0 (Poisson): L / 15                                              (exact for all L)
       beta = 1 (GOE)    : (1/pi^2)      [ ln(2 pi L) + gamma - 5/4 - pi^2/8 ]
       beta = 2 (GUE)    : (1/(2 pi^2))  [ ln(2 pi L) + gamma - 5/4 ]
       beta = 4 (GSE)    : (1/(4 pi^2))  [ ln(4 pi L) + gamma - 5/4 + pi^2/8 ]

   The RMT formulas are asymptotic: they are accurate for L >~ 2-3 and can even become negative for L << 1, where
   every spectrum behaves as Poisson, Delta_3 ~ L/15.

   Parameters:
   L_values (np.ndarray): Lengths (in units of the mean level spacing).
   beta (int): Dyson index (0 for Poisson, 1 for GOE, 2 for GUE, 4 for GSE).

   return: The value of the theoretical Delta_3 statistic for the given array of lengths.
    """
    L = np.asarray(L_values, dtype=np.float64)

    if beta == 0:
        return L / 15.0
    elif beta == 1:
        return (np.log(2.0 * np.pi * L) + EULER_GAMMA - 5.0 / 4.0 - np.pi ** 2 / 8.0) / np.pi ** 2
    elif beta == 2:
        return (np.log(2.0 * np.pi * L) + EULER_GAMMA - 5.0 / 4.0) / (2.0 * np.pi ** 2)
    elif beta == 4:
        return (np.log(4.0 * np.pi * L) + EULER_GAMMA - 5.0 / 4.0 + np.pi ** 2 / 8.0) / (4.0 * np.pi ** 2)
    else:
        raise ValueError("Invalid beta value. Choose from {0, 1, 2, 4}.")

def delta3_one_window(local_x: np.ndarray, L: float) -> float:
    """Evaluate the Delta_3 statistic using Bohigas–Giannoni closed formula for a single symmetrical window [-L/2,L/2]
    that contains the unfolded eigenvalues.

    Parameters:
    local_x (np.ndarray): The unfolded eigenvalues in the given window.
    L (float): The length of the window to evaluate Delta_3.

    Returns:
    float: The value of the Delta_3 statistic for the given window.
    """
    n = len(local_x)
    if n == 0:
        return 0.0
    #if n < 2:
    #    raise ValueError("At least two unfolded eigenvalues are required to compute Delta_3.")

    j = np.arange(1, n + 1, dtype=np.float64)
    S1 = np.sum(local_x)
    S2 = np.sum(local_x ** 2)
    Sw = np.sum((n - 2.0 * j + 1.0) * local_x)

    d3 = (
            (n ** 2) / 16.0
            - (S1 ** 2) / (L ** 2)
            + (1.5 * n * S2) / (L ** 2)
            - 3.0 * (S2 ** 2) / (L ** 4)
            + Sw / L
    )

    if d3 < 0:
        print(f"Warning: Delta_3 statistic is negative ({d3:.4e}). Setting it to zero.")

    return float(max(0.0, d3))

def evaluate_delta3_statistic(unfolded_eigenvalues: np.ndarray, L_values: np.ndarray, num_windows: int = 250) -> np.ndarray:
    """Evaluate the Delta_3 statistic for a given set of unfolded eigenvalues and lengths by averaging over multiple windows
    [alpha, alpha + L] uniformly distributed in the range of the unfolded eigenvalues.

    Parameters:
    unfolded_eigenvalues (np.ndarray): The unfolded eigenvalues in ascending order.
    L_values (np.ndarray): The lengths to evaluate the Delta_3 statistic at.
    num_windows (int): The number of windows to average over for each length L.

    Returns:
    np.ndarray: The values of the Delta_3 statistic at the given lengths.
    """
    delta3_values = []
    emin, emax = unfolded_eigenvalues[0], unfolded_eigenvalues[-1]
    interval_len = emax - emin
    delta3_prom = np.zeros_like(L_values, dtype=np.float64)

    for idx_L, L in enumerate(L_values):
        if L >= interval_len * 0.9:
            delta3_prom[idx_L] = np.nan
            continue

        # Initial positions (alpha) within the segment (leaving a small margin of 1 level)
        alphas = np.linspace(emin + 0.5, emax - L - 0.5, num_windows)

        # Binary search to find the indices of the unfolded eigenvalues that fall within each window [alpha, alpha + L]
        idx_ini = np.searchsorted(unfolded_eigenvalues, alphas, side="left")
        idx_fin = np.searchsorted(unfolded_eigenvalues, alphas + L, side="right")

        accumulator = 0.0
        for k in range(num_windows):
            center = alphas[k] + 0.5 * L
            x_local = unfolded_eigenvalues[idx_ini[k]: idx_fin[k]] - center
            accumulator += delta3_one_window(x_local, L)

        delta3_prom[idx_L] = accumulator / num_windows

    return delta3_prom

def plot_delta3_statistic(L_values: np.ndarray, delta3_values: np.ndarray, beta: int = 1,
                          title: str = "Delta_3 Statistic", color: str = "blue",
                          save_file = "d3.pdf") -> None:
    """Plot the Delta_3 statistic along with the theoretical prediction for a given beta.

    Parameters:
    L_values (np.ndarray): The lengths to evaluate the Delta_3 statistic at.
    delta3_values (np.ndarray): The computed values of the Delta_3 statistic.
    beta (int): The Dyson index for the theoretical prediction.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(L_values, delta3_values, 'o-', label="Numerical", color=color)

    # Compute theoretical Delta_3 values
    #theoretical_values = theoretical_delta3(L_values, beta=beta)
    #plt.plot(L_values, theoretical_values, 'r--', label="Theoretical", color="red")

    plt.title(title)
    plt.xlabel("L")
    plt.ylabel("Delta_3(L)")
    plt.legend()
    plt.grid()

    plt.savefig(save_file, bbox_inches="tight" )

    plt.show()

# TODO: Evaluate the Number Variance \Sigma^2(L) for each unfolded region and plot it. Compare the results with the
#  theoretical predictions for GOE, GUE and Poisson

# ----------------------------------------------------------------------------------------------------------------------
# Theoretical Sigma^2
# ----------------------------------------------------------------------------------------------------------------------
def theoretical_number_variance(L_values: np.ndarray, beta: int = 1) -> np.ndarray:
    """Asymptotic (L >> 1) theoretical number variance Sigma^2(L) for a given Dyson index (Mehta, Random Matrices):

        beta = 0 (Poisson): L                                                    (exact for all L)
        beta = 1 (GOE)    : (2/pi^2)     [ ln(2 pi L) + gamma + 1 - pi^2/8 ]
        beta = 2 (GUE)    : (1/pi^2)     [ ln(2 pi L) + gamma + 1 ]
        beta = 4 (GSE)    : (1/(2 pi^2)) [ ln(4 pi L) + gamma + 1 + pi^2/8 ]

    Valid for L >~ 1 (for L << 1 every spectrum has Sigma^2 ~ L).
    """
    L = np.asarray(L_values, dtype=np.float64)

    if beta == 0:
        return L
    elif beta == 1:
        return (2.0 / np.pi ** 2) * (np.log(2.0 * np.pi * L) + EULER_GAMMA + 1.0 - np.pi ** 2 / 8.0)
    elif beta == 2:
        return (1.0 / np.pi ** 2) * (np.log(2.0 * np.pi * L) + EULER_GAMMA + 1.0)
    elif beta == 4:
        return (1.0 / (2.0 * np.pi ** 2)) * (np.log(4.0 * np.pi * L) + EULER_GAMMA + 1.0 + np.pi ** 2 / 8.0)
    else:
        raise ValueError("Invalid beta value. Choose from {0, 1, 2, 4}.")


# ----------------------------------------------------------------------------------------------------------------------
# Numerical Sigma^2
# ----------------------------------------------------------------------------------------------------------------------
def number_variance_one_window(unfolded_eigenvalues: np.ndarray, alpha: np.ndarray, L: float) -> np.ndarray:
    """Number of unfolded eigenvalues n(alpha, L) inside the windows [alpha, alpha + L], for ALL the starting points
    in `alpha` at once (binary search, no Python loop).

    Parameters:
    unfolded_eigenvalues (np.ndarray): The unfolded eigenvalues in ascending order.
    alpha (np.ndarray): Starting points of the windows.
    L (float): Length of the windows.

    Returns:
    np.ndarray: The number of levels in each window.
    """
    idx_ini = np.searchsorted(unfolded_eigenvalues, alpha, side="left")
    idx_fin = np.searchsorted(unfolded_eigenvalues, alpha + L, side="right")
    return (idx_fin - idx_ini).astype(np.float64)


def evaluate_number_variance(unfolded_eigenvalues: np.ndarray, L_values: np.ndarray, num_windows: int = 250,
                             subtract_mean: bool = True) -> np.ndarray:
    """Evaluate the number variance Sigma^2(L) = < (n(alpha, L) - L)^2 > by averaging over windows [alpha, alpha + L]
    uniformly distributed in the range of the unfolded eigenvalues (same scheme as evaluate_delta3_statistic).

    Parameters:
    unfolded_eigenvalues (np.ndarray): The unfolded eigenvalues in ascending order.
    L_values (np.ndarray): The lengths to evaluate Sigma^2 at.
    num_windows (int): Number of windows for each L.
    subtract_mean (bool): If True, use the variance around the empirical mean count (np.var), which is robust to a small
        error in the mean density (e.g. mean spacing 1.001). If False, use exactly < (n - L)^2 >, which is the
        definition and assumes that the unfolding is perfect.

    Returns:
    np.ndarray: Values of Sigma^2 at the given lengths (NaN if L is too large for the spectrum).
    """
    emin, emax = unfolded_eigenvalues[0], unfolded_eigenvalues[-1]
    interval_len = emax - emin
    sigma2 = np.full(len(L_values), np.nan, dtype=np.float64)

    for idx_L, L in enumerate(L_values):
        if L >= interval_len * 0.9:
            continue
        alphas = np.linspace(emin + 0.5, emax - L - 0.5, num_windows)
        n = number_variance_one_window(unfolded_eigenvalues, alphas, L)
        sigma2[idx_L] = np.var(n) if subtract_mean else np.mean((n - L) ** 2)

    return sigma2


# ----------------------------------------------------------------------------------------------------------------------
# Exact theory for Sigma^2 and Delta_3 (GOE and GUE) by numerical integration of the two-point function
# ----------------------------------------------------------------------------------------------------------------------
# The asymptotic formulas (theoretical_number_variance / theoretical_delta3) are only valid for L >> 1: the GOE Delta_3
# one is even negative for L < 1 and still ~4% off at L = 10. Here the exact curves are computed from the cluster
# function Y2(r) (R2 = 1 - Y2):
#       GUE: Y2 = s(r)^2,                     s(r) = sin(pi r) / (pi r)
#       GOE: Y2 = s(r)^2 + s'(r) * [ 1/2 - Si(pi r)/pi ]
#       Sigma^2(L)  = L - 2 * int_0^L (L - r) Y2(r) dr
#       Delta_3(L)  = (2 / L^4) * int_0^L (L^3 - 2 L^2 r + r^3) Sigma^2(r) dr
_EXACT_CACHE = {}


def _exact_curves(beta: int, L_max: float, step: float = 0.002):
    key = (beta, L_max, step)
    if key in _EXACT_CACHE:
        return _EXACT_CACHE[key]

    r = np.arange(0.0, L_max + step / 2, step)
    s = np.sinc(r)  # numpy: sinc(x) = sin(pi x) / (pi x)
    if beta == 2:
        Y2 = s ** 2
    elif beta == 1:
        ds = np.zeros_like(r)  # s'(r) = [cos(pi r) - s(r)] / r ,  s'(0) = 0
        ds[1:] = (np.cos(np.pi * r[1:]) - s[1:]) / r[1:]
        Y2 = s ** 2 + ds * (0.5 - sici(np.pi * r)[0] / np.pi)
    else:
        raise ValueError("Exact curves are implemented for beta = 1 (GOE) and beta = 2 (GUE).")

    I0 = cumulative_trapezoid(Y2, r, initial=0.0)
    I1 = cumulative_trapezoid(r * Y2, r, initial=0.0)
    sigma2 = r - 2.0 * (r * I0 - I1)

    J0 = cumulative_trapezoid(sigma2, r, initial=0.0)
    J1 = cumulative_trapezoid(r * sigma2, r, initial=0.0)
    J3 = cumulative_trapezoid(r ** 3 * sigma2, r, initial=0.0)
    delta3 = np.zeros_like(r)
    delta3[1:] = 2.0 / r[1:] ** 4 * (r[1:] ** 3 * J0[1:] - 2.0 * r[1:] ** 2 * J1[1:] + J3[1:])

    _EXACT_CACHE[key] = (r, sigma2, delta3)
    return _EXACT_CACHE[key]


def theory_number_variance(L_values: np.ndarray, beta: int, exact: bool = True) -> np.ndarray:
    """Sigma^2(L). beta = 0 Poisson (exact formula), 1 GOE, 2 GUE. exact=False uses the asymptotic formulas."""
    L = np.asarray(L_values, dtype=np.float64)
    if beta == 0 or not exact:
        return theoretical_number_variance(L, beta)
    r, sigma2, _ = _exact_curves(beta, L_max=max(60.0, float(np.ceil(np.nanmax(L))) + 1.0))
    return np.interp(L, r, sigma2)


def theory_delta3(L_values: np.ndarray, beta: int, exact: bool = True) -> np.ndarray:
    """Delta_3(L). beta = 0 Poisson (exact formula), 1 GOE, 2 GUE. exact=False uses the asymptotic formulas."""
    L = np.asarray(L_values, dtype=np.float64)
    if beta == 0 or not exact:
        return theoretical_delta3(L, beta)
    r, _, delta3 = _exact_curves(beta, L_max=max(60.0, float(np.ceil(np.nanmax(L))) + 1.0))
    return np.interp(L, r, delta3)


def mean_folded_ratio_theory(beta: int) -> float:
    """<r~> of the (Wigner-like) surmise P~(r~) = 2 P(r~) on [0, 1], by numerical integration of your
    folded_ratio_distribution. Gives 0.386 (Poisson), 0.536 (GOE), 0.603 (GUE)."""
    r = np.linspace(0.0, 1.0, 20001)
    y = r * folded_ratio_distribution(r, beta=beta)
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(r)))


def semicircle_density(E: np.ndarray, N: int, normalized: bool = False) -> np.ndarray:
    """Mean density (number of levels per unit energy) of your GOE/GUE matrices (off-diagonal <|H_ij|^2> = 1).
    Unnormalized: rho(E) = sqrt(4N - E^2) / (2 pi).  Normalized (H / sqrt(N)): rho(E) = N sqrt(4 - E^2) / (2 pi)."""
    E = np.asarray(E, dtype=np.float64)
    arg = (4.0 - E ** 2) if normalized else (4.0 * N - E ** 2)
    rho = np.sqrt(np.clip(arg, 0.0, None)) / (2.0 * np.pi)
    return N * rho if normalized else rho


if __name__ == "__main__":
    # General Plot Style
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 1.0,
        "mathtext.fontset": "dejavuserif",
    })

    N = 5000  # Dimension of the matrix
    seed = 42  # Set a seed for reproducibility
    rng = np.random.default_rng(seed)
    saved_data = True
    data_file = f"data_GOE_GUE_N{N}.npz"

    if not saved_data:
        print(f"Generating a {N}x{N} GOE and GUE matrix...")
        goe_matrix = get_goe_matrix(N, rng=rng)
        gue_matrix = get_gue_matrix(N, rng=rng)
        print(f"Diagonalizing the GOE matrix...")
        goe_eigenvalues, goe_eigenvectors = get_spectrum(goe_matrix, name="GOE matrix")
        print(f"Diagonalizing the GUE matrix...")
        gue_eigenvalues, gue_eigenvectors = get_spectrum(gue_matrix, name="GUE matrix")

        # saved data for reproducibility
        print(f"Data saved in memory en store in 'data_GOE_GUE_N{N}.npz'.")
        np.savez(
            data_file,
            evals_goe=goe_eigenvalues,
            evecs_goe=goe_eigenvectors,
            evals_gue=gue_eigenvalues,
            evecs_gue=gue_eigenvectors,
        )
    else:
        data = np.load(data_file)
        goe_eigenvalues, gue_eigenvalues = data["evals_goe"], data["evals_gue"]
        goe_eigenvectors, gue_eigenvectors = data["evecs_goe"], data["evecs_gue"]
        print(f"Data recovered from {data_file}")

    # Separate the GOE and GUE spectrum into three regions and plot histograms for each region
    goe_regions = separate_spectrum_regions(goe_eigenvalues, c=10)
    gue_regions = separate_spectrum_regions(gue_eigenvalues, c=10)

    # Plot of the full spectrum before unfolding
    plot_spectrum(goe_eigenvalues, title="GOE Complete Eigenvalue Spectrum", bins=40, color="green",
                  save_file="GOE/complete_eigenv_spectrum.pdf")
    plot_spectrum(gue_eigenvalues, title="GUE Complete Eigenvalue Spectrum", bins=40, color="green",
                  save_file="GUE/complete_eigenv_spectrum.pdf")

    # Separate the GOE and GUE spectrum into three regions and plot histograms for each region before unfolding
    for region_name, region_data in goe_regions.items():
        plot_spectrum(region_data["eigenvalues"], title=f"GOE folded Eigenvalues - {region_name.capitalize()} Region",
                      save_file=f"GOE/{region_name}_GOE_spectrum.pdf",
                      bins=40, color="blue")

    for region_name, region_data in gue_regions.items():
        plot_spectrum(region_data["eigenvalues"], title=f"GUE folded Eigenvalues - {region_name.capitalize()} Region",
                      save_file=f"GUE/{region_name}_GUE_spectrum.pdf",
                      bins=40, color="blue")

    # Goe Analysis
    for region_name, region_data in goe_regions.items():
        # Goe unfolded eigenvalues for each region
        results = unfold_spectrum(region_data["eigenvalues"], region_data["eta"])
        unfolded_eigenvalues = results["unfolded_eigenvalues"]

        plot_spectrum(unfolded_eigenvalues, title=f"GOE Unfolded Eigenvalues - {region_name.capitalize()} Region",
                      save_file=f"GOE/{region_name}_GOE_unfolded_spectrum.pdf",
                      bins=40, color="purple")

        # Evaluate
        spacings = get_spacing_distribution(unfolded_eigenvalues)
        folded_ratios = get_ratio_distribution(spacings)

        # Plot spacing distribution for each region
        plot_Ps_distribution(spacings, beta=1, bins=40, color="orange",
                             save_file=f"GOE/GOE_{region_name.capitalize()}_Ps_distribution.pdf",
                             title=f"GOE Spacings Distribution - {region_name.capitalize()} Region")

        plot_Pr_distribution(folded_ratios, beta=1, bins=40, color="purple",
                             save_file=f"GOE/GOE_{region_name.capitalize()}_Pr_distribution.pdf",
                             title=f"GOE Unfolded Ratios - {region_name.capitalize()} Region")

        # Calculate Delta_3 Statistic
        L_vals = np.linspace(0.5, 50.0, 100)
        delta3_prom = evaluate_delta3_statistic(unfolded_eigenvalues, L_vals, num_windows=250)
        plot_delta3_statistic(L_vals, delta3_prom, title=f"GOE Delta 3 Statistic - {region_name.capitalize()} Region",
                              save_file=f"GOE/GOE_{region_name.capitalize()}_Delta3_statistic.pdf")

    # GUE Analysis
    for region_name, region_data in gue_regions.items():
        results = unfold_spectrum(region_data["eigenvalues"], region_data["eta"])
        unfolded_eigenvalues = results["unfolded_eigenvalues"]

        plot_spectrum(unfolded_eigenvalues, title=f"GUE Unfolded Eigenvalues - {region_name.capitalize()} Region",
                      save_file=f"GUE/{region_name}_GUE_unfolded_spectrum.pdf",
                      bins=40, color="purple")

        # Evaluate
        spacings = get_spacing_distribution(unfolded_eigenvalues)
        folded_ratios = get_ratio_distribution(spacings)

        # Plot spacing distribution for each region
        plot_Ps_distribution(spacings, beta=2, bins=40, color="orange",
                             save_file=f"GUE/GUE_{region_name.capitalize()}_Ps_distribution.pdf",
                             title=f"GUE Spacings Distribution - {region_name.capitalize()} Region")

        plot_Pr_distribution(folded_ratios, beta=2, bins=40, color="purple",
                             save_file=f"GUE/GUE_{region_name.capitalize()}_Pr_distribution.pdf",
                             title=f"GUE Unfolded Ratios - {region_name.capitalize()} Region")

        L_vals = np.linspace(0.5, 50.0, 100)
        delta3_prom = evaluate_delta3_statistic(unfolded_eigenvalues, L_vals, num_windows=250)
        plot_delta3_statistic(L_vals, delta3_prom, title=f"GUE Delta 3 Statistic - {region_name.capitalize()} Region",
                              save_file=f"GUE/GUE_{region_name.capitalize()}_Delta3_statistic.pdf")





























