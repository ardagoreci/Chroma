"""Chroma learns to reverse a correlated diffusion process which respects the chain and density constraints that
almost all structures satisfy. This allows the model to focus most of its capacity on non-trivial dependencies. This
section implements the polymer-structured diffusions: multivariate Gaussian distributions for protein structures that
(i) are SO(3) invariant, (ii) enforce protein chain and radius of gyration statistics, and (iii) can be computed in
linear time. This module implements various functions for the polymer-structured diffusion and utility functions that
compute key metrics such as radius of gyration."""

# Dependencies
import jax
import jax.numpy as jnp
import numpy as np

# Constants for linear-beta diffusion schedule (Song et al. 2021)
B_MAX = 5
B_MIN = 0.1


# -----------------------------------------------------------------------------
# Operations for ideal-chain diffusion
# -----------------------------------------------------------------------------

def diffuse_ideal(noise, x0, timestep):
    """Diffuses the coordinates of a polymer up to a given time according to the ideal chain structure.
        Args:
            noise: noise (z) that follows z ~ N(0, I) of shape [N_atoms, 3]
            x0: a matrix of size [N_atoms, 3]
            timestep: the timestep between [0, 1] where 0 is the data and 1 is noise.
        Returns:
            noised coordinates of shape [N_at, 3]
        """
    rz = ideal_covariance(noise)

    # Get alpha_t from diffusion schedule
    alpha_t = get_cosine_alpha_t(timestep)
    one_m_alpha_t = get_stable_cosine_1malpha(timestep)  # numerically stable (1 - alpha_t)

    # Diffusion Step
    x_t = jnp.sqrt(alpha_t) * x0 + jnp.sqrt(one_m_alpha_t) * rz
    return x_t


def ideal_covariance(z, delta=1.0):
    """This method computes the matmul(R, z) for the ideal chain covariance model. For the formula, see section C.2
    'Covariance model #1: Ideal Chain' in Chroma Paper.
    Args:
        z: noise samples from z ~ N(0, I) of shape x0.shape [n_atoms, 3]
        delta: amount of allowed translational noise about the origin
    (unit-tested)"""
    n_atoms = z.shape[0]
    gamma = compute_gamma(n_atoms)  # length scale of the chain
    cumulative_z = gamma * jnp.cumsum(z, axis=0)  # cumulative z
    constant_terms = delta * cumulative_z[0] - (1 / n_atoms) * (jnp.sum(cumulative_z))
    rz = cumulative_z + constant_terms

    # Normalize rz to have origin at center
    r_o = jnp.mean(rz, axis=0, keepdims=True)
    return rz - r_o


def compute_expected_rg(num_residues):
    """Computes the expected radius of gyration given the number of residues in the chain.
    From Tanner et al. 2006."""
    R0 = 2.0
    v = 0.4
    return R0 * (num_residues ** v)


def compute_gamma(n_atoms):
    """Computes the gamma factor in ideal chain diffusion given an expected radius of gyration.
    And n_atoms in the polymer chain."""
    radius_of_gyration = compute_expected_rg(num_residues=n_atoms / 4)
    return jnp.sqrt(2 * (radius_of_gyration ** 2) / n_atoms)


# -----------------------------------------------------------------------------
# Operations for rg-confined diffusion
# -----------------------------------------------------------------------------


def diffuse(noise, R, x0, timestep):
    """This method diffuses the coordinates of the polymer up to the given time. The diffusion
    is integrated forward in time according to the following formula:
    x_t = sqrt(alpha_t)*x0 + sqrt(1-alpha_t)*matmul(R,z) where z follows multivariate normal.
    Args:
        noise: noise (z) that follows z ~ N(0, I) of shape [N_atoms, 3]
        R: square root of the covariance matrix
        x0: a matrix of size [N_atoms, 3]
        timestep: the timestep between [0, 1] where 0 is the data and 1 is noise.
    Returns:
        noised coordinates of shape [N_at, 3]
    TODO: implement mean deflation
    (unit-tested)"""
    # Get alpha_t from diffusion schedule
    alpha_t = get_alpha_t(timestep)
    one_m_alpha_t = get_stable_1malpha(timestep)  # numerically stable implementation of (1 - alpha_t)
    rz = jnp.matmul(R, noise)  # Compute prior rz

    # Diffusion Step
    x_t = jnp.sqrt(alpha_t) * x0 + jnp.sqrt(one_m_alpha_t) * rz
    return x_t


def rg_confined_covariance(n_atoms, a, b):
    """This method computes the matmul(R, z) for the Rg-confined, linear-time Polymer MVNs. For the construction of
    the R matrix, see section C.3 'Covariance model #2: Rg-confined, linear-time Polymer MVNs' in the Chroma Paper.
    Args:
        n_atoms: number of atoms in protein
        a: a global scale parameter setting the 'segment length' of polymer
        b: a 'decay' parameter which sets the memory of the chain to fluctuations
    Returns:
    (unit-tested)"""
    # Construct Rg confined covariance matrix
    b_vec = b ** (np.arange(n_atoms))  # [N_at,]

    rows = []
    v = 1 / (np.sqrt(1 - b ** 2))
    for i in range(len(b_vec)):
        row = np.flip(b_vec[:len(b_vec) - i])
        row[0] = row[0] * v  # Multiplying v with the first element, jnp.at for updates
        padded_row = np.pad(row, (0, i))
        rows.append(padded_row)
    rows.reverse()
    R = a * np.stack(rows, axis=0)
    R_inverse = np.linalg.inv(R)
    return R, R_inverse


def compute_b(N, B, a):
    """This method computes b given the global scaling factor a. See section C.3.2 in Chroma Paper for the derivation
    of the formula. The derivation makes a power law approximation for the radius of gyration in the form r*(N**v)
    where r is a prefactor and v is the Flory coefficient. The values of r and v are taken from Tanner (2016) in the
    paper 'Empirical power laws for the radii of gyration of protein oligomers.' The experimentally determined values
    are: prefactor r = 2 Angstroms (0.2 nanometers) and v = 0.4. Tanner (2016) showed that the v of oligomers was
    almost identical to that of monomers which is 0.4. Since this function uses n_atoms as its input, the r is rescaled
    to 1.148698355.
    Args:
        N: number of residues in the protein
        B: number of atoms per residue
        a: global scale factor that determines the 'segment length' of the polymer
    """
    # (See Tanner, 2016)
    n_atoms = N * B
    r = (2.0 / np.sqrt(3)) / (B ** 0.4)  # r is an experimentally determined prefactor with value 2.0 Angstroms but is
    # scaled to this value when n_atoms are used. Also, the segment length 'a' of 3.8 Angstroms is scaled with
    # sqrt(3) so r has to be scaled with the same number to keep b invariant
    v = 0.4  # narrow range between 0.38-0.40, taken to be 0.40 (See Tanner, 2016)
    b_effective = 3 / n_atoms + (n_atoms ** (-v)) * np.sqrt(
        n_atoms ** (2 * v - 2) * (n_atoms ** 2 + 9) - ((a ** 2) / (r ** 2)))
    return b_effective


# -----------------------------------------------------------------------------
# Operations for cosine decay schedule
# -----------------------------------------------------------------------------

def cosine_decay_f(timestep, s=0.008):
    return jnp.cos((timestep + s) / (1 + s) * jnp.pi / 2) ** 2


def cosine_decay_df_dt(timestep, s=0.008):
    """Computes derivative of f as defined in get_cosine_alpha_t."""
    x = jnp.pi * (timestep + s) / (1 + s) / 2.0
    df_dt = (-jnp.pi * jnp.sin(x) * jnp.cos(x)) / (1 + s)
    return df_dt / cosine_decay_f(0.0)  # normalize to get alpha_t derivative


def get_cosine_alpha_t(t, s=0.008):
    """Computes the alpha values according to the cosine diffusion schedule.
    The cosine schedule decays more slowly than a linear beta schedule. """
    return cosine_decay_f(t, s) / cosine_decay_f(0.0, s)


def get_cosine_beta_t(t, s=0.008):
    """Computes the beta values according to the cosine diffusion schedule."""
    beta_t = -jnp.log(cosine_decay_f(t, s))
    return jnp.clip(beta_t, a_min=0.0001, a_max=0.999)


def get_stable_cosine_1malpha(t, epsilon=1e-3):
    """Simple clipping solution for numerical stability."""
    alpha_t = jnp.clip(get_cosine_alpha_t(t), a_min=epsilon, a_max=1-epsilon)
    return 1.0 - alpha_t



# -----------------------------------------------------------------------------
# Operations for beta-linear schedule
# -----------------------------------------------------------------------------

def get_alpha_t(t):
    """Returns the alpha_t value that is equivalent to the perturbation kernel used in Song et al. 2021"""
    x = -(1 / 2) * (t ** 2) * (B_MAX - B_MIN) - t * B_MIN
    return jnp.exp(x)


def get_stable_1malpha(t):
    """Numerically stable implementation of (1-alpha_t) with the stable jnp.expm1 primitive."""
    x = -(1 / 2) * (t ** 2) * (B_MAX - B_MIN) - t * B_MIN
    return - jnp.expm1(x)


def get_beta_t(t):
    """Returns the ß(t) value that is used in the SDEs. The ß(t) schedule is the same as that used in
    Song et al. 2021."""
    return B_MIN + t * (B_MAX - B_MIN)


def SNR(t):
    return get_alpha_t(t) / get_stable_1malpha(t)


def deflate_mean(x, xi):
    """This method implements the mean-deflation operation that re-tunes the translational variance of each chain.
    Args: x: the atom coordinates (N, 3) xi: set based on pile-of-globs or glob-of-globs covariance modes.
    Pile-of-globs: set xi so that the translational variance of each chain is unity. This will cause chains to have a
    realistic radius of gyration but pile up at the origin. Glob-of-globs covariance: set xi per chain by solving for
    the translational variance that also implements the correct whole-complex Rg scaling as a function of the number
    of residues.
    """
    pass


def compute_radius_of_gyration(coordinates):
    """This method computes the radius of gyration given the coordinates of all atoms.
    Formula from Tanner, 2016:
    Rg = sqrt(sum_over_i(r_i - r_o)/n_atoms) where r_o = sum_over_i(r_i) / n_atoms
    Args:
        coordinates: a matrix of size (N, 3) where N is the number of atoms in complex
    (unit-tested)"""
    r_o = jnp.average(coordinates, axis=0)
    squared_residuals = jnp.sum(((coordinates - r_o) ** 2), axis=1)
    rg = jnp.sqrt(jnp.average(squared_residuals, axis=0))
    return rg
