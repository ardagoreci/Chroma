"""Chroma learns to reverse a correlated diffusion process which respects the chain and density constraints that
almost all structures satisfy. This allows the model to focus most of its capacity on non-trivial dependencies. This
section implements the polymer-structured diffusions: multivariate Gaussian distributions for protein structures that
(i) are SO(3) invariant, (ii) enforce protein chain and radius of gyration statistics, and (iii) can be computed in
linear time. This module implements various functions for the polymer-structured diffusion and utility functions that
compute key metrics such as radius of gyration."""

# Dependencies
import math
import jax
import jax.numpy as jnp


@jax.jit
def diffuse(noise, coordinates, timestep, rg_confined=True, mode='pile-of-globs'):
    """This method diffuses the coordinates of the polymer up to the given time. The diffusion
    is integrated forward in time according to the following formula:
    x_t = sqrt(alpha_t)*x0 + sqrt(1-alpha_t)*matmul(R,z) where z follows multivariate normal.
    Args:
        noise: noise (z) that follows z ~ N(0, I) of shape [N_atoms, 3]
        coordinates: a matrix of size [N, B ,3] where N is the number of residues, B is the number of backbone atoms
        timestep: the timestep between [0, 1] where 0 is the data and 1 is noise.
        rg_confined: whether to use to Radius of Gyration confined (Covariance model #2) in Chroma paper, default True
        mode: the rg_confined mode has a 'pile-of-globs' and 'glob-of-globs' mode
    Returns:
        noised coordinates of shape [N, B, 3]
    TODO: get the rg_confined and mode parameters from the config file
    TODO: implement mean deflation
    (unit-tested)"""
    # Reshape to (N_atoms, 3)
    N, B, _ = coordinates.shape
    x0 = coordinates.reshape((N * B, 3))

    # Get alpha_t from diffusion schedule
    alpha_t = log_snr_schedule(timestep)

    if rg_confined:
        # Scaling a and computing b
        a = 2.1939  # # 'segment length' = 3.8 Angstroms, which becomes scaled with sqrt(3)
        b = compute_b(N, B, a)  # N (res) and B (atoms per res) is input to compute_b function
        rz = rg_confined_covariance(noise, a=a, b=b)
    else:  # use ideal-chain diffusion
        radius_g = compute_radius_of_gyration(x0)
        gamma = math.sqrt((radius_g ** 2) * 2 / N)
        rz = ideal_covariance(noise, gamma=gamma, delta=0.0)  # parameters from config file

    # Diffusion Step
    x_t = jnp.sqrt(alpha_t) * x0 + jnp.sqrt(1 - alpha_t) * rz  # * scale

    # Mean deflation operation TODO: implement this
    if rg_confined:
        # deflate_mean()
        pass
    # Reshape to [N, B, 3]
    noised_coordinates = x_t.reshape((N, B, 3))
    return noised_coordinates


# The flow here:
# I just need a pure function that returns a covariance matrix given n_atoms, a and b


def ideal_covariance(z, gamma, delta):
    """This method computes the matmul(R, z) for the ideal chain covariance model. For the formula, see section C.2
    'Covariance model #1: Ideal Chain' in Chroma Paper.
    Args:
        z: noise samples from z ~ N(0, I) of shape x0.shape
        n_atoms: number of atoms in chain
        gamma: length scale of the chain
        delta: amount of allowed translational noise about the origin
    (unit-tested)"""
    n_atoms = z.shape[0]
    cu_z = gamma * jnp.cumsum(z, axis=0)  # cumulative z
    constant_terms = delta * cu_z[0] - (1 / n_atoms) * (jnp.sum(cu_z))
    return cu_z + constant_terms


def rg_confined_covariance(z, a, b):
    """This method computes the matmul(R, z) for the Rg-confined, linear-time Polymer MVNs. For the construction of
    the R matrix, see section C.3 'Covariance model #2: Rg-confined, linear-time Polymer MVNs' in the Chroma Paper.
    Args:
        z: noise samples from z ~ N(0, I) of shape x0.shape
        a: a global scale parameter setting the 'segment length' of polymer
        b: a 'decay' parameter which sets the memory of the chain to fluctuations
    Returns:
    (unit-tested)"""
    # Construct Rg confined covariance matrix
    n_atoms = z.shape[0]
    b_vec = b ** (jnp.arange(n_atoms))  # [N_at,]

    rows = []
    v = 1 / (jnp.sqrt(1 - b ** 2))
    for i in range(len(b_vec)):
        row = jnp.flip(b_vec[:len(b_vec) - i])
        row = row.at[0].set(row[0] * v)  # Multiplying v with the first element, jnp.at for updates
        padded_row = jnp.pad(row, (0, i))
        rows.append(padded_row)
    rows.reverse()
    R_matrix = a * jnp.stack(rows, axis=0)
    rz = jnp.matmul(R_matrix, z)

    # Construct the inverse matrix
    # First and last element are edge cases, the rest is simpler
    y = jnp.array([-b, (1 + b**2), -b])
    first_row = jnp.pad(jnp.array([1, -b]), (0, n_atoms-2))
    rows = [first_row]
    for i in range(1, n_atoms-1):  # except first and last rows
        row = jnp.pad(y, (i, n_atoms-3-i))
        rows.append(row)
    last_row = jnp.flip(first_row)
    rows.append(last_row)
    inverse_covariance_matrix = jnp.stack(rows, axis=0)
    # Compute square root

    return rz, inverse_covariance_matrix


def log_snr_schedule(timestep):
    """
    This method implements a test schedule that gives a monotonically decreasing alpha_t value given a timestep. See
    Kingma et al. 2021 Args: timestep: a timestep between [0, 1]
    TODO: make sure this is the actual diffusion schedule used by Kingma et al. 2021.
     I know this is their continuous implementation of the Ho et al. 2020
     schedule, not sure if it refers to the log SNR schedule mentioned in Chroma Paper.
    """
    # jax.nn.sigmoid((timestep-0.5)*5)
    return jnp.exp(-jnp.e ** (-4) - 10 * (timestep ** 2))


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
    r = (2.0 / jnp.sqrt(3)) / (B ** 0.4)  # r is an experimentally determined prefactor with value 2.0 Angstroms but is
    # scaled to this value when n_atoms are used. Also, the segment length 'a' of 3.8 Angstroms is scaled with
    # sqrt(3) so r has to be scaled with the same number to keep b invariant
    v = 0.4  # narrow range between 0.38-0.40, taken to be 0.40 (See Tanner, 2016)
    b_effective = 3 / n_atoms + (n_atoms ** (-v)) * jnp.sqrt(
        n_atoms ** (2 * v - 2) * (n_atoms ** 2 + 9) - ((a ** 2) / (r ** 2)))
    return b_effective


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
