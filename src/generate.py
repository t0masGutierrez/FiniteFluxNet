from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr


def config_domain(min_space: float, max_space: float, nx: int, min_time: float, max_time: float, nt: int) -> tuple[jax.Array, jax.Array]:
    """
    configure spatiotemporal domain

    parameters
    ----------
    min_space: float
        minimum spatial coordinate
    max_space: float
        maximum spatial coordinate
    nx: int
        number of spatial coordinates
    min_time: float
        minimum temporal coordinate
    max_time: float
        maximum temporal coordinate
    nt: int
        number of temporal coordinates
    """
    x = jnp.linspace(min_space, max_space, nx, endpoint=False)
    t = jnp.linspace(min_time, max_time, nt, endpoint=True)
    return x, t


def covariance(x: jax.Array, *, sigma: float = 1.0, ell: float = 1.0, eps=1e-5) -> jax.Array:
    """
    compute the covariance between u(x_i) and u(x_j)
    symmetric: K = K^T
    positive semidefinite: x^T * K * x >= 0

    parameters
    ----------
    x: jax.Array
        spatial coordinates
    sigma: float
        standard deviation
    ell: float
        speed of correlation decay
    eps: float
        safety factor for numerical stability

    returns
    -------
    L: jax.Array
        lower triangular covariance matrix
    """
    distance = x[:, None] - x[None, :]

    # periodic gaussian kernel
    K = sigma**2 * jnp.exp(-1 * (1 - jnp.cos(2 * jnp.pi * distance)) / ell ** 2)

    # symmetrize covariance matrix
    K = 0.5 * (K + K.T)

    # regularize covariance matrix
    I = jnp.eye(K.shape[0])
    K += eps * I

    # factorize covariance matrix
    L = jnp.linalg.cholesky(K)
    return L


def initialize(x: jax.Array, L: jax.Array, key: jax.Array, *, n: int) -> jax.Array:
    """
    generate initial conditions

    parameters
    ----------
    x: jax.Array
        spatial coordinates
    L: jax.Array
        lower triangular covariance matrix
    key: jax.Array
        random number generator
    n: int
        number of initial conditions
    
    returns
    -------
    u_0: jax.Array
        initial conditions
    """
    nx = len(x)
    u_0 = np.empty((n, nx))
    for i in range(n):
        key, subkey = jr.split(key)
        z = jr.normal(subkey, (nx,))
        u_0[i] = jnp.matmul(L, z)
    u_0 = jnp.array(u_0)
    return u_0


def sample_coeff(domain: list, key: jax.Array, *, n: int) -> jax.Array:
    """
    sample coefficients from uniform probability distribution

    parameters
    ----------
    domain: list
        minimum and maximum value of coefficients
    key: jax.Array
        random number generator
    n: int
        number of coefficients
    
    returns
    -------
    jax.Array
        coefficients
    """
    return jr.uniform(key, (n,), minval=domain[0], maxval=domain[1])


def flux(u: jax.Array, coeffs: jax.Array) -> jax.Array:
    """
    compute flux

    parameters
    ----------
    u: jax.Array
        solution
    coeffs: jax.Array
        coefficients

    returns
    -------
    jax.Array
        flux
    """
    return coeffs[0] * u ** 3 + coeffs[1] * u ** 2 + coeffs[2] * u


def speed(u: jax.Array, coeffs: jax.Array) -> jax.Array:
    """
    compute speed

    parameters
    ----------
    u: jax.Array
        solution
    coeffs: jax.Array
        coefficients
    
    returns
    ------
    jax.Array
        speed
    """
    J = jax.jacrev(lambda value: flux(value, coeffs))(u)
    return jnp.abs(J)


def max_speed(speed: jax.Array, *, eps: float = 1e-8) -> jax.Array:
    """
    compute max speed

    parameters
    ----------
    speed: jax.Array
        rate of change of position
    eps: float
        safety factor for numerical stability
    
    returns
    ------
    jax.Array
        maximum rate of change of position
    """
    max_speed = jnp.max(speed)
    return jnp.maximum(max_speed, eps)


def time_step(x: jax.Array, dx: float, max_speed: jax.Array, *, cfl: float = 0.5) -> jax.Array:
    """
    compute time step

    parameters
    ----------
    max_speed: jax.Array
        maximum rate of change of position
    dx: float
        grid spacing
    cfl: float
        safety factor for limiting the size of time step (courant-friedrachs-lewy)
    
    returns
    ------
    jax.Array
        time step
    """
    return cfl * dx / max_speed


def neighbor(u: jax.Array, *, i: int, n: int) -> tuple[jax.Array, jax.Array]:
    """
    compute solution at neighboring spatial coordinate

    parameters
    ----------
    u: jax.Array
        solution
    i: int
        index of center solution
    n: int
        number of neighbors
    
    returns
    ------
    before_sol: float
        solution before center solution
    after_sol: float
        solution after center solution
    """
    before_index = (i - n) % len(u)
    after_index = (i + n) % len(u)
    before_sol = u[before_index]
    after_sol = u[after_index]
    return before_sol, after_sol


def numerical_flux(u: jax.Array, coeffs) -> jax.Array:
    """
    compute numerical flux

    parameters
    ----------
    u: jax.Array
        solution
    coeffs: list
        coefficients

    returns
    -------
    num_flux: jax.Array
        numerical flux
    """
    ORDER = 5  # 5th order WENO
    CENTER = 2  # index of central solution

    num_flux = np.empty(u.shape)
    for i in range(len(u)):

        def create_stencil(u: jax.Array, *, i: int, side: str) -> tuple[jax.Array, jax.Array, jax.Array]:
            """
            choose spatial coordinates for approximating solution at cell interface \\
            left = {i-2, i-1, i, i+1, i+2} \\
            right = {i+3, i+2, i+1, i, i-1}
            """
            if side == "right":
                i += 1
            stencil = np.empty((ORDER,))
            for n in range(ORDER - CENTER):
                before, after = neighbor(u, i=i, n=n)
                stencil[CENTER - n] = before
                stencil[CENTER + n] = after
            if side == "right":
                stencil = jnp.flip(stencil)
            s0 = jnp.array([stencil[0], stencil[1], stencil[2]])
            s1 = jnp.array([stencil[1], stencil[2], stencil[3]])
            s2 = jnp.array([stencil[2], stencil[3], stencil[4]])
            return s0, s1, s2

        left_s0, left_s1, left_s2 = create_stencil(u, i=i, side="left")
        right_s0, right_s1, right_s2 = create_stencil(u, i=i, side="right")

        def smooth_stencil(s0: jax.Array, s1: jax.Array, s2: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
            """
            compute smoothness of stencil
            """
            beta0 = 13 / 12 * (s0[0] - 2 * s0[1] + s0[2]) ** 2 + 1 / 4 * (
                s0[0] - 4 * s0[1] + 3 * s0[2]
            ) ** 2
            beta1 = 13 / 12 * (s1[0] - 2 * s1[1] + s1[2]) ** 2 + 1 / 4 * (
                s1[0] - s1[2]
            ) ** 2
            beta2 = 13 / 12 * (s2[0] - 2 * s2[1] + s2[2]) ** 2 + 1 / 4 * (
                3 * s2[0] - 4 * s2[1] + s2[2]
            ) ** 2
            return beta0, beta1, beta2

        left_beta0, left_beta1, left_beta2 = smooth_stencil(left_s0, left_s1, left_s2)
        right_beta0, right_beta1, right_beta2 = smooth_stencil(right_s0, right_s1, right_s2)

        def weight_stencil(
            beta0: jax.Array, beta1: jax.Array, beta2: jax.Array, *, eps: float = 1e-6
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            """
            if smooth stencil then large weight \\
            if rough stencil then small weight
            """
            alpha0 = 1 / 10 / (eps + beta0) ** 2
            alpha1 = 6 / 10 / (eps + beta1) ** 2
            alpha2 = 3 / 10 / (eps + beta2) ** 2
            return alpha0, alpha1, alpha2

        left_alpha0, left_alpha1, left_alpha2 = weight_stencil(left_beta0, left_beta1, left_beta2)
        right_alpha0, right_alpha1, right_alpha2 = weight_stencil(right_beta0, right_beta1, right_beta2)

        def norm_weight(
            alpha0: jax.Array, alpha1: jax.Array, alpha2: jax.Array
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            """
            normalize stencil weight
            """
            omega0 = alpha0 / (alpha0 + alpha1 + alpha2)
            omega1 = alpha1 / (alpha0 + alpha1 + alpha2)
            omega2 = alpha2 / (alpha0 + alpha1 + alpha2)
            return omega0, omega1, omega2

        left_omega0, left_omega1, left_omega2 = norm_weight(left_alpha0, left_alpha1, left_alpha2)
        right_omega0, right_omega1, right_omega2 = norm_weight(
            right_alpha0, right_alpha1, right_alpha2
        )

        def approx_sol(
            s0: jax.Array, s1: jax.Array, s2: jax.Array
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            """
            approximate solution at cell interface using Lagrange interpolation \\
            q \\approx u(x_{i+0.5})
            """
            q0 = 1 / 3 * s0[0] - 7 / 6 * s0[1] + 11 / 6 * s0[2]
            q1 = -1 / 6 * s1[0] + 5 / 6 * s1[1] + 1 / 3 * s1[2]
            q2 = 1 / 3 * s2[0] + 5 / 6 * s2[1] - 1 / 6 * s2[2]
            return q0, q1, q2

        left_q0, left_q1, left_q2 = approx_sol(left_s0, left_s1, left_s2)
        right_q0, right_q1, right_q2 = approx_sol(right_s0, right_s1, right_s2)

        # reconstruct solution
        """
        WENO5 (Weighted Essentially Non-Oscillatory fifth-order) reconstructs high-order estimates of the solution at cell interfaces by adaptively combining several lower-order stencils, assigning larger weights to smooth stencils and smaller weights to stencils containing discontinuities. This produces fifth-order accuracy in smooth parts of the solution while preventing the spurious oscillations that standard high-order methods generate near shocks and other sharp gradients.
        """
        left_sol = left_q0 * left_omega0 + left_q1 * left_omega1 + left_q2 * left_omega2
        right_sol = right_q0 * right_omega0 + right_q1 * right_omega1 + right_q2 * right_omega2

        # compute local lax-friedrichs / rusanov flux
        left_flux = flux(left_sol, coeffs)
        right_flux = flux(right_sol, coeffs)
        left_speed = speed(left_sol, coeffs)
        right_speed = speed(right_sol, coeffs)
        left_max_speed = max_speed(left_speed)
        right_max_speed = max_speed(right_speed)
        bi_max_speed = jnp.maximum(left_max_speed, right_max_speed)
        num_flux[i] = 0.5 * (
            left_flux + right_flux
            ) - 0.5 * bi_max_speed * (
            right_sol - left_sol
            )
    return jnp.array(num_flux)


def rhs(num_flux: jax.Array, dx: float) -> jax.Array:
    """
    compute right hand side of equation \\
    du/dt = -df(u)/dx

    parameters
    ----------
    num_flux: jax.Array
        numerical flux
    dx: float
        grid spacing
    
    returns
    -------
    jax.Array
        rhs of 1d conservation law
    """
    dfdx = np.empty(num_flux.shape)
    for i in range(len(num_flux)):
        dfdx[i] = -(num_flux[i] - num_flux[i - 1]) / dx
    return jnp.array(dfdx)


def rk4_step(u: jax.Array, dx: float, dt: float) -> jax.Array:
    """
    approximate solution at next time step using 4th order runge-kutta

    parameters
    ----------
    u: jax.Array
        solution
    dx: float
        grid spacing
    dt: float
        time step

    returns
    -------
    u_dt: jax.Array
        solution differential between time step
    """
    k1 = rhs(u, dx)
    k2 = rhs(u + k1 * dt / 2, dx)
    k3 = rhs(u + k2 * dt / 2, dx)
    k4 = rhs(u + k3 * dt, dx)
    u_dt = dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return u_dt


def rk4_evolve(u: jax.Array, dx: float, dt: float) -> jax.Array:
    """
    evolve approximate solution through time using repeated RK4 steps

    parameters
    ----------
    u: jax.Array
        solution
    dx: float
        grid spacing
    dt: float
        time step

    returns
    -------
    trajs: jax.Array
        solution trajectories
    """
    min_time = 0 # TODO: parameterize min_time
    max_time = 0.5 # TODO: parameterize max_time
    t = min_time 
    trajs = list()
    while t < max_time:
        trajs.append(u)
        u_dt = rk4_step(u, dx, dt)
        u += u_dt
        t += dt
    trajs = jnp.array(trajs)
    return trajs


def demonstrate(trajs: jax.Array, *, n: int) -> jax.Array:
    """
    create (input, target) examples

    parameters
    ----------
    trajs: jax.Array
        solution trajectories
    n: int
        number of time steps
    
    returns
    -------
    examples: jax.Array
        input-target examples
    """
    num_exs = len(trajs) - n
    examples = list()
    for i in range(num_exs):
        u_t = trajs[i]
        u_tau = trajs[i + n]
        ex = (u_t, u_tau)
        examples.append(ex)
    examples = jnp.array(examples)
    return examples


def main():
    pass


if __name__ == "__main__":
    main()
