#vfrom QTorch.Basic import *
from collections import Counter
import qutip as qt
import numpy as np
from scipy.linalg import logm, sqrtm, schur, block_diag, expm
from itertools import permutations
from thewalrus.decompositions import williamson, blochmessiah
from thewalrus.symplectic import xpxp_to_xxpp, xxpp_to_xpxp, sympmat, is_symplectic


def covariance_matrix(basis, rho):
    first_moments = np.array([qt.expect(oi, rho) for oi in basis], )  #dtype=np.float64
    cm = np.array([[0.5 * qt.expect(oi * oj + oj * oi, rho) - qt.expect(oi, rho) * qt.expect(oj, rho)
                    for oi in basis] for oj in basis], )  #dtype=np.float64

    return first_moments, cm


def measure_observable(state, operator, num_samples):
    if state.isket:
        # Pure state: Ensure state is normalized
        state = state.unit()

        # Get the eigenstates and eigenvalues of the operator
        eigenvalues, eigenstates = operator.eigenstates()

        # Calculate probabilities of measurement outcomes for pure state
        probabilities = [abs((eigenstate.dag() * state)) ** 2 for eigenstate in eigenstates]

    elif state.isoper:
        # Mixed state: Ensure state is a valid density matrix
        rho = state / state.tr()

        # Get the eigenstates and eigenvalues of the operator
        eigenvalues, eigenstates = operator.eigenstates()

        # Calculate probabilities of measurement outcomes for mixed state
        probabilities = [np.real((eigenstate.dag() * rho * eigenstate)) for eigenstate in eigenstates]

    else:
        raise TypeError("The input state must be either a pure state (ket) or a mixed state (density matrix).")

    # Get the eigenstates and eigenvalues of the operator

    # Calculate probabilities of measurement outcomes

    # Ensure probabilities sum to 1 (fix small numerical errors)
    probabilities = np.array(probabilities)
    probabilities = np.around(probabilities, 10)
    probabilities /= probabilities.sum()

    # Generate measurement samples based on probabilities
    samples = np.random.choice(np.arange(len(probabilities)), size=num_samples, p=probabilities)

    # Count occurrences of each eigenvalue
    sample_counts = Counter(samples)

    # Prepare result as list of tuples (eigenvalue, frequency, eigenstate)
    result = []
    frequency_list = []

    for i, eigenstate in enumerate(eigenstates):
        frequency = sample_counts[i] / num_samples
        frequency_list.append(frequency)

    return frequency_list, probabilities, eigenvalues, eigenstates,


def measure_observable_mean(psi, operator, num_samples):
    frequency_list, probabilities, eigenvalues, eigenstates, = measure_observable(psi, operator, num_samples)
    operator_expect = sum(eigenvalue * freq for eigenvalue, freq in list(zip(eigenvalues, frequency_list)))
    return operator_expect


def measure_covariance_matrix(psi, basis, num_samples):
    """
    Calculate the covariance matrix for a two-mode bosonic quantum state.

    Parameters:
        psi: quantum state (two-mode)
        num_samples: number of samples for the measurement
        Nt: number of levels for the Fock space truncation (default 10)

    Returns:
        covariance_matrix: 4x4 covariance matrix for the two-mode system
    """
    # Define two-mode quadrature operators using position and momentum

    # First, define the quadrature vector R = [x1, p1, x2, p2]
    quadratures = basis

    # Initialize arrays for first and second moments
    first_moments = np.zeros(len(basis))

    # Measure the first moments ⟨R_i⟩
    for i in range(len(basis)):
        first_moments[i] = measure_observable_mean(psi, quadratures[i], num_samples)

    # Compute the covariance matrix
    covariance_matrix = np.zeros((len(basis), len(basis)), dtype=np.complex128)
    for i in range(len(basis)):
        for j in range(i, len(basis)):
            print("measure cov:", i, j)
            operator = quadratures[i] * quadratures[j] + quadratures[j] * quadratures[i]
            Mij = measure_observable_mean(psi, operator, num_samples)
            covariance_matrix[i, j] = 0.5 * Mij - first_moments[i] * first_moments[j]
            covariance_matrix[j, i] = 0.5 * Mij - first_moments[i] * first_moments[j]

    return first_moments, np.real(covariance_matrix)


def unitary_beam_splitter(theta, phi, Nt):
    a = qt.tensor(qt.destroy(Nt), qt.qeye(Nt))
    b = qt.tensor(qt.qeye(Nt), qt.destroy(Nt))
    H = theta * (np.exp(-1j * phi) * a.dag() * b - np.exp(1j * phi) * a * b.dag() )
    U_bs = H.expm()
    return U_bs


def position_momentum_operators_multimode(Nt, num_modes, index):
    """
    Define the position and momentum operators for a given number of modes and cutoff dimension.

    Parameters:
    Nt (int): Cutoff dimension.
    num_modes (int): Number of modes.
    index (int): The mode index for which to define the position and momentum operators.

    Returns:
    Qobj: Position operator for the given mode.
    Qobj: Momentum operator for the given mode.
    """
    position_operators = []
    momentum_operators = []

    for i in range(num_modes):
        I = qt.qeye(Nt)
        if i == index:
            x = qt.create(Nt) + qt.destroy(Nt)
            p = -1j * (qt.create(Nt) - qt.destroy(Nt))
            position_operators.append(x)
            momentum_operators.append(p)
        else:
            position_operators.append(I)
            momentum_operators.append(I)

    # Ensure all elements are Qobj instances
    position_operators = [op if isinstance(op, qt.Qobj) else qt.Qobj(op) for op in position_operators]
    momentum_operators = [op if isinstance(op, qt.Qobj) else qt.Qobj(op) for op in momentum_operators]

    # Take the tensor product
    position_operator = qt.tensor(*position_operators)
    momentum_operator = qt.tensor(*momentum_operators)

    return position_operator, momentum_operator


def RrhoR(R, rho, epsilon=10):
    # Calculate the R(\rho)R operator
    if epsilon < 1:
        I = qt.qeye(rho.dims[0][0])
        I = qt.tensor(*[I] * len(rho.dims[0]))

        R = (I + epsilon * R) / (1 + epsilon)

    rho = R * rho * R

    rho = rho / rho.tr()
    return rho


def RrhoR_measurement_generator_onemode(psi, Nt, d, K):
    f_list = []
    p_list = []
    state_list = []
    for i in range(d):
        theta = i * np.pi * 2 / d
        operator = (np.exp(1j * theta) * qt.create(Nt) + np.exp(-1j * theta) * qt.destroy(Nt)) / np.sqrt(2)

        measurement_result = measure_observable(psi, operator, K)
        index = np.where(K * np.array(measurement_result[0]) > 0)[0]
        for j in index:
            f = measurement_result[0][j]
            p = measurement_result[1][j]
            state = measurement_result[3][j].full()
            f_list.append(f)
            p_list.append(p)
            state_list.append(state)
    return f_list, p_list, state_list


def RrhoR_state_tomograph_onemode(psi, Nt, epsilon, d, K, num_interation, disp=False):
    num_samples = d * K
    # measure real state psi
    f_list, p_list, state_list = RrhoR_measurement_generator_onemode(psi, Nt, d, K)
    rho0 = qt.qeye(Nt) / Nt
    Infidelity = 1 - qt.fidelity(rho0, psi)
    tracedistance = qt.metrics.tracedist(rho0, psi)
    rho_list = [rho0.full()]
    if_list = [Infidelity]
    td_list = [tracedistance]

    for i in range(num_interation):
        rho_i = qt.Qobj(rho_list[-1], dims=[[Nt], [Nt]])
        R_rho_i = qt.Qobj(np.zeros((rho_i.shape[0], rho_i.shape[1])), dims=rho_i.dims)
        for j in range(len(state_list)):
            state = qt.Qobj(state_list[j], dims=[[Nt], [1]])
            pro_j = state * state.dag()
            R_rho_i = R_rho_i + pro_j * f_list[j] * K / ((rho_i * pro_j).tr() * num_samples)

        rho_updated = RrhoR(R_rho_i, rho_i, epsilon=epsilon)
        Infidelity = 1 - qt.fidelity(rho_updated, psi)
        tracedistance = qt.metrics.tracedist(rho_updated, psi)

        td_list.append(tracedistance)
        if_list.append(Infidelity)
        rho_list.append(rho_updated.full())

        if disp:
            print(len(if_list), if_list[-1], np.log10(1 - if_list[-1]), td_list[-1], np.log10(1 - td_list[-1]))
    return td_list[-1], if_list[-1], rho_list[-1]


def unitary_phase_rotation_qt(theta, Nt):
    return (-1j * theta * qt.num(Nt)).expm()


def cov_rotation(theta):
    cov = [[np.cos(theta), np.sin(theta)],
           [-np.sin(theta), np.cos(theta)]]
    return np.array(cov)


def beam_splitter_cov(t1, t2, t3, t4, s):
    #t00,t01,t10,t11 = t1+t3,t2+t3,t1+t4,t2+t4
    t00, t01, t10, t11 = t1, t2, t3, t4
    R00 = cov_rotation(t00)
    R01 = cov_rotation(t01)
    R10 = cov_rotation(t10)
    R11 = cov_rotation(t11)

    # Combine the top and bottom rows into the final 4x4 matrix
    cov = np.concatenate((np.concatenate((np.cos(s) * R00, np.sin(s) * R01), axis=1),
                          np.concatenate((-np.sin(s) * R10, np.cos(s) * R11), axis=1)), axis=0)
    """
    cov = [
        [np.cos(s) * np.cos(t1), np.cos(s) * np.sin(t1), np.cos(t2) * np.sin(s), np.sin(s) * np.sin(t2)],
        [-np.cos(s) * np.sin(t1), np.cos(s) * np.cos(t1), -np.sin(s) * np.sin(t2), np.cos(t2) * np.sin(s)],
        [-np.cos(t1) * np.sin(s), -np.sin(s) * np.sin(t1), np.cos(s) * np.cos(t2), np.cos(s) * np.sin(t2)],
        [np.sin(s) * np.sin(t1), -np.cos(t1) * np.sin(s), -np.cos(s) * np.sin(t2), np.cos(s) * np.cos(t2)]]
    """

    return cov


def symplectic_matrix(num_modes):
    omega_2x2 = np.array([[0, 1], [-1, 0]])

    # Use the Kronecker product to generate the full symplectic matrix for N modes
    symplectic_matrix = np.kron(np.eye(num_modes), omega_2x2)

    return symplectic_matrix


def bms_cov_to_gate2(cov, Nt):
    sub1 = cov[0:2, 0:2]
    sub2 = cov[0:2, 2:4]
    sub3 = cov[2:4, 0:2]
    sub4 = cov[2:4, 2:4]
    cos_s = np.sqrt(sub1[0, 0] ** 2 + sub1[0, 1] ** 2)
    sin_s = np.sqrt(sub2[0, 0] ** 2 + sub2[0, 1] ** 2)
    s = np.arctan2(sin_s, cos_s)
    if cos_s >= 1e-10:
        sub1 = sub1 / cos_s
        t1prime = np.arctan2(sub1[0, 1], sub1[0, 0])
        sub4 = sub4 / cos_s
        t4prime = np.arctan2(sub4[0, 1], sub4[0, 0])
    else:
        t1prime = 0
        t4prime = 0
    if sin_s >= 1e-10:
        sub2 = sub2 / sin_s
        t2prime = np.arctan2(sub2[0, 1], sub2[0, 0])
        sub3 = -sub3 / sin_s
        t3prime = np.arctan2(sub3[0, 1], sub3[0, 0])
    else:
        t2prime = 0
        t3prime = 0
    print("solution")
    print([t1prime, t2prime, t3prime, t4prime, s])
    cov_trial = beam_splitter_cov(t1prime, t2prime, t3prime, t4prime, s)

    error = np.sum(np.abs(cov_trial - cov))
    if error < 1e-5:
        print("solution verified:", )

    else:
        raise Exception("Solution Wrong")

    t1 = 0
    t2 = -t1prime + t2prime
    t3 = t1prime
    t4 = t3prime

    U = qt.tensor(unitary_phase_rotation_qt(t1, Nt), unitary_phase_rotation_qt(t2, Nt))
    U = unitary_beam_splitter(s, 0, Nt) * U
    U = qt.tensor(unitary_phase_rotation_qt(t3, Nt), unitary_phase_rotation_qt(t4, Nt)) * U

    return U


def omega_xpxp(n):
    # build Ω_xpxp by permuting Walrus’ Ω_xxpp
    return xxpp_to_xpxp(sympmat(n))


def williamson_xpxp(V_xpxp, rtol=1e-7, atol=1e-10):
    """Return (Db_xpxp, S_xpxp) with V_xpxp = S_xpxp @ Db_xpxp @ S_xpxp.T."""
    V_xxpp = xpxp_to_xxpp(V_xpxp)
    Db_xxpp, S_xxpp = williamson(V_xxpp, rtol=rtol, atol=atol)  # Walrus (xxpp)
    # convert results back to xpxp
    Db_xpxp = xxpp_to_xpxp(Db_xxpp)
    S_xpxp = xxpp_to_xpxp(S_xxpp)
    # checks in xpxp
    Om = omega_xpxp(V_xpxp.shape[0] // 2)
    recon_err = np.linalg.norm(S_xpxp @ Db_xpxp @ S_xpxp.T - V_xpxp, ord='fro')
    sympl_err = np.linalg.norm(S_xpxp.T @ Om @ S_xpxp - Om, ord='fro')
    #print({"recon_err": recon_err, "sympl_err": sympl_err})
    #print("Db_xpxp:", Db_xpxp)

    return Db_xpxp, S_xpxp, {"recon_err": recon_err, "sympl_err": sympl_err}


def blochmessiah_xpxp(S_xpxp):
    """
    Return (O_xpxp, D_xpxp, Q_xpxp) with S_xpxp = O @ D @ Q in xpxp,
    where O,Q are orthogonal–symplectic and D is single-mode squeezers.
    """
    S_xxpp = xpxp_to_xxpp(S_xpxp)
    O_xxpp, D_xxpp, Q_xxpp = blochmessiah(S_xxpp)  # Walrus (xxpp)
    # convert factors back to xpxp
    O_xpxp = xxpp_to_xpxp(O_xxpp)
    D_xpxp = xxpp_to_xpxp(D_xxpp)
    Q_xpxp = xxpp_to_xpxp(Q_xxpp)
    # checks in xpxp
    Om = omega_xpxp(S_xpxp.shape[0] // 2)
    recon_err = np.linalg.norm(O_xpxp @ D_xpxp @ Q_xpxp - S_xpxp, ord='fro')
    ok_O_sym = np.linalg.norm(O_xpxp.T @ Om @ O_xpxp - Om, ord='fro')
    ok_O_ort = np.linalg.norm(O_xpxp.T @ O_xpxp - np.eye(S_xpxp.shape[0]), ord='fro')
    ok_Q_sym = np.linalg.norm(Q_xpxp.T @ Om @ Q_xpxp - Om, ord='fro')
    ok_Q_ort = np.linalg.norm(Q_xpxp.T @ Q_xpxp - np.eye(S_xpxp.shape[0]), ord='fro')
    #print("O_xpxp:", O_xpxp)
    #print("D_xpxp:",D_xpxp)
    #print("Q_xpxp:", Q_xpxp)
    return O_xpxp, D_xpxp, Q_xpxp, {"recon_err": recon_err,
                                    "O_sympl": ok_O_sym,
                                    "ok_O_ort": ok_O_ort,
                                    "Q_sympl": ok_Q_sym,
                                    "ok_Q_ort": ok_Q_ort}


def omega_xpxp(n: int) -> np.ndarray:
    """Ω for xpxp ordering: (q1,p1,q2,p2,...)"""
    J = np.array([[0.0, 1.0], [-1.0, 0.0]])
    return np.kron(np.eye(n), J)


def omega_xxpp(n: int) -> np.ndarray:
    """Ω for xxpp ordering: (q1,q2,...,p1,p2,...)"""
    Z = np.zeros((n, n));
    I = np.eye(n)
    return np.block([[Z, I], [-I, Z]])


def P_xpxp_to_xxpp(n: int) -> np.ndarray:
    """
    Permutation that maps v_xpxp -> v_xxpp:
    [q1,p1,q2,p2,...] -> [q1,q2,...,p1,p2,...]
    So v_xxpp = P @ v_xpxp, and K_xxpp = P @ K_xpxp @ P.T
    """
    P = np.zeros((2 * n, 2 * n))
    for i in range(n):
        P[i, 2 * i] = 1.0  # q_i
        P[n + i, 2 * i + 1] = 1.0  # p_i
    return P


def detect_and_to_xpxp(K: np.ndarray):
    """Detect whether K is xpxp or xxpp (via symplectic test) and return (K_xpxp, ordering_label)."""
    n = K.shape[0] // 2
    Om_xpxp = omega_xpxp(n)
    Om_xxpp = omega_xxpp(n)
    e_xpxp = np.linalg.norm(K.T @ Om_xpxp @ K - Om_xpxp, ord='fro')
    e_xxpp = np.linalg.norm(K.T @ Om_xxpp @ K - Om_xxpp, ord='fro')
    if e_xxpp < e_xpxp:
        P = P_xpxp_to_xxpp(n)
        return P.T @ K @ P, "xxpp→xpxp"
    return K, "xpxp"


# ==========================================
# Passive xpxp <-> 2x2 unitary (2-mode only)
# ==========================================
def U_from_K_xpxp_blocks(K: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Read U from xpxp by 2×2 micro-blocks [[a,-b],[b,a]]."""
    assert K.shape == (4, 4), "Expect a 4x4 K for 2 modes."
    U = np.zeros((2, 2), dtype=complex)
    for i in range(2):
        for j in range(2):
            B = K[2 * i:2 * i + 2, 2 * j:2 * j + 2]
            a, b = B[0, 0], B[1, 0]
            if not np.allclose(B, np.array([[a, -b], [b, a]]), atol=tol):
                raise ValueError("Block not [[a,-b],[b,a]] — not purely passive in xpxp.")
            U[i, j] = a + 1j * b
    return U


def U_from_K_xpxp_macro(K: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Read U from xpxp using macro partition: q' = A q + B p, p' = C q + D p."""
    A = K[0::2, 0::2]  # (q rows, q cols)
    B = K[0::2, 1::2]  # (q rows, p cols)
    C = K[1::2, 0::2]  # (p rows, q cols)
    D = K[1::2, 1::2]  # (p rows, p cols)
    if not (np.allclose(A, D, atol=tol) and np.allclose(C, -B, atol=tol)):
        raise ValueError("K is not passive in xpxp (A!=D or C!=-B).")
    return A + 1j * C  # Re U = A, Im U = C


def K_from_U_xpxp(U: np.ndarray) -> np.ndarray:
    """Embed U into xpxp using [[a,-b],[b,a]] per entry."""
    K = np.zeros((4, 4))
    for i in range(2):
        for j in range(2):
            a, b = U[i, j].real, U[i, j].imag
            K[2 * i:2 * i + 2, 2 * j:2 * j + 2] = np.array([[a, -b], [b, a]])
    return K


def K_from_U_xpxp_alt(U: np.ndarray) -> np.ndarray:
    """Alternative intra-block sign: [[a, +b], [-b, a]] (equiv. to p -> -p)."""
    K = np.zeros((4, 4))
    for i in range(2):
        for j in range(2):
            a, b = U[i, j].real, U[i, j].imag
            K[2 * i:2 * i + 2, 2 * j:2 * j + 2] = np.array([[a, +b], [-b, a]])
    return K


# =========================
# Parameterization helpers
# =========================
def wrap(x: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return (x + np.pi) % (2 * np.pi) - np.pi


def project_to_unitary(U: np.ndarray) -> np.ndarray:
    """Nearest unitary (polar decomposition via SVD)."""
    X, _, Yh = np.linalg.svd(U)
    return X @ Yh


def unitary_from_angles(theta, t1, t2, t3, t4, bs_sign: int = -1) -> np.ndarray:
    """
    Build U = diag(e^{i t3}, e^{i t4}) @ BS(theta) @ diag(e^{i t1}, e^{i t2})
    bs_sign = -1  => BS = [[c, s], [-s, c]]   (common in QO)
            = +1  => BS = [[c, s], [ s, c]]
    """
    c, s = np.cos(theta), np.sin(theta)
    BS = np.array([[c, s], [bs_sign * s, c]], dtype=complex)
    L = np.diag(np.exp(1j * np.array([t3, t4])))
    R = np.diag(np.exp(1j * np.array([t1, t2])))
    return L @ BS @ R


def angles_from_unitary(U: np.ndarray, gauge: float = 0.0, bs_sign: int = -1):
    """
    Extract (theta, t1, t2, t3, t4) up to a chosen gauge t2 for:
      U = diag(e^{i t3}, e^{i t4}) [[c,s],[-s,c]] diag(e^{i t1}, e^{i t2})
    """
    # U = project_to_unitary(U)
    chi = 0.5 * np.angle(np.linalg.det(U))  # remove global phase
    Up = U * np.exp(-1j * chi)

    c, s = np.abs(Up[0, 0]), np.abs(Up[0, 1])
    theta = np.arctan2(s, c)

    alpha = np.angle(Up[0, 0])  # t3 + t1
    beta = np.angle(Up[0, 1])  # t3 + t2
    gamma = np.angle(bs_sign * Up[1, 0])  # t4 + t1
    delta = np.angle(Up[1, 1])  # t4 + t2

    t2 = float(gauge)
    t1 = wrap(t2 + (alpha - beta))
    t3 = wrap(beta - t2)
    t4 = wrap(delta - t2)
    return theta, t1, t2, t3, t4


def angles_from_unitary2(U: np.ndarray, gauge: float = 0.0, bs_sign: int = -1):
    """
    Extract (theta, t1, t2, t3, t4) up to a chosen gauge t2 for:
      U = diag(e^{i t3}, e^{i t4}) [[c,s],[-s,c]] diag(e^{i t1}, e^{i t2})
    """
    # U = project_to_unitary(U)
    #chi = 0.5 * np.angle(np.linalg.det(U))  # remove global phase
    #Up = U * np.exp(-1j * chi)
    Up = U

    c, s = np.abs(Up[0, 0]), np.abs(Up[0, 1])
    theta = np.arctan2(s, c)

    alpha = np.angle(Up[0, 0])  # t3 + t1
    beta = np.angle(Up[0, 1])  # t3 + t2
    gamma = np.angle(bs_sign * Up[1, 0])  # t4 + t1
    delta = np.angle(Up[1, 1])  # t4 + t2

    t3 = float(gauge)
    t1 = alpha
    t2 = beta - t3
    t4 = delta - t2
    return theta, wrap(t1), wrap(t2), wrap(t3), wrap(t4)


# =========================
# Main decomposition
# =========================
def decompose_passive_2mode(K_in: np.ndarray, gauge: float = 0.0, bs_sign: int = -1, verbose: bool = True):
    assert K_in.shape == (4, 4), "K must be 4x4 for 2 modes."

    # 0) ensure xpxp ordering
    K_xpxp, ordering = detect_and_to_xpxp(K_in)

    # 1) sanity on input (symplectic + orthogonal)
    Om = omega_xpxp(2)
    e_sympl = np.linalg.norm(K_xpxp.T @ Om @ K_xpxp - Om, ord='fro')
    e_orth = np.linalg.norm(K_xpxp.T @ K_xpxp - np.eye(4), ord='fro')

    # 2) extract U two ways (consistency)
    U_b = U_from_K_xpxp_blocks(K_xpxp)
    U_m = U_from_K_xpxp_macro(K_xpxp)
    e_ubm = np.linalg.norm(U_b - U_m, ord='fro')
    U = 0.5 * (U_b + U_m)  # de-noise

    # 3) round-trip K->U->K (no decomposition)
    K_rt = K_from_U_xpxp(U)
    err_rt = np.linalg.norm(K_xpxp - K_rt, ord='fro')

    # 4) decompose U, rebuild U_rec and **align global phase (FIXED SIGN: +i phi)**
    theta, t1, t2, t3, t4 = angles_from_unitary2(U, gauge=gauge, bs_sign=bs_sign)
    U_rec = unitary_from_angles(theta, t1, t2, t3, t4, bs_sign=bs_sign)
    phi = np.angle(np.trace(U_rec.conj().T @ U))  # arg tr(U_rec^† U)
    U_rec_aligned = U_rec * np.exp(+1j * phi)  # align to U

    # (compare in U-space with det-fixed versions)
    chiU = 0.5 * np.angle(np.linalg.det(U))
    chiUr = 0.5 * np.angle(np.linalg.det(U_rec_aligned))
    err_U = np.linalg.norm(U * np.exp(-1j * chiU) - U_rec_aligned * np.exp(-1j * chiUr), ord='fro')

    # 5) embed back to K (std & alt micro-conventions)
    K_rec = K_from_U_xpxp(U_rec_aligned)
    K_rec_alt = K_from_U_xpxp_alt(U_rec_aligned)

    # compare in xpxp and in input ordering
    if ordering == "xxpp→xpxp":
        P = P_xpxp_to_xxpp(2)
        K_rec_in = P @ K_rec @ P.T
        K_alt_in = P @ K_rec_alt @ P.T
    else:
        K_rec_in = K_rec
        K_alt_in = K_rec_alt

    err_K = np.linalg.norm(K_in - K_rec_in, ord='fro')
    err_Kalt = np.linalg.norm(K_in - K_alt_in, ord='fro')
    err_Kx = np.linalg.norm(K_xpxp - K_rec, ord='fro')
    err_Kxa = np.linalg.norm(K_xpxp - K_rec_alt, ord='fro')

    use_alt = err_Kalt < err_K
    K_best = K_alt_in if use_alt else K_rec_in
    err_best = min(err_K, err_Kalt)

    if verbose:
        print(f"[ordering]  assumed: {ordering}")
        print(f"[checks  ]  ||K^T Ω_xpxp K - Ω||_F : {e_sympl:.3e}")
        print(f"[checks  ]  ||K^T K - I||_F       : {e_orth:.3e}")
        print(f"[U-cons ]  ||U_blocks - U_macro|| : {e_ubm:.3e}")
        print(f"[RT K→U→K] ||K - K_rt||_F         : {err_rt:.3e}   (no decomposition)")
        print(f"[U-space ]  ||U - U_rec||_F       : {err_U:.3e}")
        print(f"[K-space ]  ||K_in - K_rec||_F    : {err_K:.3e} (std)")
        print(f"[K-space ]  ||K_in - K_rec_alt||_F: {err_Kalt:.3e} (alt p-sign)")
        print(f"[K-space ]  picked {'ALT' if use_alt else 'STD'} mapping; final err = {err_best:.3e}")

    return {
        "theta": float(wrap(theta)),
        "theta1": float(wrap(t1)),
        "theta2": float(wrap(t2)),
        "theta3": float(wrap(t3)),
        "theta4": float(wrap(t4)),
        "U": U,
        "U_rec_aligned": U_rec_aligned,
        "K_rec": K_best,
        "errors": {
            "symplectic": float(e_sympl),
            "orthogonal": float(e_orth),
            "U_blocks_vs_macro": float(e_ubm),
            "roundtrip_KU": float(err_rt),
            "U_match": float(err_U),
            "K_match": float(err_best)
        }
    }


def angles_to_gate(angles, Nt):
    theta = angles[0]
    t1 = -angles[1]
    t2 = -angles[2]
    t3 = -angles[3]
    t4 = 0

    U = qt.tensor(unitary_phase_rotation_qt(t1, Nt), unitary_phase_rotation_qt(t2, Nt))
    U = unitary_beam_splitter(theta, 0, Nt) * U
    U = qt.tensor(unitary_phase_rotation_qt(t3, Nt), unitary_phase_rotation_qt(t4, Nt)) * U
    return U


def RM(theta):
    M = [[np.cos(theta), np.sin(theta)],
         [-np.sin(theta), np.cos(theta)]]
    return np.array(M)



def angles_to_K(angles):
    theta = angles[0]
    t1 = angles[1]
    t2 = angles[2]
    t3 = angles[3]
    t4 = 0
    M0 = np.zeros((2, 2))
    I = np.eye(2)
    K1 = np.block([[RM(t1), M0], [M0, RM(t2)]])
    K2 = np.kron(RM(theta), I)
    K3 = np.block([[RM(t3), M0], [M0, RM(t4)]])
    return K3 @ K2 @ K1


from scipy.optimize import root


def K_to_angeles(K):
    def g(x):
        K_x = angles_to_K(x)
        dis = np.linalg.norm((K_x - K), ord='fro')
        return np.array([dis, 0, 0, 0])

    for i in range(1000):
        x0 = np.random.rand(4)
        result = root(g, x0)
        print(i, result["success"])
        if result["success"]:
            x = result["x"]
            break
    return x


def passive_K_to_gate(K_in, Nt):
    """
    result = decompose_passive_2mode(K_in, verbose=False)
    theta = result["theta"]
    t1 = result["theta1"]
    t2 = result["theta2"]
    t3 = result["theta3"]
    t4 = result["theta4"]
    """
    angeles = K_to_angeles(K_in)
    theta = angeles[0]
    t1 = angeles[1]
    t2 = angeles[2]
    t3 = angeles[3]

    print("angles:", theta, t1, t2, t3, )
    gate = angles_to_gate([theta, t1, t2, t3, ], Nt)
    return gate


def US_gate_2mode(V, Nt):
    Db, S, error1 = williamson_xpxp(V)
    print("Db:", Db)
    print("williamson error:", error1)
    K1, Delta, K2, error2 = blochmessiah_xpxp(S)
    print("K1:", K1)
    print("Delta:", Delta)
    print("K2:", K2)
    print(error2)

    r1 = np.abs(np.log(Delta[0, 0]))
    r2 = np.abs(np.log(Delta[2, 2]))
    U_squeeze1 = qt.tensor(qt.squeeze(Nt, r1), qt.squeeze(Nt, r2))
    U_squeeze2 = qt.tensor(qt.squeeze(Nt, -r1), qt.squeeze(Nt, r2))
    U_squeeze3 = qt.tensor(qt.squeeze(Nt, r1), qt.squeeze(Nt, -r2))
    U_squeeze4 = qt.tensor(qt.squeeze(Nt, -r1), qt.squeeze(Nt, -r2))
    Uo1 = passive_K_to_gate(K1, Nt)
    Uo2 = passive_K_to_gate(K2, Nt)
    return Uo2 * U_squeeze1 * Uo1, Uo2 * U_squeeze2 * Uo1, Uo2 * U_squeeze3 * Uo1, Uo2 * U_squeeze4 * Uo1


def pairwise_difference(rho1, rho2):
    rho_diff = (rho1 - rho2).full()
    return np.max(np.abs(rho_diff))
