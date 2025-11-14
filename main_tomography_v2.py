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
    H = theta * (np.exp(-1j * phi) * a.dag() * b - np.exp(1j * phi) * a * b.dag())
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


def symplectic_matrix(num_modes):
    omega_2x2 = np.array([[0, 1], [-1, 0]])

    # Use the Kronecker product to generate the full symplectic matrix for N modes
    symplectic_matrix = np.kron(np.eye(num_modes), omega_2x2)

    return symplectic_matrix


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

    # fixing gauge
    O_xpxp, Q_xpxp = O_xpxp @ O_xpxp.T, O_xpxp @ Q_xpxp

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


def angles_to_gate(angles, Nt):
    theta = angles[0]
    t1 = angles[1]
    t2 = angles[2]
    t3 = angles[3]
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
    angeles = K_to_angeles(K_in)
    theta = angeles[0]
    t1 = angeles[1]
    t2 = angeles[2]
    t3 = angeles[3]

    #print("angles:", theta, t1, t2, t3, )
    gate = angles_to_gate([theta, t1, t2, t3, ], Nt)
    return gate


def US_gate_2mode(V, Nt):
    Db, S, error1 = williamson_xpxp(V)
    print("Db:", Db)
    #print("williamson error:", error1)
    K1, Delta, K2, error2 = blochmessiah_xpxp(S)
    #print("K1:", K1)
    #print("Delta:", Delta)
    #print("K2:", K2)
    #print(error2)

    r1 = np.abs(np.log(Delta[0, 0]))
    r2 = np.abs(np.log(Delta[2, 2]))
    U_squeeze1 = qt.tensor(qt.squeeze(Nt, r1), qt.squeeze(Nt, r2))
    Uo1 = passive_K_to_gate(K1, Nt)
    Uo2 = passive_K_to_gate(K2, Nt)
    return Uo2 * U_squeeze1 * Uo1


def pairwise_difference(rho1, rho2):
    rho_diff = (rho1 - rho2).full()
    return np.max(np.abs(rho_diff))


def avg_photon_number(state, mode: int | None = None):
    """
    Average photon number ⟨a†a⟩ for a state (ket or density matrix).
    Works for single- and multi-mode Fock-space states.

    Parameters
    ----------
    state : qutip.Qobj
        |ψ⟩ or ρ in the Fock basis.
    mode : int | None
        Which mode to measure if the state is multimode.
        If None, assumes single-mode.

    Returns
    -------
    float
    """
    # Hilbert-space dimensions per (left) subsystem
    left_dims = state.dims[0]

    if len(left_dims) == 1:
        # single-mode
        N = left_dims[0]
        return float(qt.expect(qt.num(N), state))

    # multi-mode
    if mode is None:
        raise ValueError("For multimode states, specify `mode` (0..M-1).")

    ops = []
    for m, N in enumerate(left_dims):
        ops.append(qt.num(N) if m == mode else qt.qeye(N))
    n_op = qt.tensor(ops)
    return float(qt.expect(n_op, state))


def bosonic_pure_loss_channel_BSD_2mode(state_MM, eta, Nt):
    """
     beam splitter dilation
    :param rho:
    :param eta: eta = cos(theta) **2 # this is different from the transduction protocol
    :return:
    """
    theta = np.arccos(np.sqrt(eta))

    state_E = qt.basis(Nt, 0)

    U_BS = unitary_beam_splitter(theta, 0, Nt)

    state_MMEE = qt.tensor(state_MM, state_E, state_E)
    U_BS = U_BS.full()
    state_MMEE = state_MMEE.full().reshape(Nt, Nt, Nt, Nt)

    state_MEME = np.einsum("abcd->acbd", state_MMEE)
    state_MEME = state_MEME.reshape(Nt*Nt, Nt*Nt)

    state_MEME = np.einsum("ab,bc->ac", U_BS, state_MEME) # first mode
    state_MEME = np.einsum("ab,cb->ca", U_BS, state_MEME) # second mode
    state_MEME = state_MEME.reshape(Nt*Nt*Nt*Nt, 1)


    state_MEME = qt.Qobj(state_MEME, dims=[[Nt, Nt, Nt,Nt], [1, 1, 1, 1]])
    rho_MM = qt.ptrace(state_MEME, [0, 2])
    return rho_MM
