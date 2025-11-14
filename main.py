import colorsys

import matplotlib
import torch
import numpy as np
import scipy
import math
import scipy.linalg as linalg
from scipy.special import assoc_laguerre
from scipy.integrate import dblquad
import qutip as qt
from multiprocessing import Pool
from datetime import datetime
import itertools
import random
import math
from scipy.integrate import quad
from torch.optim.lr_scheduler import StepLR
from scipy.integrate import simps
from scipy.linalg import logm, sqrtm, schur, block_diag, expm
from collections import Counter

# define some  basic operators in pytorch
paulio = torch.tensor([[1, 0], [0, 1]], dtype=torch.complex128)
paulix = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
pauliy = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
pauliz = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)


def dagger(A):
    return A.conj().t()


def qt2torch(qt_object):
    return torch.tensor(qt_object.full(), dtype=torch.complex128, requires_grad=False)


def destroy(N):
    a = qt.destroy(N)
    return qt2torch(a)


def create(N):
    a_dag = qt.create(N)
    return qt2torch(a_dag)


def position(N):
    a, a_dag = qt.destroy(N), qt.create(N)
    q = (a + a_dag) / np.sqrt(2)
    return qt2torch(q)


def momentum(N):
    a, a_dag = qt.destroy(N), qt.create(N)
    p = 1j * (a_dag - a) / np.sqrt(2)
    return qt2torch(p)


def q_displacemetn(r, Nt):
    """

    :param r:
    :return: exp(-1j* r* operator(p))
    """
    EM = -1j * r * momentum(Nt)
    eig, T = torch.linalg.eig(EM)  # diagonalizing Exponent matrix
    eig = eig.to(torch.complex128)
    X = T @ torch.diag_embed(torch.exp(eig)) @ dagger(T)

    return X


def complex_displacement_operator(alpha_real, alpha_img, Nt):
    """
    :param alpha_real: the real part of alpha
    :param alpha_img: the imaginary part of alpha
    :param Nt: the truncation dimension
    :return: the displacement operatorD(alpha) representation in the number basis
    """

    x = position(Nt)
    p = momentum(Nt)

    eigx, dx = torch.linalg.eigh(x)  # diagonalizing x
    Dx = torch.mm(torch.mm(dx, torch.diag_embed(torch.exp(1j * np.sqrt(2) * alpha_img * eigx))),
                  dagger(dx))  # displacement in x

    eigp, dp = torch.linalg.eigh(p)  # diagonalizing p
    Dp = torch.mm(torch.mm(dp, torch.diag_embed(torch.exp(-1j * np.sqrt(2) * alpha_real * eigp))),
                  dagger(dp))  # displacement in p

    # D = torch.mm(Dx, Dp) * torch.exp(-1j * alpha_img * alpha_real)
    cmu = torch.diag(torch.mm(x, p) - torch.mm(p, x))
    # print("cmu:", cmu)
    D = torch.mm(Dx, Dp) @ torch.diag_embed(torch.exp(-alpha_real * alpha_img * cmu))
    return D


def squeezed_operator(r, theta, Nt):
    """

    :param r:
    :param theta:
    :return: the matrix representation of S(xi) = exp( (xi^* a^2- xi a^dagger^2 )/2 ), xi = r exp(i theta)
    """
    xi = r * torch.exp(1j * theta)
    EM = (torch.conj(xi) * destroy(Nt) @ destroy(Nt) - xi * create(Nt) @ create(Nt)) / 2
    eig, T = torch.linalg.eig(EM)  # diagonalizing Exponent matrix
    # print(np.around(dagger(T) @ EM @ T, 2))
    eig = eig.to(torch.complex128)
    S = T @ torch.diag_embed(torch.exp(eig)) @ dagger(T)
    return S


def unitary_beam_splitter(theta, Nt):
    """

    :param theta:
    :param Nt: cutoff dimension
    :return:
    """
    a = destroy(Nt)
    adg = create(Nt)
    b = destroy(Nt)
    bdg = create(Nt)

    H = theta * (-torch.kron(a, bdg) + torch.kron(adg, b))  # the Exponent matrix
    """
    eig, T = torch.linalg.eigh(H)  # diagonalizing Exponent matrix
    # print(np.around(dagger(T) @ H @ T, 2))
    eig = eig.to(torch.complex128)
    U = T @ torch.diag_embed(torch.exp(-1j * eig)) @ dagger(T)
    """

    U2 = torch.matrix_exp(1 * H)
    return U2


def unitary_phase_rotation(theta, Nt):
    n = torch.arange(0, Nt)
    R = torch.diag_embed(torch.exp(-1j * theta * n))
    R = R.to(torch.complex128)
    return R


def universal_Gaussian(parameters, Nt):
    U1 = unitary_beam_splitter(parameters[0], Nt)
    U2 = torch.kron(squeezed_operator(parameters[1], torch.tensor(0), Nt) @ unitary_phase_rotation(parameters[2], Nt),
                    torch.eye(Nt))
    U3 = unitary_beam_splitter(parameters[3], Nt)
    return U3 @ U2 @ U1


def kerr(theta, Nt):
    n = torch.arange(0, Nt)
    H = n * n
    K = torch.diag_embed(torch.exp(-1j * theta * H))
    K = K.to(torch.complex128)
    return K


def SUM_gate(Nt):
    H = torch.kron(position(Nt), momentum(Nt))
    U = torch.matrix_exp(-1j * H)
    return U


# ECD gate acting on one qubit and one mode







def beam_splitter_matrix(theta, phi1, phi2, phi3=torch.tensor([0])):
    theta = theta.reshape(1, 1)
    phi1 = phi1.reshape(1, 1)
    phi2 = phi2.reshape(1, 1)
    r1 = torch.cat(
        (torch.cos(theta) * torch.cos(phi1), -torch.cos(theta) * torch.sin(phi1), torch.cos(phi2) * torch.sin(theta),
         -torch.sin(theta) * torch.sin(phi2)), 1)
    r2 = torch.cat(
        (torch.cos(theta) * torch.sin(phi1), torch.cos(theta) * torch.cos(phi1), torch.sin(theta) * torch.sin(phi2),
         torch.cos(phi2) * torch.sin(theta)), 1)
    r3 = torch.cat((-torch.cos(phi1 + phi3) * torch.sin(theta), torch.sin(theta) * torch.sin(phi1 + phi3),
                    torch.cos(theta) * torch.cos(phi2 + phi3), -torch.cos(theta) * torch.sin(phi2 + phi3)), 1)
    r4 = torch.cat((-torch.sin(theta) * torch.sin(phi1 + phi3), -torch.cos(phi1 + phi3) * torch.sin(theta),
                    torch.cos(theta) * torch.sin(phi2 + phi3), torch.cos(theta) * torch.cos(phi2 + phi3)), 1)
    matrix = torch.cat((r1, r2, r3, r4), 0)

    """  
    theta = theta.reshape(1, 1)
    phi1 = phi1.reshape(1, 1)
    phi2 = phi2.reshape(1, 1)
    r1 = torch.cat((torch.cos(theta) * torch.cos(phi1), -torch.cos(theta) * torch.sin(phi1),
                    torch.cos(phi2) * torch.sin(theta), -torch.sin(theta) * torch.sin(phi2)), 1)
    r2 = torch.cat((torch.cos(theta) * torch.sin(phi1), torch.cos(theta) * torch.cos(phi1),
                    torch.sin(theta) * torch.sin(phi2), torch.cos(phi2) * torch.sin(theta)), 1)
    r3 = torch.cat((-torch.cos(phi2) * torch.sin(theta), -torch.sin(theta) * torch.sin(phi2),
                    torch.cos(theta) * torch.cos(phi1), torch.cos(theta) * torch.sin(phi1)), 1)
    r4 = torch.cat((torch.sin(theta) * torch.sin(phi2), -torch.cos(phi2) * torch.sin(theta),
                    -torch.cos(theta) * torch.sin(phi1), torch.cos(theta) * torch.cos(phi1)), 1)
    matrix = torch.cat((r1, r2, r3, r4), 0)
    """

    matrix = matrix.to(torch.float32)

    return matrix






def polar_to_cartesian(polar_coordinates):
    """
    Convert polar coordinates to Cartesian coordinates in n-dimensional space.

    :param polar_coordinates: A PyTorch tensor of polar coordinates [r, theta_1, theta_2, ..., theta_(n-1)]
                              where r is the radial distance and theta_i are the angles in radians.
    :return: A PyTorch tensor representing the Cartesian coordinates [x_1, x_2, ..., x_n].
    """
    r = polar_coordinates[0]
    angles = polar_coordinates[1:]

    cartesian_coords = []
    for i in range(len(angles)):
        cartesian_coords.append(r * torch.cos(angles[i]))
        r = r * torch.sin(angles[i])
    cartesian_coords.append(r)

    return torch.stack(cartesian_coords)


default_colors = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#7f7f7f',  # Gray
    '#bcbd22',  # Olive
    '#17becf'  # Cyan
]






def matrix_one_norm(rho):
    A = rho @ np.conjugate(np.transpose(rho))
    return np.trace(scipy.linalg.sqrtm(A))


def matrix_one_norm_torch(rho):
    A = rho @ dagger(rho)
    eigenvalues, eigenvectors = torch.linalg.eig(A)
    sqrt_eigenvalues = torch.sqrt(eigenvalues)
    # sqrt_A = eigenvectors @ torch.diag(sqrt_eigenvalues) @ dagger(eigenvectors)

    return torch.sum(sqrt_eigenvalues)


def trace_distance(rho1, rho2):
    return 0.5 * matrix_one_norm(rho1 - rho2)


def trace_distance_torch(rho1, rho2):
    return 0.5 * matrix_one_norm_torch(rho1 - rho2)


def helstrom_limit(rho1, rho2):
    return 0.5 * (1 - trace_distance(rho1, rho2))


def helstrom_limit_torch(rho1, rho2):
    return 0.5 * (1 - trace_distance_torch(rho1, rho2))


def displacement_discrimination_HL(inputrho, data, type):
    Nt = inputrho.shape[0]
    rho0 = np.zeros((Nt, Nt))
    rho1 = np.zeros((Nt, Nt))
    if type == '1dlinear':
        data0 = data[data[:, 1] == 0]
        for k in range(data0.shape[0]):
            alpha = data0[k, 0]
            D = complex_displacement_operator(alpha, 0, Nt).detach().numpy()
            rho0 = rho0 + D @ inputrho @ np.conjugate(np.transpose(D))
        rho0 = rho0 / data0.shape[0]

        data1 = data[data[:, 1] == 1]
        for k in range(data1.shape[0]):
            alpha = data1[k, 0]
            D = complex_displacement_operator(alpha, 0, Nt).detach().numpy()
            rho1 = rho1 + D @ inputrho @ np.conjugate(np.transpose(D))
        rho1 = rho1 / data1.shape[0]
    elif type == '1dcomplexnonlinear':
        data0 = data[data[:, 2] == 0]
        for k in range(data0.shape[0]):
            alpha1 = data0[k, 0]
            alpha2 = data0[k, 1]
            D = complex_displacement_operator(alpha1, alpha2, Nt).detach().numpy()
            rho0 = rho0 + D @ inputrho @ np.conjugate(np.transpose(D))
        rho0 = rho0 / data0.shape[0]

        data1 = data[data[:, 2] == 1]
        for k in range(data1.shape[0]):
            alpha1 = data1[k, 0]
            alpha2 = data1[k, 1]
            D = complex_displacement_operator(alpha1, alpha2, Nt).detach().numpy()
            rho1 = rho1 + D @ inputrho @ np.conjugate(np.transpose(D))
        rho1 = rho1 / data1.shape[0]
    else:
        raise ValueError("wrong type")
    hl = helstrom_limit(rho0, rho1)
    return np.real(hl)


def continuous_color_transition(num_colors, start_color="#FF0000"):
    color_list = []

    # Convert start color from hex to RGB
    start_rgb = tuple(int(start_color[i:i + 2], 16) for i in (1, 3, 5))
    start_hsv = colorsys.rgb_to_hsv(*[val / 255.0 for val in start_rgb])

    for i in range(num_colors):
        hue = (start_hsv[0] + i / num_colors) % 1.0
        rgb_color = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        int_color = tuple(int(val * 255) for val in rgb_color)
        hex_color = "#{:02x}{:02x}{:02x}".format(*int_color)
        color_list.append(hex_color)

    return color_list


def colorlist_generator(num, cmap):
    norm = matplotlib.colors.Normalize(vmin=0, vmax=num + 1)
    mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorlist = np.array([(mapper.to_rgba(v)) for v in range(1, num + 1)])
    return colorlist


def entanglement_entropy(state, dim1, dim2, mixed):
    if not mixed:
        rho = state @ (state.conj().t())
    rho_reshaped = rho.view(dim1, dim2, dim1, dim2)
    rho_A = torch.einsum('ijik->jk', rho_reshaped)
    eigenvalues_A, _ = torch.linalg.eigh(rho_A)
    eigenvalues_A = eigenvalues_A.clamp(min=1e-12)
    entropyA = -torch.sum(eigenvalues_A * torch.log2(eigenvalues_A))
    return entropyA


def ECD_state_preparation_2mode(parameters_ECD, depth, Nt):
    dimension = 2 * Nt * Nt
    state = torch.zeros((dimension, 1), dtype=torch.complex128)
    state[5, 0] = 1 / np.sqrt(2)
    state[5 * Nt, 0] = 1 / np.sqrt(2)

    alphas_t0_p = parameters_ECD[0 * depth:2 * depth]  # alphas for the  1st mode in preparation
    alphas_t1_p = parameters_ECD[2 * depth:4 * depth]  # alphas for the  2nd mode in preparation
    betas_t0_p = parameters_ECD[4 * depth:6 * depth]  # betas for the  1st mode in preparation
    betas_t1_p = parameters_ECD[6 * depth:8 * depth]  # betas for the  2nd mode in preparation

    for i in range(depth):
        state = ECD_unitary_QMM(alpha1=alphas_t0_p[2 * i], alpha2=alphas_t0_p[2 * i + 1],
                                psi=betas_t0_p[2 * i], theta=betas_t0_p[2 * i + 1],
                                Nt=Nt, target=0, state=state)
        state = ECD_unitary_QMM(alpha1=alphas_t1_p[2 * i], alpha2=alphas_t1_p[2 * i + 1],
                                psi=betas_t1_p[2 * i], theta=betas_t1_p[2 * i + 1],
                                Nt=Nt, target=1, state=state)

    # the energy of the input state
    n_in = energy_n_QMM(state, Nt)

    state_in = state.clone()
    state0in = state[0:Nt * Nt]
    state1in = state[Nt * Nt:2 * Nt * Nt]
    p0_in = torch.sum(torch.abs(state0in) ** 2)
    p1_in = torch.sum(torch.abs(state1in) ** 2)
    state0in = state0in / torch.sqrt(p0_in)
    state1in = state1in / torch.sqrt(p1_in)
    return state0in


def ECD_GaussianGate_Entanglement(parameters_ECD, parameters_Gaussian, depth, Nt, ):
    dimension = 2 * Nt * Nt
    state = torch.zeros((dimension, 1), dtype=torch.complex128)
    state[0, 0] = 1

    alphas_t0_p = parameters_ECD[0 * depth:2 * depth]  # alphas for the  1st mode in preparation
    alphas_t1_p = parameters_ECD[2 * depth:4 * depth]  # alphas for the  2nd mode in preparation
    betas_t0_p = parameters_ECD[4 * depth:6 * depth]  # betas for the  1st mode in preparation
    betas_t1_p = parameters_ECD[6 * depth:8 * depth]  # betas for the  2nd mode in preparation

    for i in range(depth):
        state = ECD_unitary_QMM(alpha1=alphas_t0_p[2 * i], alpha2=alphas_t0_p[2 * i + 1],
                                psi=betas_t0_p[2 * i], theta=betas_t0_p[2 * i + 1],
                                Nt=Nt, target=0, state=state)
        state = ECD_unitary_QMM(alpha1=alphas_t1_p[2 * i], alpha2=alphas_t1_p[2 * i + 1],
                                psi=betas_t1_p[2 * i], theta=betas_t1_p[2 * i + 1],
                                Nt=Nt, target=1, state=state)

    # the energy of the input state

    state0in = state[0:Nt * Nt]
    state1in = state[Nt * Nt:2 * Nt * Nt]
    p0_in = torch.sum(torch.abs(state0in) ** 2)
    p1_in = torch.sum(torch.abs(state1in) ** 2)
    state0in = state0in / torch.sqrt(p0_in)
    state1in = state1in / torch.sqrt(p1_in)
    n0in = energy_n_MM(state0in, Nt)

    state_out = universal_Gaussian(parameters_Gaussian, Nt) @ state0in
    rho = state_out @ (state_out.conj().t())
    rho_reshaped = rho.view(Nt, Nt, Nt, Nt)
    rho_A = torch.einsum('ijik->jk', rho_reshaped)

    eigenvalues_A, _ = torch.linalg.eigh(rho_A)
    eigenvalues_A = eigenvalues_A.clamp(min=1e-12)
    entropyA = -torch.sum(eigenvalues_A * torch.log2(eigenvalues_A))

    """
    rho_B = torch.einsum('jiki->jk', rho_reshaped)
    eigenvalues_B, _ = torch.linalg.eigh(rho_B)
    eigenvalues_B = eigenvalues_B.clamp(min=1e-12)
    entropyB = -torch.sum(eigenvalues_B * torch.log(eigenvalues_B))
    """
    return entropyA, n0in


def ECD_NonGaussian_Entanglement(parameters_ECD, depth, Nt, num_steps=500):
    y_init = torch.rand(4)
    y = y_init.clone().requires_grad_(True)  # y needs to require gradients
    optimizer = torch.optim.Adam([y], lr=0.001)
    entropy_list = []

    for step in range(num_steps):
        optimizer.zero_grad()
        loss = ECD_GaussianGate_Entanglement(parameters_ECD, y, depth, Nt, )[0]  # Compute f(x, y)
        loss.backward()  # Backpropagate the gradients
        optimizer.step()  # Update y
        entropy = loss.clone()
        entropy_list.append(entropy.detach().item())
        print(step, entropy.detach().item())

    return np.min(entropy_list)


def symplectic_matrix(num_modes):
    omega_2x2 = np.array([[0, 1], [-1, 0]])

    # Use the Kronecker product to generate the full symplectic matrix for N modes
    symplectic_matrix = np.kron(np.eye(num_modes), omega_2x2)

    return symplectic_matrix


def williamson_xpxp(V, tol=1e-11):
    r"""Williamson decomposition of positive-definite (real) symmetric matrix.

    See :ref:`williamson`.

    Note that it is assumed that the symplectic form is

    .. math:: \Omega = \begin{bmatrix}0&I\\-I&0\end{bmatrix}

    where :math:`I` is the identity matrix and :math:`0` is the zero matrix.

    See https://math.stackexchange.com/questions/1171842/finding-the-symplectic-matrix-in-williamsons-theorem/2682630#2682630

    Args:
        V (array[float]): positive definite symmetric (real) matrix
        tol (float): the tolerance used when checking if the matrix is symmetric: :math:`|V-V^T| \leq` tol

    Returns:
        tuple[array,array]: ``(Db, S)`` where ``Db`` is a diagonal matrix
            and ``S`` is a symplectic matrix such that :math:`V = S^T Db S`
    """
    (n, m) = V.shape

    if n != m:
        raise ValueError("The input matrix is not square")

    diffn = np.linalg.norm(V - np.transpose(V))

    if diffn >= tol:
        raise ValueError("The input matrix is not symmetric")

    if n % 2 != 0:
        raise ValueError("The input matrix must have an even number of rows/columns")

    n = n // 2
    omega = symplectic_matrix(n)
    vals = np.linalg.eigvalsh(V)

    for val in vals:
        if val <= 0:
            raise ValueError("Input matrix is not positive definite")

    Mm12 = sqrtm(np.linalg.inv(V)).real

    r1 = Mm12 @ omega @ Mm12
    s1, K = schur(r1)

    X = np.array([[0, 1], [1, 0]])
    I = np.identity(2)
    seq = []

    for i in range(n):
        if s1[2 * i, 2 * i + 1] > 0:
            seq.append(I)
        else:
            seq.append(X)

    p = block_diag(*seq)
    Kt = K @ p
    s1t = p @ s1 @ p
    O = np.transpose(Kt)

    Db = np.diag([1 / s1t[2 * i, 2 * i + 1] for i in range(n) for _ in range(2)])

    S = sqrtm(Db) @ O @ Mm12
    return Db, S


def measure_observable(state, operator, num_samples):
    if state.isket:
        # Pure state: Ensure state is normalized
        state = state.unit()

        # Get the eigenstates and eigenvalues of the operator
        eigenvalues, eigenstates = operator.eigenstates()

        # Calculate probabilities of measurement outcomes for pure state
        probabilities = [abs((eigenstate.dag() * state).full()[0, 0]) ** 2 for eigenstate in eigenstates]

    elif state.isoper:
        # Mixed state: Ensure state is a valid density matrix
        state = state / state.tr()

        # Get the eigenstates and eigenvalues of the operator
        eigenvalues, eigenstates = operator.eigenstates()

        # Calculate probabilities of measurement outcomes for mixed state
        probabilities = [np.real((eigenstate.dag() * state * eigenstate).full()[0, 0]) for eigenstate in eigenstates]

    else:
        raise TypeError("The input state must be either a pure state (ket) or a mixed state (density matrix).")

    # Get the eigenstates and eigenvalues of the operator

    # Calculate probabilities of measurement outcomes

    # Ensure probabilities sum to 1 (fix small numerical errors)
    probabilities = np.array(probabilities)
    probabilities = np.around(probabilities, 10)
    probabilities /= probabilities.sum()

    # deal with degeneracy
    unique_eigenvalues = []
    unique_probabilities = []
    eigenvalues = np.around(eigenvalues, 10)
    #print(np.sum(np.array(probabilities) * eigenvalues))

    for value in np.unique(eigenvalues):
        # Find the indices of the repeated values
        indices = np.where(eigenvalues == value)[0]
        # Calculate the combined probability
        combined_probability = np.sum(probabilities[indices])

        # Append the value and the combined probability
        unique_eigenvalues.append(value)
        unique_probabilities.append(combined_probability)

    # Generate measurement samples based on probabilities
    samples = np.random.choice(unique_eigenvalues, size=num_samples, p=unique_probabilities)

    # Count occurrences of each eigenvalue
    sample_counts = Counter(samples)

    # Prepare result as list of tuples (eigenvalue, frequency, eigenstate)
    result = []
    counted_values = set()
    frequency_list = []
    for eigenvalue in unique_eigenvalues:
        frequency = sample_counts[eigenvalue] / num_samples
        frequency_list.append(frequency)
        result.append((eigenvalue, frequency))

        """
        for eigenvalue, eigenstate in zip(eigenvalues, eigenstates):
                if eigenvalue not in counted_values:
            frequency = sample_counts[eigenvalue] / num_samples
            counted_values.add(eigenvalue)
        else:
            frequency = 0
        result.append((eigenvalue, frequency, eigenstate))
        """
    #print(unique_eigenvalues, np.sum(np.abs(np.array(frequency_list)-np.array(unique_probabilities))))
    #exact_expect = np.sum(np.array(unique_eigenvalues) * np.array(unique_probabilities))
    #exact_sample = np.sum(np.array(unique_eigenvalues) * np.array(frequency_list))
    #print(exact_expect,exact_sample)
    #print(np.abs(np.array(frequency_list)-np.array(unique_probabilities)))
    #print(np.sum(np.abs(np.array(frequency_list)-np.array(unique_probabilities)) * np.array(unique_eigenvalues)))

    return result


def measure_observable2(state, operator, num_samples):
    if state.isket:
        # Pure state: Ensure state is normalized
        state = state.unit()

        # Get the eigenstates and eigenvalues of the operator
        eigenvalues, eigenstates = operator.eigenstates()

        # Calculate probabilities of measurement outcomes for pure state
        probabilities = [abs((eigenstate.dag() * state).full()[0, 0]) ** 2 for eigenstate in eigenstates]

    elif state.isoper:
        # Mixed state: Ensure state is a valid density matrix
        state = state / state.tr()

        # Get the eigenstates and eigenvalues of the operator
        eigenvalues, eigenstates = operator.eigenstates()

        # Calculate probabilities of measurement outcomes for mixed state
        probabilities = [np.real((eigenstate.dag() * state * eigenstate).full()[0, 0]) for eigenstate in eigenstates]

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
    measurement_result = measure_observable(psi, operator, num_samples)
    operator_expect = sum(eigenvalue * freq for eigenvalue, freq in measurement_result)
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


def covariance_matrix(basis, rho):
    first_moments = np.array([qt.expect(oi, rho) for oi in basis], )  #dtype=np.float64
    cm = np.array([[0.5 * qt.expect(oi * oj + oj * oi, rho) - qt.expect(oi, rho) * qt.expect(oj, rho)
                    for oi in basis] for oj in basis], )  #dtype=np.float64

    return first_moments, cm


def calculate_R_rho(measurement_result, rho, operator):
    # Get the eigenstates of the operator
    eigenvalues, eigenstates = operator.eigenstates()

    # Initialize R(\rho) and G with zeros, ensuring the dimensions match rho
    R_rho = qt.Qobj(np.zeros((rho.shape[0], rho.shape[1])), dims=rho.dims)
    G = qt.Qobj(np.zeros((rho.shape[0], rho.shape[1])), dims=rho.dims)

    # Iterate over measurement results to calculate R(\rho) and G
    for eigenvalue, frequency, eigenstate in measurement_result:
        # Projector onto the eigenstate
        projector = eigenstate * eigenstate.dag()

        # Expand the projector if necessary to match the tensor space of rho
        if projector.dims != rho.dims:
            projector = qt.tensor(*[projector] * len(rho.dims[0]))

        # Add contribution to R(\rho) only if the denominator is non-zero

        denominator = abs((eigenstate.dag() * rho * eigenstate).full()[0, 0])
        if denominator > 1e-10:
            R_rho += (frequency / denominator) * projector

        # Add contribution to G only if frequency is non-zero
        if frequency > 0:
            G += projector

    return R_rho, G


def RrhoR(R, rho, epsilon=10):
    # Calculate the R(\rho)R operator
    if epsilon < 1:
        I = qt.qeye(rho.dims[0][0])
        I = qt.tensor(*[I] * len(rho.dims[0]))

        R = (I + epsilon * R) / (1 + epsilon)

    rho = R * rho * R

    rho = rho / rho.tr()
    return rho


def RrhoR_state_tomograph_multi_mode(psi, Nt, num_samples, num_modes, num_thetas=20):
    rho = qt.qeye(Nt) / Nt
    rho = qt.tensor(*[rho] * num_modes)
    Infidelity_list = [1]
    Tracedistance_list = [1]
    state_list = [rho]
    operator_list = []
    #for i, theta in enumerate(np.linspace(0, 2 * np.pi, num_thetas)):
    for i in range(num_thetas):
        operators = []
        for _ in range(num_modes):
            theta = np.random.rand() * 2 * np.pi
            operator = (np.exp(1j * theta) * qt.create(Nt) + np.exp(-1j * theta) * qt.destroy(Nt)) / np.sqrt(2)
            operators.append(operator)

        total_operator = qt.tensor(*operators)
        #theta = np.random.rand() * 2 * np.pi
        #operator = (np.exp(1j * theta) * qt.create(Nt) + np.exp(-1j * theta) * qt.destroy(Nt)) / np.sqrt(2)
        #operator = qt.tensor(*[operator] * num_modes)
        operator_list.append(total_operator)
    #operator_list.append(qt.tensor(*[qt.num(Nt)] * num_modes))
    for j, operator in enumerate(operator_list):
        measurement_result = measure_observable(psi, operator, num_samples)
        R_rho, G = calculate_R_rho(measurement_result, rho, operator)
        rho = RrhoR(R_rho, rho)
        Infidelity = 1 - qt.fidelity(rho, psi)
        tracedistance = qt.metrics.tracedist(rho, psi)

        """
                  if Infidelity > errorlist[-1]:
            rho = RrhoR(R_rho, state_list[-1], 0.1)
            Infidelity = 1 - qt.fidelity(rho, psi)      
        """

        Tracedistance_list.append(tracedistance)
        Infidelity_list.append(Infidelity)
        state_list.append(rho)
        #print(j, f"Infidelity: {Infidelity:.4f}", f"trace distance: {tracedistance:.4f}", )
    return Tracedistance_list[-1], Infidelity_list[-1], state_list[-1]


def RrhoR_state_tomograph_multi_mode2(psi, Nt, num_samples, num_modes, num_interation, save, savepath):
    # measure real state psi
    state_list = []
    for i in range(num_samples):
        operators = []
        for _ in range(num_modes):
            theta = np.random.rand() * 2 * np.pi
            operator = (np.exp(1j * theta) * qt.create(Nt) + np.exp(-1j * theta) * qt.destroy(Nt)) / np.sqrt(2)
            operators.append(operator)

        total_operator = qt.tensor(*operators)
        measurement_result = measure_observable2(psi, total_operator, 1)
        index = np.where(np.array(measurement_result[0]) > 0)
        state = measurement_result[3][index][0]
        state_list.append(state)
        #state = qt.Qobj(state,dims=[[Nt],[1]])
        #projector = state * (state.dag())
        #projector_list.append(projector)

    # iteration
    rho = qt.qeye(Nt) / Nt
    rho = qt.tensor(*[rho] * num_modes)
    Infidelity_list = [1 - qt.fidelity(rho, psi)]
    Tracedistance_list = [qt.metrics.tracedist(rho, psi)]
    rho_list = [rho]

    for i in range(num_interation):
        R_rho = qt.Qobj(np.zeros((rho.shape[0], rho.shape[1])), dims=rho.dims)
        for j in range(len(state_list)):
            state = state_list[j]
            pj = state * state.dag()
            R_rho = R_rho + pj / ((rho * pj).tr())
        rho = RrhoR(R_rho, rho, 0.1)
        Infidelity = 1 - qt.fidelity(rho, psi)
        tracedistance = qt.metrics.tracedist(rho, psi)

        print("Iteration:", i, "infidelity:", Infidelity, "Trace distace:", tracedistance)

        Tracedistance_list.append(tracedistance)
        Infidelity_list.append(Infidelity)
        rho_list.append(rho)
        if save:
            np.save(savepath, [Tracedistance_list, Infidelity_list])
    return Tracedistance_list[-1], Infidelity_list[-1], rho_list[-1]


def symplectic_matrix(num_modes):
    omega_2x2 = np.array([[0, 1], [-1, 0]])

    # Use the Kronecker product to generate the full symplectic matrix for N modes
    symplectic_matrix = np.kron(np.eye(num_modes), omega_2x2)

    return symplectic_matrix


def U_s_gate(V, num_modes, quadratures):
    V = np.around(V, 5)
    Omega = symplectic_matrix(num_modes)

    Db, S = williamson_xpxp(V)
    #print(S @ V @ np.transpose(S))
    print("S Eig")
    print(np.linalg.eig(S)[0])

    # Construct xi^T S log(S) xi
    D1 = Omega @ logm(S)  # Compute log(S)
    #D1 = (np.transpose(D1) + D1)/2
    H = 0
    for i in range(2 * num_modes):
        for j in range(2 * num_modes):
            H += quadratures[i] * D1[i, j] * quadratures[j]

    # Construct the unitary transformation
    H = (-1j / 4) * H
    U_s = H.expm()
    return U_s


from scipy.linalg import sqrtm




from itertools import permutations

def diagonalize_order(S):
    Omega = symplectic_matrix(2)
    eigenvalues, eigenvectors = np.linalg.eig(S)
    perm = list(permutations([0,1,2,3]))
    perm_array = np.array(perm)
    #final_eigenvalues, final_eigenvectors = eigenvalues, eigenvectors
    for permutation in perm_array:
        sorted_eigenvalues = eigenvalues[permutation]
        sorted_eigenvectors = eigenvectors[:, permutation]
        error = sorted_eigenvectors @ Omega @ np.transpose(sorted_eigenvectors) - Omega
        error = np.sum(np.abs(error))
        print(permutation,error)
        if error<1e-10:
            final_eigenvalues = sorted_eigenvalues
            final_eigenvectors = sorted_eigenvectors
            break

    return final_eigenvalues, final_eigenvectors


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


def unitary_phase_rotation_qt(theta, Nt):
    return (-1j * theta * qt.num(Nt)).expm()



def bms_cov_to_gate(cov, Nt):
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
    U = unitary_beam_splitter(s, Nt) * U
    U = qt.tensor(unitary_phase_rotation_qt(t3, Nt), unitary_phase_rotation_qt(t4, Nt)) * U

    return U


def U_s_gate2(V, num_modes, quadratures):
    Nt = quadratures[0].dims[0][0]
    V = np.around(V, 5)
    Omega = symplectic_matrix(num_modes)

    Db, S = williamson_xpxp(V)
    print("Db")
    print(Db)
    Sigma = sqrtm(S @ np.transpose(S))
    U = np.linalg.inv(Sigma) @ S

    Delta, O = diagonalize_order(Sigma)
    Oprime = np.linalg.inv(O) @ U

    Delta = np.diag(Delta)
    print("Delta")
    print(Delta)
    print("O")
    print(O)
    print("Oprime")
    print(Oprime)
    print("S- O*Delta*Oprime")
    print(S - O @ Delta @ Oprime)

    print("Sympletic O")
    print(np.around(O @ Omega @ np.transpose(O), 3))
    print("Orthogonal O")
    print(np.around(O @ np.transpose(O), 3))

    print("Sympletic Oprime")
    print(np.around(Oprime @ Omega @ np.transpose(Oprime), 3))
    print("Orthogonal Oprime")
    print(np.around(Oprime @ np.transpose(Oprime), 3))

    r1 = np.abs(np.log(Delta[0, 0]))
    r2 = np.abs(np.log(Delta[2, 2]))

    U_squeeze1 = qt.tensor(qt.squeeze(Nt, r1), qt.squeeze(Nt, r2))
    U_squeeze2 = qt.tensor(qt.squeeze(Nt, -r1), qt.squeeze(Nt, r2))
    U_squeeze3 = qt.tensor(qt.squeeze(Nt, r1), qt.squeeze(Nt, -r2))
    U_squeeze4 = qt.tensor(qt.squeeze(Nt, -r1), qt.squeeze(Nt, -r2))

    print("Decompose Oprime")
    Uo1 = bms_cov_to_gate(Oprime, Nt)
    print("Decompose O")
    Uo2 = bms_cov_to_gate(O, Nt)
    return Uo2 * U_squeeze1 * Uo1,Uo2 * U_squeeze2 * Uo1,Uo2 * U_squeeze3 * Uo1,Uo2 * U_squeeze4 * Uo1


def unitary_beam_splitter(theta, Nt):
    a = qt.tensor(qt.destroy(Nt), qt.qeye(Nt))
    adg = qt.tensor(qt.create(Nt), qt.qeye(Nt))
    b = qt.tensor(qt.qeye(Nt), qt.destroy(Nt))
    bdg = qt.tensor(qt.qeye(Nt), qt.create(Nt))

    H = theta * (adg * b - a * bdg)  # the Exponent matrix
    U = H.expm()
    return U


def two_mode_squeezed_vacuum(r, theta, Nt):
    psi = 0
    for n in range(Nt):
        coeff = (-np.exp(1j * theta) * np.tanh(r)) ** n / np.cosh(r)
        psi_n = qt.tensor(qt.basis(Nt, n), qt.basis(Nt, n))  # Tensor product of |n>_1 and |n>_2
        psi += coeff * psi_n

    return psi.unit()


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
        if i == index:
            x = qt.create(Nt) + qt.destroy(Nt)
            p = -1j * (qt.create(Nt) - qt.destroy(Nt))
            position_operators.append(x)
            momentum_operators.append(p)
        else:
            eye = qt.qeye(Nt)
            position_operators.append(eye)
            momentum_operators.append(eye)

    # Ensure all elements are Qobj instances
    position_operators = [op if isinstance(op, qt.Qobj) else qt.Qobj(op) for op in position_operators]
    momentum_operators = [op if isinstance(op, qt.Qobj) else qt.Qobj(op) for op in momentum_operators]

    # Take the tensor product
    position_operator = qt.tensor(*position_operators)
    momentum_operator = qt.tensor(*momentum_operators)

    """
    x = qt.create(Nt) + qt.destroy(Nt)
    p = -1j * (qt.create(Nt) - qt.destroy(Nt))
    I = qt.qeye(Nt)
    x1 = qt.tensor(*[x, I, I])
    p1 = qt.tensor(*[p, I, I])
    x2 = qt.tensor(*[I, x, I])
    p2 = qt.tensor(*[I, p, I])
    x3 = qt.tensor(*[I, I, x])
    p3 = qt.tensor(*[I, I, p])
    quadratures = [x1, p1, x2, p2, x3, p3]
    """

    return position_operator, momentum_operator


def generate_random_bosonic_state(Nt, E):
    """
    Generate a one-mode Bosonic state in the number basis such that
    the average photon number is smaller than E and the cutoff dimension is Nt.

    Parameters:
    Nt (int): Cutoff dimension.
    E (int): Energy threshold (average photon number), E < Nt.

    Returns:
    Qobj: Quantum object representing the generated state.
    """
    if E >= Nt:
        raise ValueError("Energy threshold E must be smaller than the cutoff dimension Nt.")

    # Create random coefficients for the superposition of Fock states
    coefficients = np.random.rand(Nt) + 1j * np.random.rand(Nt)

    # Construct the initial state in the number basis
    state = sum(coeff * qt.basis(Nt, n) for n, coeff in enumerate(coefficients))

    # Normalize the initial state
    state = state.unit()

    # Adjust the state to ensure the average photon number is less than E
    photon_number_operator = sum(n * qt.basis(Nt, n) * qt.basis(Nt, n).dag() for n in range(Nt))
    avg_photon_number = qt.expect(photon_number_operator, state)

    # Reduce the coefficients for the basis states larger than E until avg_photon_number <= E
    while avg_photon_number > E:
        for n in range(E, Nt):
            coefficients[n] *= 0.5  # Reduce the coefficient for states larger than E

        # Reconstruct the state with modified coefficients
        state = sum(coeff * qt.basis(Nt, n) for n, coeff in enumerate(coefficients))
        state = state.unit()  # Normalize the state again
        avg_photon_number = qt.expect(photon_number_operator, state)

    return state
