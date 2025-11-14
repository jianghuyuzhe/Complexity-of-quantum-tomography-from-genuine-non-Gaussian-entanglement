import numpy as np

from main_tomography_v2 import *

Nt = 20
num_modes = 2
dim = Nt ** num_modes
K = 1  # number of samples in each angle
d = 11  # number of sampled angles
num_samples = K * (d ** num_modes)  #number of total samples
print("number of total samples:", num_samples)
epsilon = 10
randomization = 0
iteration = 100
eta = 0.9

path = f"Data/two_mode_catstate_RrhoR_noisy_eta={eta}_K={K}_d={d}_randomization={randomization}"

state1 = (qt.coherent(Nt, 0.1) + qt.coherent(Nt, -0.1)).unit()
state2 = (qt.coherent(Nt, 1.0) + qt.coherent(Nt, -1.0)).unit()
#state1 = qt.basis(Nt, 0)
#state2 = qt.basis(Nt, 1)
state = qt.tensor(state1, state2)
"""
rho1 = qt.ket2dm(state1)
rho2 = qt.ket2dm(state2)
rho = qt.tensor(rho1, rho2)
U_BS = unitary_beam_splitter(np.pi / 4, 0, Nt)
rho = U_BS * rho * U_BS.dag()
"""
angles = [np.pi / 4, 0, 0, 0]
U_BS = angles_to_gate(angles, Nt)
state = U_BS * state

rho = bosonic_pure_loss_channel_BSD_2mode(state, eta, Nt)


quadratures = []
for i in range(num_modes):
    xi, pi = position_momentum_operators_multimode(Nt, num_modes, i)
    quadratures.extend([xi, pi])


def RrhoR_measurement_generator(psi, Nt):
    f_list = []
    p_list = []
    state_list = []
    for i1 in range(d):
        theta1 = i1 * np.pi * 2 / d
        operator1 = (np.exp(1j * theta1) * qt.create(Nt) + np.exp(-1j * theta1) * qt.destroy(Nt)) / np.sqrt(2)
        for i2 in range(d):
            theta2 = i2 * np.pi * 2 / d
            operator2 = (np.exp(1j * theta2) * qt.create(Nt) + np.exp(-1j * theta2) * qt.destroy(Nt)) / np.sqrt(2)

            operator = qt.tensor(operator1, operator2)
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


f_list, p_list, state_list = RrhoR_measurement_generator(rho, Nt)
print(len(f_list))
rho0 = qt.qeye(Nt) / Nt
rho0 = qt.tensor(*[rho0] * num_modes)

rho_list = [rho0.full()]
if_list = [1 - qt.fidelity(rho0, state)]
td_list = [qt.metrics.tracedist(rho0, state)]
#pd_list = [pairwise_difference(rho0, state)]


def RrhoR_state_tomograph_iteration():
    rho_i = qt.Qobj(rho_list[-1], dims=[[Nt, Nt], [Nt, Nt]])

    R_rho_i = qt.Qobj(np.zeros((rho_i.shape[0], rho_i.shape[1])), dims=rho_i.dims)
    for j in range(len(state_list)):
        state_j = qt.Qobj(state_list[j], dims=[[Nt, Nt], [1, 1]])
        pro_j = state_j * state_j.dag()
        R_rho_i = R_rho_i + pro_j * f_list[j] * K / ((rho_i * pro_j).tr() * num_samples)

    rho_updated = RrhoR(R_rho_i, rho_i, epsilon=epsilon)
    Infidelity = 1 - qt.fidelity(rho_updated, state)
    tracedistance = qt.metrics.tracedist(rho_updated, state)

    td_list.append(tracedistance)
    if_list.append(Infidelity)
    rho_list.append(rho_updated.full())
    #pd_list.append(pairwise_difference(rho_updated, psi))
    np.save(path + "_if_list", if_list)
    np.save(path + "_td_list", td_list)
    print(len(if_list),
          if_list[-1], np.log10(if_list[-1]),
          td_list[-1], np.log10(td_list[-1]),)


for i in range(iteration):
    RrhoR_state_tomograph_iteration()
