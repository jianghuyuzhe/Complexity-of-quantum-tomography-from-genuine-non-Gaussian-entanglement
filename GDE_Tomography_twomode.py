import numpy as np

from main_tomography_v2 import *

Nt = 20
num_modes = 2
epsilon = 10
d = 11
K = int(1e4)
num_interation = 1000
num_samples_tomo = K * d
num_samples_cov = 0
randomization = 0
path = f"Data/two_mode_numberstate_numsamplescov={num_samples_cov}_K={K}_d={d}_randomization={randomization}_PT"


def process_tomograph(num_samples_cov, randomization):
    #state1 = (qt.coherent(Nt,0.1) + qt.coherent(Nt, -0.1)).unit()
    #state2 = (qt.coherent(Nt, 1.0) + qt.coherent(Nt, -1.0)).unit()
    #state = qt.tensor(state1,state2)
    #state1 = qt.basis(Nt, 0)
    #state2 = qt.basis(Nt, 1)
    #rho1 = qt.ket2dm(state1)
    #rho2 = qt.ket2dm(state2)
    #rho = qt.tensor(rho1, rho2)

    #print(avg_photon_number(state1), avg_photon_number(state2))
    rho = qt.tensor(qt.thermal_dm(Nt, 0.2), qt.thermal_dm(Nt, 0.3), )
    angles = [np.pi / 4, 0, 0, 0]
    U_BS = angles_to_gate(angles, Nt)
    rho = U_BS * rho * U_BS.dag()
    #state = U_BS * state
    #eta = 0.9
    #rho = bosonic_pure_loss_channel_BSD_2mode(state, eta, Nt)


    quadratures = []
    for i in range(num_modes):
        xi, pi = position_momentum_operators_multimode(Nt, num_modes, i)
        quadratures.extend([xi, pi])
    #print(covariance_matrix(basis=quadratures, rho=rho, ))



    if num_samples_cov == 0:
        first_moments1, cm1 = covariance_matrix(basis=quadratures, rho=rho, )
        #print("Covariance Matrix:", np.around(cm1,3))
    else:
        first_moments1, cm1 = measure_covariance_matrix(rho, quadratures, num_samples=num_samples_cov)

    first_moments = first_moments1
    #print("first_moments:", first_moments)
    #print("cm1:", cm1)
    V = cm1

    U_alpha = qt.tensor(qt.displace(Nt, (first_moments[0] + 1j * first_moments[1]) / 2),
                        qt.displace(Nt, (first_moments[2] + 1j * first_moments[3]) / 2))
    U_S = US_gate_2mode(V, Nt)

    rho_disent = U_S.dag() * U_alpha.dag() * rho * U_alpha * U_S
    rho1 = qt.ptrace(rho_disent, 0)
    rho2 = qt.ptrace(rho_disent, 1)

    print("Entropy:", qt.entropy_vn(rho1), qt.entropy_vn(rho2))

    # error0, if0, state = RrhoR_state_tomograph_multi_mode(rho, Nt, num_samples, num_modes, num_thetas=200)
    td1, if1, rho1_rec = RrhoR_state_tomograph_onemode(rho1, Nt, epsilon, d, K, num_interation, )
    td2, if2, rho2_rec = RrhoR_state_tomograph_onemode(rho2, Nt, epsilon, d, K, num_interation, )

    rho_rec = qt.tensor(qt.Qobj(rho1_rec), qt.Qobj(rho2_rec))
    rho_rec = U_alpha * U_S * rho_rec * U_S.dag() * U_alpha.dag()
    td4 = qt.metrics.tracedist(rho, rho_rec)  # 1 - qt.fidelity(state, psi)
    if4 = 1 - qt.fidelity(rho, rho_rec)
    print(randomization, td1, td2, td4)

    data = np.array([[td1, td2, td4],
                     [if1, if2, if4]])
    np.save(path, data)


process_tomograph(num_samples_cov, randomization)
