from main_tomography import *

Nt = 20
num_modes = 1
K = 10000  # number of samples in each angle
d = 11  # number of sampled angles
num_samples = K*d  #number of total samples
epsilon = 0.1
randomization = 0


#state = (qt.coherent(Nt, 0.2) + qt.coherent(Nt, -0.2)).unit()
#state = qt.coherent(Nt, 0.5)
state = (qt.basis(Nt,0)+ qt.basis(Nt,1)).unit()
#state = qt.basis(Nt,1)
rho = qt.ket2dm(state)

#RrhoR_state_tomograph_onemode(state, Nt, epsilon, d,K, 20, save=False, savepath=None)
#RrhoR_state_tomograph_onemode(state, Nt, epsilon, d,K, 100)



x, p = position_momentum_operators_multimode(Nt, 1, 0)
f_list, p_list, state_list = RrhoR_measurement_generator_onemode(state, Nt, d, K)
print(len(f_list))


rho0 = qt.qeye(Nt) / Nt
rho0 = qt.tensor(*[rho0] * num_modes)

Infidelity = 1 - qt.fidelity(rho0, rho)
tracedistance = qt.metrics.tracedist(rho0, rho)

rho_list = [rho0.full()]
if_list = [Infidelity]
td_list = [tracedistance]
pd_list = [pairwise_difference(rho0,rho)]




def RrhoR_state_tomograph_iteration():

    psi = qt.Qobj(rho, dims=[[Nt], [Nt]])
    rho_i = qt.Qobj(rho_list[-1], dims=[[Nt], [Nt]])
    R_rho_i = qt.Qobj(np.zeros((rho_i.shape[0], rho_i.shape[1])), dims=rho_i.dims)
    for j in range(len(state_list)):
        state = qt.Qobj(state_list[j], dims=[[Nt], [1]])
        pro_j = state * state.dag()
        R_rho_i = R_rho_i + pro_j * f_list[j]*K / ((rho_i * pro_j).tr()* num_samples)

    rho_updated = RrhoR(R_rho_i, rho_i, epsilon=epsilon)
    Infidelity = 1 - qt.fidelity(rho_updated, psi)
    tracedistance = qt.metrics.tracedist(rho_updated, psi)

    td_list.append(tracedistance)
    if_list.append(Infidelity)
    rho_list.append(rho_updated.full())
    pd_list.append(pairwise_difference(rho_updated, psi))
    if len(if_list) % 100 == 0:
        print(len(if_list),
              if_list[-1], np.log10(if_list[-1]),
              td_list[-1], np.log10(td_list[-1]),
              pd_list[-1], np.log10(pd_list[-1]))



for i in range(1000):
    RrhoR_state_tomograph_iteration()


