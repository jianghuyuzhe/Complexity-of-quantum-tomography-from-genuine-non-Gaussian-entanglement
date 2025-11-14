import numpy as np
import matplotlib.pyplot as plt

Nt = 20
d = 11
import numpy as np

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


ub = 1e6
lb = 1e2
def lower_envelope(x, y, return_indices=True):
    """
    Compute the lower envelope of 2D points.

    Parameters
    ----------
    x, y : array-like, shape (n,)
        Coordinates of points.
    return_indices : bool
        If True, also return indices into the original arrays for the envelope points.

    Returns
    -------
    x_env, y_env : 1D numpy arrays
        The breakpoints of the monotone nonincreasing lower envelope, ordered by x.
    idx_env : 1D numpy array (optional)
        Indices into the original (x, y) arrays for each envelope point.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    assert x.shape == y.shape

    # 1) Sort by x (and y for stability)
    order = np.lexsort((y, x))
    xs, ys = x[order], y[order]

    # 2) For each unique x, keep its minimum y (the bottom of each vertical column)
    ux, start_idx, counts = np.unique(xs, return_index=True, return_counts=True)
    # reduceat to get per-group minima
    ymins = np.minimum.reduceat(ys, start_idx)

    # 3) Make the envelope monotone by taking a cumulative minimum left→right
    ycum = np.minimum.accumulate(ymins)

    # 4) Keep only the breakpoints where the envelope actually drops
    keep = np.r_[True, ycum[1:] < ycum[:-1]]
    x_env = ux[keep]
    y_env = ycum[keep]

    if not return_indices:
        return x_env, y_env

    # 5) Map envelope points back to indices in the original data
    # For each kept x, pick the first occurrence with y == group-min at that x
    idx_env = []
    for xv, yv in zip(x_env, y_env):
        # among original points with this x, find the one with minimal y that matches the envelope
        candidates = np.where(x == xv)[0]
        # minimal y at this x in original data
        y_min_at_x = np.min(y[candidates])
        # pick first match (stable)
        idx_env.append(candidates[np.where(y[candidates] == y_min_at_x)[0][0]])
    idx_env = np.asarray(idx_env)

    return x_env, y_env, idx_env


def plot_number_GDE(ax):
    num_sample_cov_list = [i * 10 ** j for j in range(5) for i in [1, 4, 7]]
    K_list = [i * 10 ** j for j in range(5) for i in [1, 4, 7]]
    randomization_list = np.arange(20)

    error_matrix = np.ones((len(num_sample_cov_list), len(K_list), len(randomization_list)))
    numsamples_matrix = np.zeros((len(num_sample_cov_list), len(K_list)))
    for i, num_sample_cov in enumerate(num_sample_cov_list):
        for j, K in enumerate(K_list):
            numsamples_matrix[i, j] = num_sample_cov + K * d
            for r in randomization_list:
                try:
                    data = np.load(
                        f"Data_HPC/01/two_mode_numberstate_numsamplescov={num_sample_cov}_K={K}_d={d}_randomization={r}_PT.npy")
                    error_matrix[i, j, r] = data[0, 2]
                except:
                    print(" number_GDE fail", num_sample_cov, K, r)

    error_matrix = np.mean(error_matrix, axis=2)
    #print(numsamples_matrix.reshape(1, -1))
    #print(error_matrix.reshape(1, -1))

    # plt.scatter(numsamples_matrix.flatten(),error_matrix.flatten())
    x, y, _ = lower_envelope(numsamples_matrix.flatten(), error_matrix.flatten())

    mask = (np.array(x) <= ub) & (lb <= np.array(x))
    slope, intercept = np.polyfit(np.log10(np.array(x)[mask]), np.log10(np.array(y)[mask]),1)
    ax.scatter(x, y, label="GDE "+ r"$k$="+f"{np.around(slope,2)}", color="red", s=20)
    ax.plot(x, (10 ** intercept) * np.array(x) ** slope, color="red", ls="--")  # + r"$k_2$="+f"{np.around(slope2,2)}"


def plot_number_RrhoR(ax):
    K_list = [i * 10 ** j for j in range(5) for i in [1, 4, 7]]
    randomization_list = np.arange(20)
    error_matrix = np.ones((len(K_list), len(randomization_list)))
    for i, K in enumerate(K_list):
        for r in randomization_list:
            try:
                data = np.load(f"Data_HPC/02/two_mode_numberstate_RrhoR_K={K}_d=11_randomization={r}_td_list.npy")
                error_matrix[i, r] = np.min(data)
            except:
                error_matrix[i, r] = np.nan
                print(" number RrhoR fail", K, r)

    numsamples_list = np.array(K_list) * (d ** 2)
    error_list = np.nanmean(error_matrix, 1)
    # ax.scatter(numsamples_list, error_list, label="RrhoR", color="black", marker="s",s=20)

    x = numsamples_list
    y = error_list
    mask = (np.array(x) <= ub) & (lb <= np.array(x))
    slope, intercept = np.polyfit(np.log10(np.array(x)[mask]), np.log10(np.array(y)[mask]), 1)
    ax.scatter(numsamples_list, error_list, label=r"R$\rho$R " + r"$k$="+f"{np.around(slope,2)}", color="black", marker="s", s=20)
    ax.plot(x, (10 ** intercept) * np.array(x) ** slope, color="black", ls="--")


def plot_numberstate_GDE_noise(ax):
    num_sample_cov_list = [i * 10 ** j for j in range(5) for i in [1, 4, 7]]
    K_list = [i * 10 ** j for j in range(5) for i in [1, 4, 7]]
    randomization_list = np.arange(20)

    error_matrix = np.ones((len(num_sample_cov_list), len(K_list), len(randomization_list)))
    numsamples_matrix = np.zeros((len(num_sample_cov_list), len(K_list)))
    for i, num_sample_cov in enumerate(num_sample_cov_list):
        for j, K in enumerate(K_list):
            numsamples_matrix[i, j] = num_sample_cov + K * d
            for r in randomization_list:
                try:
                    data = np.load(
                        f"Data_HPC/09/two_mode_numberstate_noisy_eta=0.9_numsamplescov={num_sample_cov}_K={K}_d={d}_randomization={r}_PT.npy")
                    error_matrix[i, j, r] = data[0, 2]
                except:
                    print("number state GDE noise fail", num_sample_cov, K, r)

    error_matrix = np.mean(error_matrix, axis=2)
    # print(numsamples_matrix.reshape(1, -1))
    # print(error_matrix.reshape(1, -1))

    # plt.scatter(numsamples_matrix.flatten(),error_matrix.flatten())
    x, y, _ = lower_envelope(numsamples_matrix.flatten(), error_matrix.flatten())


    mask = (np.array(x) <= ub) & (lb <= np.array(x))
    slope, intercept = np.polyfit(np.log10(np.array(x)[mask]), np.log10(np.array(y)[mask]), 1)
    ax.scatter(x, y, label="GDE-lossy " + r"$k$=" + f"{np.around(slope, 2)}", color="pink", s=20, marker="^")
    ax.plot(x, (10 ** intercept) * np.array(x) ** slope, color="pink", ls="--")

def plot_numberstate_RrhoR_noise(ax):
    K_list = [i * 10 ** j for j in range(5) for i in [1, 4, 7]]
    randomization_list = np.arange(20)
    error_matrix = np.ones((len(K_list), len(randomization_list)))
    for i, K in enumerate(K_list):
        for r in randomization_list:
            try:
                data = np.load(
                    f"Data_HPC/10/two_mode_numberstate_RrhoR_noisy_eta=0.9_K={K}_d=11_randomization={r}_td_list.npy")
                error_matrix[i, r] = np.min(data)
            except:
                error_matrix[i, r] = np.nan
                print("number-RrhoR-noisy fail", K, r)

    numsamples_list = np.array(K_list) * (d ** 2)
    error_list = np.nanmean(error_matrix, 1)


    x = numsamples_list
    y = error_list
    mask = (np.array(x) <= ub) & (lb <= np.array(x))
    slope, intercept = np.polyfit(np.log10(np.array(x)[mask]), np.log10(np.array(y)[mask]), 1)
    ax.scatter(numsamples_list, error_list, label=r"R$\rho$R"+"-lossy " + r"$k$=" + f"{np.around(slope, 2)}",
               color="grey", s=20, marker="*")
    ax.plot(x, (10 ** intercept) * np.array(x) ** slope, color="grey", ls="--")

def plot_catstate_GDE(ax):
    num_sample_cov_list = [i * 10 ** j for j in range(5) for i in [1, 4, 7]]
    K_list = [i * 10 ** j for j in range(5) for i in [1, 4, 7]]
    randomization_list = np.arange(20)

    error_matrix = np.ones((len(num_sample_cov_list), len(K_list), len(randomization_list)))
    numsamples_matrix = np.zeros((len(num_sample_cov_list), len(K_list)))
    for i, num_sample_cov in enumerate(num_sample_cov_list):
        for j, K in enumerate(K_list):
            numsamples_matrix[i, j] = num_sample_cov + K * d
            for r in randomization_list:
                try:
                    data = np.load(
                        f"Data_HPC/03/two_mode_catstate_numsamplescov={num_sample_cov}_K={K}_d={d}_randomization=13_PT.npy")
                    error_matrix[i, j, r] = data[0, 2]
                except:
                    print("fail", num_sample_cov, K, r)

    error_matrix = np.mean(error_matrix, axis=2)
    # print(numsamples_matrix.reshape(1, -1))
    # print(error_matrix.reshape(1, -1))

    # plt.scatter(numsamples_matrix.flatten(),error_matrix.flatten())
    x, y, _ = lower_envelope(numsamples_matrix.flatten(), error_matrix.flatten())


    mask = (np.array(x) <= ub) & (lb <= np.array(x))
    slope, intercept = np.polyfit(np.log10(np.array(x)[mask]), np.log10(np.array(y)[mask]), 1)
    ax.scatter(x, y, label="GDE " + r"$k$=" + f"{np.around(slope, 2)}", color="red", s=20)
    ax.plot(x, (10 ** intercept) * np.array(x) ** slope, color="red", ls="--")


def plot_catstate_RrhoR(ax):
    K_list = [i * 10 ** j for j in range(5) for i in [1, 4, 7]]
    randomization_list = np.arange(20)
    error_matrix = np.ones((len(K_list), len(randomization_list)))
    for i, K in enumerate(K_list):
        for r in randomization_list:
            try:
                data = np.load(f"Data_HPC/04/two_mode_catstate_RrhoR_K={K}_d=11_randomization={r}_td_list.npy")
                error_matrix[i, r] = np.min(data)
            except:
                error_matrix[i, r] = np.nan
                print("fail", K, r)

    numsamples_list = np.array(K_list) * (d ** 2)
    error_list = np.nanmean(error_matrix, 1)


    x = numsamples_list
    y = error_list
    mask = (np.array(x) <= ub) & (lb <= np.array(x))
    slope, intercept = np.polyfit(np.log10(np.array(x)[mask]), np.log10(np.array(y)[mask]), 1)
    ax.scatter(numsamples_list, error_list, label=r"R$\rho$R " + r"$k$=" + f"{np.around(slope, 2)}", color="black", s=20, marker="s")
    ax.plot(x, (10 ** intercept) * np.array(x) ** slope, color="black", ls="--")




def plot_catstate_GDE_noise(ax):
    num_sample_cov_list = [i * 10 ** j for j in range(5) for i in [1, 4, 7]]
    K_list = [i * 10 ** j for j in range(5) for i in [1, 4, 7]]
    randomization_list = np.arange(20)

    error_matrix = np.ones((len(num_sample_cov_list), len(K_list), len(randomization_list)))
    numsamples_matrix = np.zeros((len(num_sample_cov_list), len(K_list)))
    for i, num_sample_cov in enumerate(num_sample_cov_list):
        for j, K in enumerate(K_list):
            numsamples_matrix[i, j] = num_sample_cov + K * d
            for r in randomization_list:
                try:
                    data = np.load(
                        f"Data_HPC/05/two_mode_numberstate_noisy_eta=0.9_numsamplescov={num_sample_cov}_K={K}_d={d}_randomization={r}_PT.npy")
                    error_matrix[i, j, r] = data[0, 2]
                except:
                    print("catstate_GDE_noise fail", num_sample_cov, K, r)

    error_matrix = np.mean(error_matrix, axis=2)
    # print(numsamples_matrix.reshape(1, -1))
    # print(error_matrix.reshape(1, -1))

    # plt.scatter(numsamples_matrix.flatten(),error_matrix.flatten())
    x, y, _ = lower_envelope(numsamples_matrix.flatten(), error_matrix.flatten())


    mask = (np.array(x) <= ub) & (lb <= np.array(x))
    slope, intercept = np.polyfit(np.log10(np.array(x)[mask]), np.log10(np.array(y)[mask]), 1)
    ax.scatter(x, y, label="GDE-lossy " + r"$k$=" + f"{np.around(slope, 2)}", color="pink", s=20, marker="^")
    ax.plot(x, (10 ** intercept) * np.array(x) ** slope, color="pink", ls="--")


def plot_catstate_RrhoR_noise(ax):
    K_list = [i * 10 ** j for j in range(5) for i in [1, 4, 7]]
    randomization_list = np.arange(20)
    error_matrix = np.ones((len(K_list), len(randomization_list)))
    for i, K in enumerate(K_list):
        for r in randomization_list:
            try:
                data = np.load(
                    f"Data_HPC/06/two_mode_catstate_RrhoR_noisy_eta=0.9_K={K}_d=11_randomization={r}_td_list.npy")
                error_matrix[i, r] = np.min(data)
            except:
                error_matrix[i, r] = np.nan
                print("cat-RrhoR-noisy fail", K, r)

    numsamples_list = np.array(K_list) * (d ** 2)
    error_list = np.nanmean(error_matrix, 1)


    x = numsamples_list
    y = error_list
    mask = (np.array(x) <= ub) & (lb <= np.array(x))
    slope, intercept = np.polyfit(np.log10(np.array(x)[mask]), np.log10(np.array(y)[mask]), 1)
    ax.scatter(numsamples_list, error_list, label=r"R$\rho$R"+"-lossy " + r"$k$=" + f"{np.around(slope, 2)}", color="grey", s=20, marker="*")
    ax.plot(x, (10 ** intercept) * np.array(x) ** slope, color="grey", ls="--")


def plot_thermalstate_GDE(ax):
    num_sample_cov_list = [i * 10 ** j for j in range(5) for i in [1, 4, 7]]
    K_list = [i * 10 ** j for j in range(5) for i in [1, 4, 7]]
    randomization_list = np.arange(20)

    error_matrix = np.ones((len(num_sample_cov_list), len(K_list), len(randomization_list)))
    numsamples_matrix = np.zeros((len(num_sample_cov_list), len(K_list)))
    for i, num_sample_cov in enumerate(num_sample_cov_list):
        for j, K in enumerate(K_list):
            numsamples_matrix[i, j] = num_sample_cov + K * d
            for r in randomization_list:
                try:
                    data = np.load(
                        f"Data_HPC/07/two_mode_thermalstate_numsamplescov={num_sample_cov}_K={K}_d={d}_randomization={r}_PT.npy")
                    error_matrix[i, j, r] = data[0, 2]
                except:
                    print("thermal state GDE fail", num_sample_cov, K, r)

    error_matrix = np.mean(error_matrix, axis=2)
    # print(numsamples_matrix.reshape(1, -1))
    # print(error_matrix.reshape(1, -1))

    # plt.scatter(numsamples_matrix.flatten(),error_matrix.flatten())
    x, y, _ = lower_envelope(numsamples_matrix.flatten(), error_matrix.flatten())


    mask = (np.array(x) <= ub) & (lb <= np.array(x))
    slope, intercept = np.polyfit(np.log10(np.array(x)[mask]), np.log10(np.array(y)[mask]), 1)
    ax.scatter(x, y, label="GDE " + r"$k$=" + f"{np.around(slope, 2)}", color="red", s=20, )
    ax.plot(x, (10 ** intercept) * np.array(x) ** slope, color="red", ls="--")




def plot_thermalstate_RrhoR(ax):
    K_list = [i * 10 ** j for j in range(5) for i in [1, 4, 7]]
    randomization_list = np.arange(20)
    error_matrix = np.ones((len(K_list), len(randomization_list)))
    for i, K in enumerate(K_list):
        for r in randomization_list:
            try:
                data = np.load(f"Data_HPC/08/two_mode_thermalstate_RrhoR_K={K}_d=11_randomization={r}_td_list.npy")
                error_matrix[i, r] = np.min(data)
            except:
                error_matrix[i, r] = np.nan
                print("fail", K, r)

    numsamples_list = np.array(K_list) * (d ** 2)
    error_list = np.nanmean(error_matrix, 1)


    x = numsamples_list
    y = error_list
    mask = (np.array(x) <= ub) & (lb <= np.array(x))
    slope, intercept = np.polyfit(np.log10(np.array(x)[mask]), np.log10(np.array(y)[mask]), 1)
    ax.scatter(numsamples_list, error_list, label=r"R$\rho$R " + r"$k$=" + f"{np.around(slope, 2)}", color="black", marker="s", s=20)
    ax.plot(x, (10 ** intercept) * np.array(x) ** slope, color="black", ls="--")


fig, axes = plt.subplots(1, 3, sharey=True, figsize=(12, 4))
plot_number_GDE(axes[0])
plot_number_RrhoR(axes[0])
plot_numberstate_GDE_noise(axes[0])
plot_numberstate_RrhoR_noise(axes[0])
plot_catstate_GDE(axes[1])
plot_catstate_RrhoR(axes[1])
plot_catstate_GDE_noise(axes[1])
plot_catstate_RrhoR_noise(axes[1])
plot_thermalstate_GDE(axes[2])
plot_thermalstate_RrhoR(axes[2])

for i in range(3):
    axes[i].set_yscale("log")
    axes[i].set_xscale("log")
    axes[i].set_xlim(100, 1e6)
    axes[i].set_xlabel("Number of Copies " + r"$M$")
    axes[i].legend(loc = 'lower left')

axes[0].set_ylabel("Error")
axes[0].set_title("number state")
axes[1].set_title("cat state")
axes[2].set_title("thermal state")
"""
plt.yscale("log")
plt.xscale("log")
plt.xlim(10, 1e6)
plt.xlabel("Number of Copies M")
plt.ylabel("Error")
"""
#plt.legend()
plt.tight_layout()
plt.savefig("Figs/2mode_GDE_vs_RrhoR_tomography.pdf", dpi=500)
plt.show()
