import numpy as np

# ---- (A) Use the correct Ω for the interleaved ordering (x1,p1,x2,p2)
J2 = np.array([[0, 1], [-1, 0]], dtype=float)
Omega_int = np.block([[J2, np.zeros_like(J2)],
                      [np.zeros_like(J2), J2]])


def sympl_defect_interleaved(S):
    return np.linalg.norm(S @ Omega_int @ S.T - Omega_int, ord='fro')


# ---- (B) From interleaved S, recover the complex mode-mixing K
# r_int = (x1,p1,x2,p2). Partition S into 2x2 blocks w.r.t. modes.
# A maps (x1,p1)->(x1',p1'), B maps (x2,p2)->(x1',p1') etc.
def K_from_S_interleaved(S):
    A = S[0:2, 0:2]  # (x1,p1) -> (x1',p1')
    B = S[0:2, 2:4]  # (x2,p2) -> (x1',p1')
    C = S[2:4, 0:2]  # (x1,p1) -> (x2',p2')
    D = S[2:4, 2:4]  # (x2,p2) -> (x2',p2')

    def R_to_C(M):  # 2x2 real block -> complex scalar mapping for a' = u a + v a† (here v=0 for passive)
        # For passive, x' = Re(u) x - Im(u) p, p' = Im(u) x + Re(u) p
        # i.e. M = [[Re(u), -Im(u)], [Im(u), Re(u)]], so u = M00 + i M10
        return (M[0, 0] + 1j * M[1, 0])  # = Re(u) + i Im(u)

    u11 = R_to_C(A)
    u12 = R_to_C(B)
    u21 = R_to_C(C)
    u22 = R_to_C(D)
    return np.array([[u11, u12], [u21, u22]], dtype=complex)


# ---- (C) A beamsplitter unitary that matches a' = K a with K defined below
def U_beamsplitter_numpy(Nt, theta, xi):
    # Hermitian BS generator:
    # H_BS = i ( e^{iξ} a1^† a2 - e^{-iξ} a1 a2^† )
    a = np.zeros((Nt, Nt), dtype=complex)
    for n in range(1, Nt): a[n - 1, n] = np.sqrt(n)
    I = np.eye(Nt, dtype=complex)
    a1 = np.kron(a, I);
    a2 = np.kron(I, a)
    adag1 = a1.conj().T;
    adag2 = a2.conj().T

    H = 1j * (np.exp(1j * xi) * adag1 @ a2 - np.exp(-1j * xi) * a1 @ adag2)  # Hermitian
    # U = exp(-i θ H)
    w, V = np.linalg.eig(H)
    U = (V * np.exp(1j * theta * w)) @ np.linalg.inv(V)
    return U


# ---- (D) Single-mode phase rotations; matches a' = e^{iφ} a
def U_phase_numpy(Nt, phi):
    # U = exp(-i φ n); Heisenberg: U† a U = e^{+iφ} a
    diag = np.exp(-1j * phi * np.arange(Nt))
    return np.diag(diag)


# ---- (E) Build full passive U from K = R_out * B * R_in
def build_U_passive_numpy(Nt, theta, xi, a1_out, a2_out, b1_in, b2_in):
    U_B = U_beamsplitter_numpy(Nt, theta, xi)
    U_R1o = U_phase_numpy(Nt, a1_out);
    U_R2o = U_phase_numpy(Nt, a2_out)
    U_R1i = U_phase_numpy(Nt, b1_in);
    U_R2i = U_phase_numpy(Nt, b2_in)
    U_Rout = np.kron(U_R1o, U_R2o)
    U_Rin = np.kron(U_R1i, U_R2i)
    return U_Rout @ U_B @ U_Rin


# ---- (F) Analytic S from K in interleaved ordering (same as before)
def S_from_K_interleaved(K):
    Re, Im = np.real(K), np.imag(K)
    Sg = np.block([[Re, -Im],
                   [Im, Re]])  # grouped (x1,x2,p1,p2)
    P = np.array([[1, 0, 0, 0],
                  [0, 0, 1, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1]], dtype=float)  # r_group = P r_int
    return P.T @ Sg @ P


# ---- (G) Your existing S-from-U projection (unchanged)
def two_mode_quadratures(Nt):
    a = np.zeros((Nt, Nt), dtype=complex);
    for n in range(1, Nt): a[n - 1, n] = np.sqrt(n)
    I = np.eye(Nt, dtype=complex)
    x = (a + a.conj().T) / np.sqrt(2.0)
    p = (a - a.conj().T) / (1j * np.sqrt(2.0))
    return [np.kron(x, I), np.kron(p, I), np.kron(I, x), np.kron(I, p)]


def gaussian_S_from_gate(U, Nt):
    R = two_mode_quadratures(Nt)
    Udag = U.conj().T
    G = np.array([[np.trace(Rj @ Rk) for Rk in R] for Rj in R], dtype=complex)
    S = np.zeros((4, 4), dtype=float)
    for i in range(4):
        Ri = Udag @ R[i] @ U
        b = np.array([np.trace(Rk @ Ri) for Rk in R], dtype=complex)
        c, *_ = np.linalg.lstsq(G, b, rcond=None)
        S[i, :] = np.real_if_close(c, tol=1e-10)
    return S


def two_mode_block_10_01(U, Nt):
    # indices of |n1,n2>
    def idx(n1, n2): return n1 * Nt + n2

    i10 = idx(1, 0);
    i01 = idx(0, 1)
    block = U[[i10, i01]][:, [i10, i01]]  # 2x2 block
    # strip global phase to compare to B(θ,ξ)
    phase = np.linalg.det(block) ** (1 / 2)  # any unit-modulus scalar works
    return block / phase


def B_target(theta, xi):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -np.exp(1j * xi) * s],
                     [np.exp(-1j * xi) * s, c]], dtype=complex)


# ---- (H) End-to-end check you can run
if __name__ == "__main__":
    Nt = 20
    theta = 0.37
    xi = 0.51
    a1_out, a2_out = 0.2, -0.35
    b1_in, b2_in = -0.4, 0.75

    # Intended K and analytic S
    K_target = np.diag([np.exp(1j * a1_out), np.exp(1j * a2_out)]) @ \
               np.array([[np.cos(theta), -np.exp(1j * xi) * np.sin(theta)],
                         [np.exp(-1j * xi) * np.sin(theta), np.cos(theta)]]) @ \
               np.diag([np.exp(1j * b1_in), np.exp(1j * b2_in)])
    S_ana = S_from_K_interleaved(K_target)

    # Build U and recover S_num, then K_num
    U = build_U_passive_numpy(Nt, theta, xi, a1_out, a2_out, b1_in, b2_in)
    B_num = two_mode_block_10_01(U, Nt)
    Bt = B_target(theta, xi)
    print("||B_num - Bt||_F =", np.linalg.norm(B_num - Bt))

    S_num = gaussian_S_from_gate(U, Nt)
    K_num = K_from_S_interleaved(S_num)

    # Errors
    errS_inf = np.max(np.abs(S_num - S_ana))
    errS_fro = np.linalg.norm(S_num - S_ana, ord='fro')
    errK = np.linalg.norm(K_num - K_target, ord='fro')

    print("Analytic S:\n", np.round(S_ana, 6))
    print("\nNumeric  S:\n", np.round(S_num, 6))
    print("\n||S_num - S_ana||_∞ =", f"{errS_inf:.3e}", "  ||·||_F =", f"{errS_fro:.3e}")
    print("\nRecovered K_num vs intended K_target (Fro):", f"{errK:.3e}")

    print("\nSymplectic defect (analytic):", f"{sympl_defect_interleaved(S_ana):.3e}")
    print("Symplectic defect (numeric): ", f"{sympl_defect_interleaved(S_num):.3e}")
