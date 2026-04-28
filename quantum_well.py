
import numpy as np
from scipy.linalg import eigh
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

Nx, Ny   = 20, 20
x0, xL   = -1.0,  1.0
y0, yL   = -0.5,  0.5
kappa    = 2326.0    # Coulomb strength (Mott-Hubbard crossover, U/t ~ 1)
epsilon  = 0.01 
Nstates  = 40   
params   = {'a': 0.6, 'k': 1000.0, 'ky': 1500.0, 'delta': 0.0, 'd': 1.0}
N_CI_COMPUTE  = 200   # raised from 40; covers L0-L4 x R0-R4
SMOOTH_PL = False
M_LOC     =32     # localisation subspace size (try 8, 12, 16, ...)
X_CUT     = 0.0       # dividing surface between wells (barrier centre)
SIGMA_PL  = 0.03      



def sine_dvr_1d(x0, xL, N):
    """1-D sine-DVR on [x0, xL] with N grid points.  hbar^2/2m = 1."""
    L = xL - x0
    x = x0 + np.arange(1, N + 1) * L / (N + 1)
    w = np.full(N, L / (N + 1))
    j = np.arange(1, N + 1)
    U = np.sqrt(2 / (N + 1)) * np.sin(np.outer(j, j * np.pi / (N + 1)))
    T = (U.T * (j * np.pi / L) ** 2) @ U
    return x, w, T


def build_2d_dvr(Nx=20, Ny=20, x0=-1.0, xL=1.0, y0=-0.5, yL=0.5):
    xg, wx, Tx = sine_dvr_1d(x0, xL, Nx)
    yg, wy, Ty = sine_dvr_1d(y0, yL, Ny)
    T2D = np.kron(Tx, np.eye(Ny)) + np.kron(np.eye(Nx), Ty)
    return xg, yg, wx, wy, T2D

x_grid, y_grid, w_x, w_y, T2D = build_2d_dvr(Nx, Ny, x0, xL, y0, yL)

def double_well_potential(x, y, params):
    a     = params.get('a',     1.0)
    k     = params.get('k',    50.0)
    ky    = params.get('ky',  100.0)
    delta = params.get('delta', 0.5)
    d     = params.get('d',     1.0)
    return k * ((x / d)**2 - a**2)**2 + delta * x + 0.5 * ky * y**2


def build_potential_matrix(x_grid, y_grid, params):
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    V = double_well_potential(X, Y, params).ravel()
    return np.diag(V)

V2D = build_potential_matrix(x_grid, y_grid, params)


def single_particle_eigenstates(T2D, V2D, Nstates=40):
    e_vals, e_vecs = eigh(T2D + V2D)
    return e_vals[:Nstates], e_vecs[:, :Nstates]

single_energies, single_vecs = single_particle_eigenstates(T2D, V2D, Nstates)

def build_left_projector_mask(x_grid, y_grid, x_cut=0.0, smooth=False, sigma=0.02):
    """
    Return P_L(x,y) on the DVR grid as a flattened array in [0,1].

    If smooth=False: hard step at x_cut.
    If smooth=True : smooth Fermi-type step of width sigma.
    """
    X, _ = np.meshgrid(x_grid, y_grid, indexing='ij')
    if not smooth:
        PL = (X < x_cut).astype(float)
    else:
        PL = 1.0 / (1.0 + np.exp((X - x_cut) / sigma))
    return PL.ravel()

def localise_orbitals_projector_DVR(single_vecs, x_grid, y_grid,
                                    M=16, x_cut=0.0, smooth=True, sigma=0.03):
    PLg = build_left_projector_mask(x_grid, y_grid, x_cut=x_cut,
                                    smooth=smooth, sigma=sigma)   # (G,)
    V = np.asarray(single_vecs[:, :M], dtype=complex)             # (G,M)

    # Psub_ij = <psi_i | PL | psi_j> in the DVR inner product
    Psub = V.conj().T @ (PLg[:, None] * V)

    lam, U = eigh((Psub + Psub.conj().T) / 2.0)
    idx = np.argsort(lam)[::-1]
    lam, U = lam[idx], U[:, idx]

    vecs_loc = V @ U
    labels = np.array(['L' if l > 0.5 else 'R' for l in lam], dtype=object)

    # sanity: orthonormality in DVR inner product
    S = vecs_loc.conj().T @ vecs_loc
    return U, vecs_loc, lam, labels, S

U_loc, vecs_loc, lam_loc, lr_labels, S_ortho = localise_orbitals_projector_DVR(
    single_vecs, x_grid, y_grid,
    M=M_LOC, x_cut=X_CUT, smooth=SMOOTH_PL, sigma=SIGMA_PL
)

def plot_localised_orbitals(vecs_loc, Nx, Ny,
                            x0, xL, y0, yL,
                            lam=None, labels=None,
                            idx_list=None, n_show=6,
                            cmap='hot', savepath=None):

    M = vecs_loc.shape[1]

    if idx_list is None:
        idx_list = list(range(min(n_show, M)))
    else:
        idx_list = list(idx_list)

    n_show = len(idx_list)
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 8))
    axes = axes.ravel()

    for j, ax in enumerate(axes):
        if j >= n_show:
            ax.axis('off')
            continue

        mu = idx_list[j]
        Z = (vecs_loc[:, mu].reshape(Nx, Ny))**2
        im = ax.imshow(Z.T, origin='lower', extent=(x0, xL, y0, yL),
                       aspect='auto', cmap=cmap)

        title = f'phi_{mu}'
        if lam is not None:
            title += f'  lam={float(lam[mu]):.3f}'
        if labels is not None:
            title += f'  ({labels[mu]})'
        ax.set_title(title)
        ax.set_xlabel('x'); ax.set_ylabel('y')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=120, bbox_inches='tight')
    plt.show()
    return 

def build_slater_basis(Nb):
    """Original index-ordered basis (kept for reference/backward compatibility)."""
    basis = []
    for a in range(Nb):
        for b in range(a, Nb):
            if a == b:
                basis.append((a, b, 'ud', 'singlet'))
            else:
                basis.append((a, b, 'singlet',   'singlet'))
                basis.append((a, b, 'uu',         'triplet_p'))
                basis.append((a, b, 'triplet_0',  'triplet_0'))
                basis.append((a, b, 'dd',         'triplet_m'))
    return basis


def build_slater_basis_sorted(Nb, single_energies):
    """
    Energy-sorted Slater basis.

    Sorts all (a,b) orbital pairs by E_a + E_b before returning, so that
    truncating to n_compute determinants captures all configurations in
    the lowest energy window -- including L-excited (1,1) states such as
    (L1,R0) that the index-ordered basis placed at index >= 158.
    """
    raw = []
    for a in range(Nb):
        for b in range(a, Nb):
            Eab = float(single_energies[a]) + float(single_energies[b])
            if a == b:
                raw.append((Eab, a, b, 'ud',        'singlet'))
            else:
                raw.append((Eab, a, b, 'singlet',   'singlet'))
                raw.append((Eab, a, b, 'uu',        'triplet_p'))
                raw.append((Eab, a, b, 'triplet_0', 'triplet_0'))
                raw.append((Eab, a, b, 'dd',        'triplet_m'))
    raw.sort(key=lambda x: (x[0], x[1], x[2]))
    return [(a, b, sc, st) for (_, a, b, sc, st) in raw]

old_basis    = build_slater_basis(Nstates)
sorted_basis = build_slater_basis_sorted(Nstates, single_energies)

for i, (a, b, sc, st) in enumerate(sorted_basis[:10]):
    Eab = single_energies[a] + single_energies[b]
    print(f"  [{i:3d}]  ({a:2d},{b:2d})  {st:<12}  E_a+E_b = {Eab:.2f}")

def orbital_localization(single_vecs, single_energies, x_grid, y_grid,
                          Nstates=40, loc_threshold=0.20):
    Nx, Ny = len(x_grid), len(y_grid)
    X2d, _ = np.meshgrid(x_grid, y_grid, indexing='ij')
    Xf     = X2d.ravel()
    prob   = single_vecs[:, :Nstates]**2
    ex_all = Xf @ prob
    info   = []
    for idx in range(Nstates):
        ex = float(ex_all[idx])
        if   ex < -loc_threshold: label, wc = "Left well",   'L'
        elif ex >  loc_threshold: label, wc = "Right well",  'R'
        else:                     label, wc = "Delocalized", 'D'
        info.append((idx, float(single_energies[idx]), ex, label, wc))
    return info


orbital_info = orbital_localization(single_vecs, single_energies,
                                     x_grid, y_grid, Nstates)
well   = {o[0]: o[4] for o in orbital_info}
L_orbs = sorted([o[0] for o in orbital_info if o[4]=='L'], key=lambda i: single_energies[i])
R_orbs = sorted([o[0] for o in orbital_info if o[4]=='R'], key=lambda i: single_energies[i])
rank_L = {orb: k for k, orb in enumerate(L_orbs)}
rank_R = {orb: k for k, orb in enumerate(R_orbs)}


def precompute_coulomb_kernel(x_grid, y_grid, wx, wy, kappa=1.0, epsilon=0.1):
    X, Y    = np.meshgrid(x_grid, y_grid, indexing='ij')
    Xf, Yf  = X.ravel(), Y.ravel()
    Wx, Wy  = np.meshgrid(wx, wy, indexing='ij')
    Wf      = (Wx * Wy).ravel()
    dx = Xf[:, None] - Xf[None, :]
    dy = Yf[:, None] - Yf[None, :]
    r12 = np.sqrt(dx**2 + dy**2 + epsilon**2)
    return kappa / r12 * Wf[:, None] * Wf[None, :]

K = precompute_coulomb_kernel(x_grid, y_grid, w_x, w_y, kappa, epsilon)

def build_ci_hamiltonian(slater_basis, single_energies, sv, K, n_compute=40):
    n  = min(n_compute, len(slater_basis))
    H  = np.zeros((n, n), dtype=complex)
    _te_cache = {}
    def te(a, b, c, d):
        key = (a, b, c, d)
        if key not in _te_cache:
            left  = np.conj(sv[:, a]) * sv[:, c]
            right = np.conj(sv[:, b]) * sv[:, d]
            _te_cache[key] = left @ K @ right
        return _te_cache[key]

    for I in range(n):
        a_I, b_I, _, stype_I = slater_basis[I]
        same_I = (a_I == b_I)
        for J in range(I, n):
            a_J, b_J, _, stype_J = slater_basis[J]
            if stype_I != stype_J:
                continue
            same_J = (a_J == b_J)
            val = 0.0 + 0.0j
            if I == J:
                val += single_energies[a_I] + single_energies[b_I]
            if stype_I == 'singlet':
                if same_I and same_J:
                    if a_I == a_J:
                        val += te(a_I, a_I, a_I, a_I)
                    else:
                        val += 2.0 * te(a_I, a_J, a_I, a_J)
                elif same_I:
                    val += np.sqrt(2.0) * te(a_I, a_J, a_I, b_J)
                elif same_J:
                    val += np.sqrt(2.0) * te(a_I, a_J, b_I, a_J)
                else:
                    val += te(a_I, a_J, b_I, b_J) + te(a_I, b_J, b_I, a_J)
            else:
                if not same_I and not same_J:
                    val += te(a_I, a_J, b_I, b_J) - te(a_I, b_J, b_I, a_J)
            H[I, J] += val
            if I != J:
                H[J, I] = np.conj(H[I, J])
    return 0.5 * (H + H.conj().T)


# Config label helper
def cfg_label(a, b):
    wa = well.get(a, 'D'); wb = well.get(b, 'D')
    def lbl(idx, w):
        if w == 'L': return f"L{rank_L[idx]}"
        if w == 'R': return f"R{rank_R[idx]}"
        return f"D{idx}"
    return f"{lbl(a,wa)}-{lbl(b,wb)}"

_SPIN_TABLE = {
    'singlet':   (0.0,  0.0),
    'triplet_p': (2.0,  1.0),
    'triplet_0': (2.0,  0.0),
    'triplet_m': (2.0, -1.0),
}




def ci_to_spinorbital_Omega(coeffs, slater_basis, Nb):
    """
    Build antisymmetric 2-fermion amplitude matrix Ω_{pq} (p,q are spin-orbital indices)
    from CI coefficients in the spin-adapted slater_basis.
    NB: Gets built for one state only, so pass in the coeffs for that state (e.g. C2[:, n]) and the full slater_basis.

    This matches the structure used in compute_spin_and_entanglement(), but returns Ω.
    Conventions:
      spin-orbital index: p = 2*a (up), 2*a+1 (down)
      state |Ψ> = sum_{p<q} Ω_{pq} c_p^† c_q^† |0>
      Ω is antisymmetric: Ω_{pq} = -Ω_{qp}
    """
    coeffs = np.asarray(coeffs, dtype=complex)
    d = 2 * Nb
    Om = np.zeros((d, d), dtype=complex)

    for i, c in enumerate(coeffs):
        if abs(c) < 1e-14:
            continue
        a, b, _, stype = slater_basis[i]
        ia_up, ia_dn = 2*a, 2*a + 1
        ib_up, ib_dn = 2*b, 2*b + 1

        if a == b:
            # double occupancy (must be singlet): (|a↑ a↓> - |a↓ a↑>)/√2
            f = c / np.sqrt(2.0)
            Om[ia_up, ia_dn] += f
            Om[ia_dn, ia_up] -= f

        elif stype == 'singlet':
            # (|a↑ b↓> - |a↓ b↑> + |b↑ a↓> - |b↓ a↑>)/2
            f = c / 2.0
            Om[ia_up, ib_dn] += f
            Om[ia_dn, ib_up] -= f
            Om[ib_up, ia_dn] += f
            Om[ib_dn, ia_up] -= f

        elif stype == 'triplet_p':
            # (|a↑ b↑> - |b↑ a↑>)/√2
            f = c / np.sqrt(2.0)
            Om[ia_up, ib_up] += f
            Om[ib_up, ia_up] -= f

        elif stype == 'triplet_0':
            # (|a↑ b↓> + |a↓ b↑> - |b↑ a↓> - |b↓ a↑>)/2
            f = c / 2.0
            Om[ia_up, ib_dn] += f
            Om[ia_dn, ib_up] += f
            Om[ib_up, ia_dn] -= f
            Om[ib_dn, ia_up] -= f

        elif stype == 'triplet_m':
            # (|a↓ b↓> - |b↓ a↓>)/√2
            f = c / np.sqrt(2.0)
            Om[ia_dn, ib_dn] += f
            Om[ib_dn, ia_dn] -= f

    return Om

def spinorbital_U_from_spatial(U_spatial):
    """
    Given U_spatial of shape (M, M) mapping spatial orbitals,
    build U_spin of shape (2M, 2M) acting on spin-orbitals ordered as:
      (0↑,0↓, 1↑,1↓, ..., (M-1)↑,(M-1)↓)
    """
    M = U_spatial.shape[0]
    U_spin = np.zeros((2*M, 2*M), dtype=complex)
    for a in range(M):
        for mu in range(M):
            U_spin[2*a,   2*mu]   = U_spatial[a, mu]  # up->up
            U_spin[2*a+1, 2*mu+1] = U_spatial[a, mu]  # dn->dn
    return U_spin

def pair_basis(d):
    pairs = []
    idx = {}
    k = 0
    for p in range(d):
        for q in range(p+1, d):
            pairs.append((p, q))
            idx[(p, q)] = k
            k += 1
    return pairs, idx

def Omega_to_rho2_pair(Om, tol=1e-14):
    """
    Om: antisymmetric amplitude matrix Ω in spin-orbital basis (d x d)
    Returns: rho2 in antisymmetrised pair basis |pq> (p<q), size n2 x n2, trace=1
    This is the 2-fermion density matrix in the pair basis, which is rank-1 for pure states. Which state is determined by the input Ω.
    """
    d = Om.shape[0]
    pairs, pidx = pair_basis(d)
    amp = np.zeros(len(pairs), dtype=complex)

    for k, (p, q) in enumerate(pairs):
        amp[k] = Om[p, q]

    # normalise the state (just in case)
    nrm = np.vdot(amp, amp).real
    if nrm < tol:
        raise ValueError("State norm ~0 in pair basis; check Ω construction.")
    amp = amp / np.sqrt(nrm)

    rho2 = np.outer(amp, amp.conj()) #Density matrix coefficients in pair basis
    return rho2, pairs

# --- (E) Partial trace over right-well spin-orbitals to get ρ_L in NL=0,1,2 blocks ---

def rhoL_from_rho2_pairs_spin(rho2, pairs, mode_tags_spin):
    """
    mode_tags_spin: list length d with 'L' or 'R' for each spin-orbital index.
    rho2: 2-fermion density matrix in pair basis (pairs list gives ordering).

    Returns: rhoL0 (scalar), rhoL1 (nLso x nLso), rhoL2 (nLLpairs x nLLpairs), plus meta.
    """
    d = len(mode_tags_spin)
    pair_index = {p: i for i, p in enumerate(pairs)}

    L_modes = [m for m,t in enumerate(mode_tags_spin) if t == 'L']
    R_modes = [m for m,t in enumerate(mode_tags_spin) if t == 'R']
    nLso = len(L_modes)
    Lpos = {m:i for i,m in enumerate(L_modes)}

    # classify pairs
    LL_pairs = [(p,q) for (p,q) in pairs if mode_tags_spin[p]=='L' and mode_tags_spin[q]=='L']
    RR_pairs = [(p,q) for (p,q) in pairs if mode_tags_spin[p]=='R' and mode_tags_spin[q]=='R']

    # NL=2 block
    nLL = len(LL_pairs)
    rhoL2 = np.zeros((nLL, nLL), dtype=complex)
    for i,(p,q) in enumerate(LL_pairs):
        I = pair_index[(p,q)]
        for j,(p2,q2) in enumerate(LL_pairs):
            J = pair_index[(p2,q2)]
            rhoL2[i,j] = rho2[I,J]

    # NL=1 block
    rhoL1 = np.zeros((nLso, nLso), dtype=complex)
    for a in L_modes:
        ia = Lpos[a]
        for ap in L_modes:
            iap = Lpos[ap]
            s = 0.0 + 0.0j
            for r in R_modes:
                pr  = (a,r)  if a<r  else (r,a)
                prp = (ap,r) if ap<r else (r,ap)
                I = pair_index.get(pr, None)
                J = pair_index.get(prp, None)
                if I is None or J is None:
                    continue
                s += rho2[I,J]
            rhoL1[ia,iap] = s

    # NL=0 scalar
    rhoL0 = 0.0 + 0.0j
    for (r,s) in RR_pairs:
        I = pair_index[(r,s)]
        rhoL0 += rho2[I,I]

    meta = dict(L_modes=L_modes, R_modes=R_modes, LL_pairs=LL_pairs)
    return rhoL0, rhoL1, rhoL2, meta
# --- (F) Entropies ---

def vn_entropy(rho, tol=1e-12):
    if np.isscalar(rho):
        p = float(np.real(rho))
        if p < tol:
            return 0.0
        return float(-p*np.log(p))
    # hermitise
    rhoH = (rho + rho.conj().T) / 2.0
    w = np.linalg.eigvalsh(rhoH)
    w = w[w > tol]
    if w.size == 0:
        return 0.0
    return float(-np.dot(w, np.log(w)))

def entropy_blockdiag(rhoL0, rhoL1, rhoL2, tol=1e-12):
    return vn_entropy(rhoL0, tol) + vn_entropy(rhoL1, tol) + vn_entropy(rhoL2, tol)

def accessible_entropy(rhoL0, rhoL1, rhoL2, tol=1e-12):
    p0 = float(np.real(rhoL0))
    p1 = float(np.real(np.trace(rhoL1)))
    p2 = float(np.real(np.trace(rhoL2)))
    E = 0.0
    if p1 > tol:
        E += p1 * vn_entropy(rhoL1 / p1, tol)
    if p2 > tol:
        E += p2 * vn_entropy(rhoL2 / p2, tol)
    return float(E), (p0,p1,p2)

mode_tags_spin = []
for mu in range(M_LOC):
    mode_tags_spin.append(lr_labels[mu])  # up
    mode_tags_spin.append(lr_labels[mu])  # down


_SPIN_TABLE = {
    'singlet':   (0.0,  0.0),
    'triplet_p': (2.0,  1.0),
    'triplet_0': (2.0,  0.0),
    'triplet_m': (2.0, -1.0),
}

def compute_spin_and_entanglement(eigenstate_coeffs, slater_basis, Nb):
    """<S^2>, <Sz> and von Neumann entropy of the 1-RDM."""
    coeffs = np.asarray(eigenstate_coeffs, dtype=complex)
    probs  = np.abs(coeffs)**2
    S2 = Sz = 0.0
    for i, (_, _, _, st) in enumerate(slater_basis):
        s2, sz = _SPIN_TABLE.get(st, (0., 0.))
        S2 += probs[i] * s2
        Sz += probs[i] * sz
    d   = 2 * Nb
    rho = np.zeros((d, d), dtype=complex)
    for i, c in enumerate(coeffs):
        if abs(c) < 1e-12:
            continue
        a, b, _, stype = slater_basis[i]
        ia_up, ia_dn = 2*a, 2*a+1
        ib_up, ib_dn = 2*b, 2*b+1
        if a == b:
            f = c / np.sqrt(2)
            rho[ia_up, ia_dn] += f
            rho[ia_dn, ia_up] -= f
        elif stype == 'singlet':
            f = c / 2.
            rho[ia_up, ib_dn] += f
            rho[ia_dn, ib_up] -= f
            rho[ib_up, ia_dn] += f
            rho[ib_dn, ia_up] -= f
        elif stype == 'triplet_p':
            f = c / np.sqrt(2)
            rho[ia_up, ib_up] += f
            rho[ib_up, ia_up] -= f
        elif stype == 'triplet_0':
            f = c / 2.
            rho[ia_up, ib_dn] += f
            rho[ia_dn, ib_up] += f
            rho[ib_up, ia_dn] -= f
            rho[ib_dn, ia_up] -= f
        elif stype == 'triplet_m':
            f = c / np.sqrt(2)
            rho[ia_dn, ib_dn] += f
            rho[ib_dn, ia_dn] -= f
    rho1 = rho @ rho.conj().T
    eigv = np.linalg.eigvalsh((rho1 + rho1.conj().T) / 2.0)
    eigv = eigv[eigv > 1e-12]
    ent  = float(-np.dot(eigv, np.log(eigv))) if len(eigv) else 0.
    return float(S2), float(Sz), ent

def purify_degenerate_spin_subspaces(E, C, slater_basis, energy_tol=1e-6, spin_tol=1e-8):
    """
    Rotate CI eigenvectors within nearly degenerate energy blocks so they become
    simultaneous eigenvectors of S^2 and (where needed) S_z.

    This does not change the energies; it only chooses a cleaner basis inside each
    degenerate manifold.
    """
    n = min(len(E), C.shape[0], C.shape[1], len(slater_basis))
    E = np.asarray(E[:n], dtype=float)
    C_fix = np.array(C[:n, :n], dtype=complex, copy=True)

    s2_diag = np.array([_SPIN_TABLE[slater_basis[i][3]][0] for i in range(n)], dtype=float)
    sz_diag = np.array([_SPIN_TABLE[slater_basis[i][3]][1] for i in range(n)], dtype=float)
    S2_op = np.diag(s2_diag)
    Sz_op = np.diag(sz_diag)

    # group consecutive energies into nearly degenerate blocks
    blocks = []
    start = 0
    for i in range(1, n):
        if abs(E[i] - E[start]) > energy_tol:
            blocks.append(np.arange(start, i))
            start = i
    blocks.append(np.arange(start, n))

    def group_by_value(vals, tol):
        groups = []
        used = np.zeros(len(vals), dtype=bool)
        for i in range(len(vals)):
            if used[i]:
                continue
            grp = np.where(np.abs(vals - vals[i]) <= tol)[0]
            groups.append(grp)
            used[grp] = True
        return groups

    for blk in blocks:
        if len(blk) <= 1:
            continue

        C_blk = C_fix[:, blk]

        # First diagonalise S^2 within the degenerate manifold
        S2_blk = C_blk.conj().T @ S2_op @ C_blk
        s2_vals, U2 = eigh((S2_blk + S2_blk.conj().T) / 2.0)
        order = np.argsort(s2_vals)
        s2_vals = s2_vals[order]
        C_blk = C_blk @ U2[:, order]

        # Then diagonalise S_z inside each equal-S^2 subgroup (e.g. triplet manifold)
        for sub in group_by_value(s2_vals, spin_tol):
            if len(sub) <= 1:
                continue
            C_sub = C_blk[:, sub]
            Sz_blk = C_sub.conj().T @ Sz_op @ C_sub
            sz_vals, Uz = eigh((Sz_blk + Sz_blk.conj().T) / 2.0)
            order_z = np.argsort(sz_vals)
            C_blk[:, sub] = C_sub @ Uz[:, order_z]

        C_fix[:, blk] = C_blk

    return C_fix



# --- (H) Compute ρ_L for a chosen CI eigenstate ---

# IMPORTANT: U_loc was built only in the M_LOC subspace of delocalised eigenstates.
# Therefore we should restrict the CI->Ω construction to spatial orbitals < M_LOC.
# Easiest: build Ω on full 2*Nstates and then truncate to the 2*M_LOC block.

def truncate_Omega_to_subspace(Om_full, M_LOC):
    """Truncate Ω matrix to M_LOC spatial subspace (2*M_LOC spin-orbitals).
    
    Assertions verify that:
    1. Om_full is large enough (built from Nstates >= M_LOC)
    2. U_loc shape matches M_LOC (from localization step)
    """
    d = 2*M_LOC
    assert Om_full.shape[0] >= d, (
        f"Om_full shape {Om_full.shape[0]} < expected minimum 2*M_LOC={d}. "
        f"Check that Nstates >= M_LOC and M_LOC matches localization parameter."
    )
    assert U_loc.shape[0] == M_LOC, (
        f"U_loc shape {U_loc.shape[0]} != M_LOC={M_LOC}. "
        f"Localization was computed with different M_LOC!"
    )
    return Om_full[:d, :d].copy()

