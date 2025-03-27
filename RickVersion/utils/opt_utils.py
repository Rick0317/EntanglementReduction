import numpy as np
from scipy.optimize import minimize
from RickVersion.utils.bog_utils import get_Smat_from_X
from RickVersion.utils.util_mutualinfo import mutual_info_full_n_sites, mutual_info_full
from RickVersion.utils.util_tensornetwork import get_approx_tensor, get_mps
from openfermion import get_sparse_operator, get_boson_operator, normal_ordered
from RickVersion.utils.ham_utils import get_H_op, get_H_ground_state, get_Hmol_params, get_H_ground_state_bosonic
from RickVersion.utils.util_gfro import quad_diagonalization

def optimize_bog_transfrom(X, H_params, nmax, bond_dimension, n_modes, options, cost_function = 'MI', tol = None, squeezing = True, mol = None):
    '''
    X : initial guess for bogoliubov transform
    H_params : [P2, V0, V1, V2, V3, V4] parameters of the original Hamiltonian
    nmax: truncation to construct Hamiltonian matrix
    bond_dimension: to be used to construct MPS
    options : options for minimization
    tol: tolerance for minimization. Default is none.
    cost function: one of mutual info ('MI') or energy ('energy')
    squeezing: yes or no. True by default.

    Returns
    -------
    Optimized parameters for bogoliubov transform based on given cost function

    '''

    def cost(X):
        if cost_function == 'MI':
            cf = MI_cost_func(X, H_params, nmax, bond_dimension, n_modes, squeezing = squeezing, mol=mol)

        elif cost_function == 'energy':
            cf = energy_cost_func(X, H_params, nmax, bond_dimension, squeezing = squeezing, mol=mol)

        elif cost_function == 'error':
            cf = mps_error_cost_func(X, H_params, nmax, bond_dimension, squeezing = squeezing)

        else :
            raise Exception("Cost function is not chosen as one of: 'MI'")

        return cf

    intermediate_data = []
    def printx(xk):
        cost_value = cost(xk)
        intermediate_error = get_intermediate_error(X, H_params, nmax, bond_dimension, n_modes, squeezing = squeezing, mol=mol)
        intermediate_data.append(cost_value)
        print(f"Current cost function: {cost_value}")

    print(f'Cost function before starting optimization: {cost(X)}')

    return minimize(cost, X, method='COBYLA', tol = tol, options=options, callback=printx), intermediate_data


def MI_cost_func(X, H_params, nmax, bond_dim, n_modes, squeezing = True, mol = None):
    '''
    X : list of Bogoliubov parameters
    H_params : [P2, V0, V1, V2, V3, V4] parameters of the original Hamiltonian
    nmax: truncation to construct Hamiltonian matrix
    bond_dimension: to be used to construct MPS
    squeezing: yes or no. True by default.

    Returns
    -------
    MI based cost function: \sum_ij I(i:j)

    '''

    # Read in molecular Hamiltonian

    S = get_Smat_from_X(n_modes, X, squeezing=squeezing)

    Ht_params = modified_H_params(H_params, S)

    del H_params

    #Get Ht as open fermion operator
    Ht = get_H_op(Ht_params)

    del Ht_params

    H2 = quad_diagonalization(normal_ordered(get_boson_operator(Ht)), n_modes)

    #Get Ht ground state eigenvector and eigenvalue
    _, gst = get_H_ground_state_bosonic(H2, nmax)

    del Ht

    reshaped_gst = gst.reshape(*(nmax,) * n_modes)

    MI_list = mutual_info_full_n_sites(reshaped_gst, nmax, n_modes)

    # MI_list = [I_xy, I_yz, I_xz]

    del reshaped_gst

    return weighted_sum_MI(MI_list, n_modes)

def energy_cost_func(X, H_params, nmax, bond_dim, squeezing=True, mol=None):
    '''
    X : list of Bogoliubov parameters
    H_params : [P2, V0, V1, V2, V3, V4] parameters of the original Hamiltonian
    nmax: truncation to construct Hamiltonian matrix
    bond_dimension: to be used to construct MPS
    squeezing: yes or no. True by default.

    Returns
    -------
    MI based cost function: \sum_ij I(i:j)

    '''

    if mol is not None:
        _, H_params = get_Hmol_params(mol)

    P2, V0, V1, V2, V3, V4 = H_params
    n = np.shape(V1)[0]

    S = get_Smat_from_X(n, X, squeezing=squeezing)

    Ht_params = modified_H_params(H_params, S)

    # Get Ht as open fermion operator
    Ht = get_H_op(Ht_params)

    # Get Ht ground state eigenvector and eigenvalue
    _, gst = get_H_ground_state(Ht, nmax)

    reshaped_gs = gst.reshape(*(nmax,) * n)

    approximate_gs = get_approx_tensor(reshaped_gs, bond_dim)

    reshaped_approximate = approximate_gs.reshape(nmax ** n, )

    sparse_op = get_sparse_operator(Ht, trunc=nmax)
    sparse_op_array = sparse_op.toarray()

    exp_value = np.vdot(reshaped_approximate,
                        sparse_op_array @ reshaped_approximate)
    norm_factor = np.vdot(reshaped_approximate, reshaped_approximate)
    normalized_exp_value = exp_value / norm_factor

    return normalized_exp_value.real


def mps_error_cost_func(X, H_params, nmax, bond_dim, squeezing=True):
    '''
    X : list of Bogoliubov parameters
    H_params : [P2, V0, V1, V2, V3, V4] parameters of the original Hamiltonian
    nmax: truncation to construct Hamiltonian matrix
    bond_dimension: to be used to construct MPS
    squeezing: yes or no. True by default.

    Returns
    -------
    MI based cost function: \sum_ij I(i:j)

    '''

    P2, V0, V1, V2, V3, V4 = H_params
    n = np.shape(V1)[0]

    S = get_Smat_from_X(n, X, squeezing=squeezing)

    Ht_params = modified_H_params(H_params, S)

    # Get Ht as open fermion operator
    Ht = get_H_op(Ht_params)

    # Get Ht ground state eigenvector and eigenvalue
    _, gst = get_H_ground_state(Ht, nmax)

    reshaped_gs = gst.reshape(nmax, nmax, nmax)

    error, _ = get_mps(reshaped_gs, bond_dim)

    return error


def modified_H_params(H_params, S):
    '''
    Parameters
    ----------
    H_params : [T2, V0, V1, V2, V3, V4] parameters of the original Hamiltonian
    S = [A,B] needed to transform H_params

    Returns
    -------
    Modified H_params :  [T2t, V0t, V1t, V2t, V3t, V4t]

    '''

    T2, V0, V1, V2, V3, V4 = H_params

    #Define C = B.T and D = A.T, matrices need to transform tensors.
    A, B = S
    C = B.T
    D = A.T

    #Define transformed tensors
    V0t = V0
    V1t = np.einsum('i,ia->a',V1,C)
    V2t = np.einsum('ij,ia,jb->ab',V2,C,C)
    T2t = np.einsum('ij,ia,jb->ab',T2,D,D)
    V3t = np.einsum('ijk,ia,jb,kc->abc',V3,C,C,C)
    V4t = np.einsum('ijkl,ia,jb,kc,ld->abcd',V4,C,C,C,C)

    return [T2t, V0t, V1t, V2t, V3t, V4t]




def get_intermediate_error(X, H_params, nmax, bond_dim, n_modes, squeezing = True, mol = None):
    '''
    X : list of Bogoliubov parameters
    H_params : [P2, V0, V1, V2, V3, V4] parameters of the original Hamiltonian
    nmax: truncation to construct Hamiltonian matrix
    bond_dimension: to be used to construct MPS
    squeezing: yes or no. True by default.

    Returns
    -------
    MI based cost function: \sum_ij I(i:j)

    '''

    # Read in molecular Hamiltonian
    if mol is not None:
        _, H_params = get_Hmol_params(mol)

    _, _, V1, _, _, _ = H_params
    n= np.shape(V1)[0]

    S = get_Smat_from_X(n, X, squeezing=squeezing)

    Ht_params = modified_H_params(H_params, S)

    del H_params

    #Get Ht as open fermion operator
    Ht = get_H_op(Ht_params)

    del Ht_params

    #Get Ht ground state eigenvector and eigenvalue
    _, gst = get_H_ground_state(Ht, nmax)

    reshaped_gs = gst.reshape(*(nmax,) * n_modes)
    error, _ = get_mps(reshaped_gs, bond_dim)
    return error


def weighted_sum_MI(MI, n):
    """
    Compute the weighted sum of mutual information elements, weighted by |i - j|^2.

    :param MI: List of mutual information values in upper triangular order.
    :param n: Number of modes.
    :return: Weighted sum.
    """
    weighted_sum = 0
    index = 0

    for i in range(n):
        for j in range(i + 1, n):
            weight = (i - j) ** 2  # Compute |i - j|^2
            weighted_sum += weight * MI[index]
            index += 1

    return weighted_sum
