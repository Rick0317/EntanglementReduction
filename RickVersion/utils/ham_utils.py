import numpy as np
import h5py
import itertools
from openfermion import get_sparse_operator, is_hermitian
from openfermion.ops import QuadOperator
from openfermion.transforms import get_boson_operator, normal_ordered
from scipy.sparse.linalg import eigsh


def get_Hmol_params(mol):

    if mol in ['H2S','CO2','CH2O','C2N2']:
        nmodes, Ham = hamil_params(mol)

    #elif mol in ['H2O','CO']:
    #    Ham = operator_quad(mol)

    else:
        print('Molecule not supported')
        exit

    return nmodes, Ham


def hamil_params(name):
    with h5py.File(rf'../../Mol_data/{name}.hdf5', 'r') as f:
    #with h5py.File(rf'C:/Users/shrey/Documents/Izmaylov Lab/Vibrational Spectra/4T3M_Hamiltonians/{name}.hdf5', 'r') as f:
        freqs = f["freqs"][()]
        taylor_1D = f["taylor_1D"][()]
        taylor_2D = f["taylor_2D"][()]
        taylor_3D = f["taylor_3D"][()]

    nmodes = len(freqs)
    _, num_1D_coeffs = np.shape(taylor_1D)

    # Up to quartic. Taylor_deg = 4
    taylor_deg = num_1D_coeffs+2
    assert taylor_deg <= 4

    T2 = 1/2 * np.diag(freqs)
    V0 = 0
    V1 = np.zeros(nmodes)
    V2 = 0.5*np.diag(freqs)
    V3 = np.zeros((nmodes,nmodes,nmodes))
    V4 = np.zeros((nmodes,nmodes,nmodes,nmodes))

    # Add in one-mode anharmonic terms:
    for m in range(nmodes):
        for deg_i in range(3, taylor_deg+1):
            if deg_i == 3:
                V3[m, m, m] = taylor_1D[m, deg_i-3]
            elif deg_i == 4:
                V4[m, m, m, m] = taylor_1D[m, deg_i-3]
            else:
                ValueError(f'Taylor degree >4 not supported')

    #Add in two-mode anharmonic terms:
    degs_2d = find_2d_degs(taylor_deg)
    for m1 in range(nmodes):
        for m2 in range(m1):
            for deg_idx, Qs in enumerate(degs_2d):
                idx = Qs_to_tensor_index_2mode(Qs,m1,m2)
                if np.sum(Qs) == 3:
                    V3[idx] = taylor_2D[m1,m2,deg_idx]
                    #print(f"V3[{idx}] --> {V3[idx]}")
                elif np.sum(Qs) == 4:
                    V4[idx] = taylor_2D[m1,m2,deg_idx]
                    #print(f"V4[{idx}] --> {V4[idx]}")
                q1deg = Qs[0]
                q2deg = Qs[1]
                #print(f"q{m1}^{q1deg}*q{m2}^{q2deg} --> {taylor_2D[m1,m2,deg_idx]}")


    #Add in three-mode anharmonic terms:
    degs_3d = find_3d_degs(taylor_deg)

    for m1 in range(nmodes):
        for m2 in range(m1):
            for m3 in range(m2):
                for deg_idx, Qs in enumerate(degs_3d):
                    idx = Qs_to_tensor_index_3mode(Qs,m1,m2,m3)
                    if np.sum(Qs) == 3:
                        V3[idx] = taylor_3D[m1,m2,m3,deg_idx]
                        #print(f"V3[{idx}] --> {V3[idx]}")
                    elif np.sum(Qs) == 4:
                        V4[idx] = taylor_3D[m1,m2,m3,deg_idx]
                        #print(f"V4[{idx}] --> {V4[idx]}")
                    q1deg = Qs[0]
                    q2deg = Qs[1]
                    q3deg = Qs[2]
                    #print(f"q{m1}^{q1deg}*q{m2}^{q2deg}*q{m3}^{q3deg} --> {taylor_3D[m1,m2,m3,deg_idx]}")

    return nmodes, [T2, V0, V1, V2, V3, V4]

def Qs_to_tensor_index_2mode(Qs,m1,m2):
    assert len(Qs) == 2
    idx = ()
    for i in range(Qs[0]):
        idx = (*idx, m1)
    for i in range(Qs[1]):
        idx = (*idx, m2)

    return idx

def Qs_to_tensor_index_3mode(Qs,m1,m2,m3):
    assert len(Qs) == 3
    idx = ()
    for i in range(Qs[0]):
        idx = (*idx, m1)
    for i in range(Qs[1]):
        idx = (*idx, m2)
    for i in range(Qs[2]):
        idx = (*idx, m3)

    return idx

def find_2d_degs(deg):
	fit_degs = []
	deg_idx = 0
	for feat_deg in range(3,deg+1):
		max_deg = feat_deg - 1
		for deg_dist in range(1,max_deg+1):
			q1deg = max_deg-deg_dist+1
			q2deg = deg_dist
			fit_degs.append((q1deg,q2deg))
			deg_idx += 1

	return fit_degs

def find_3d_degs(deg):
	fit_degs = []
	deg_idx = 0
	for feat_deg in range(3,deg+1):
		max_deg = feat_deg - 3
		possible_occupations = generate_bin_occupations(max_deg, 3)
		for occ in possible_occupations:
			q1deg = 1 + occ[0]
			q2deg = 1 + occ[1]
			q3deg = 1 + occ[2]
			fit_degs.append((q1deg,q2deg,q3deg))
			deg_idx += 1

	return fit_degs

def generate_bin_occupations(max_occ, nbins):
    # Generate all combinations placing max_occ balls in nbins
    combinations = list(itertools.product(range(max_occ+1), repeat=nbins))

    # Filter valid combinations
    valid_combinations = [combo for combo in combinations if sum(combo) == max_occ]

    return valid_combinations

def get_H_op(H_params):
    '''
    Parameters
    ----------
    H_params : [T2, V0, V1, V2, V3, V4]
        T2 is the coefficient matrix for the kinetic energy
        Vj is the rank-j tensor of coefficients for the j^th order term in position

    H = \sum_ij P2_ij p_i p_j + V0 + \sum_i V1_i x_i + \sum_ij V2_ij x_i x_j
      + \sum_ijk V3_ij x_i x_j x_k  + \sum_ijkl V4_ijkl x_i x_j x_k x_l

    For V2, sum is only terms with i<=j are non-zero to avoid over counting.
    Similar for V3 and V4.

    Returns: H as open_fermion operator

    '''
    T2, V0, V1, V2, V3, V4 = H_params
    nmodes = np.shape(V1)[0]

    H = QuadOperator('',V0)

    for i in range(nmodes):
        H +=  QuadOperator(f'q{i}', V1[i])
        for j in range(nmodes):
            H += QuadOperator(f'q{i} q{j}', V2[i,j]) + QuadOperator(f'p{i} p{j}', T2[i,j])
            for k in range(nmodes):
                H += QuadOperator(f'q{i} q{j} q{k}', V3[i,j,k])
                for l in range(nmodes):
                    H += QuadOperator(f'q{i} q{j} q{k} q{l}', V4[i,j,k,l])

    return H

def get_H_ground_state(H, nmax):
    '''
     H: of Quadoperator
     nmax: truncation for constructing H matrix

    Return: Ground eigenvalue E0, and ground state as array
    '''

    H_bosonic = normal_ordered(get_boson_operator(H))
    H_sparse  = get_sparse_operator(H_bosonic, trunc=nmax)

    return eigsh(H_sparse, k=1, which='SA')


def get_H_ground_state_bosonic(H_bosonic, nmax):
    '''
     H: of Quadoperator
     nmax: truncation for constructing H matrix

    Return: Ground eigenvalue E0, and ground state as array
    '''
    H_sparse  = get_sparse_operator(H_bosonic, trunc=nmax).toarray()

    return eigsh(H_sparse, k = 1, which = 'SA')

# def boson_eigenspectrum_sparse(operator, truncation, k):
#     sparse_op = get_sparse_operator(operator, trunc=truncation)
#     sparse_op = sparse_op.toarray()
#     eigenvalues, eigenvectors = eigsh(sparse_op, k=k, which='SA')
#
#     return eigenvalues, eigenvectors


def truncate_quad_operator(op: QuadOperator,
                           threshold: float = 1e-4) -> QuadOperator:
    """Removes terms from a QuadOperator that have coefficients below the threshold."""
    truncated_op = QuadOperator()

    for term, coeff in op.terms.items():
        if abs(coeff) >= threshold:
            truncated_op += QuadOperator(term, coeff)

    return truncated_op
