
import numpy as np

from RickVersion.utils.util_hamil import extract_coeffs_and_ops,H_real
from RickVersion.utils_temporary.util_qudratic import H_quadatic, H_cubic_only,b_transform, H_quartic_only

from openfermion.ops import BosonOperator
from openfermion.transforms import normal_ordered
from openfermion import get_sparse_operator

from scipy.sparse.linalg import eigsh, expm
from scipy import linalg



# find norm of the Hamiltonian but only for cubic and quartic terms
def norm_m(H):

    H_vector,ops = extract_coeffs_and_ops(H)

    indices_cq = [i for i, op in enumerate(ops) if len(op) == 3 or len(op) == 4]

    norm_cq = []

    for i in indices_cq:
        value = H_vector[i]**2
        norm_cq.append(value)

    norm_cqs = np.real(sum(norm_cq))

    return norm_cqs


# creates vectors that contains the coefficients with matching indices of two bosonic operators
def vectors(H, H_f):

    H_coeffs, H_ops = extract_coeffs_and_ops(H)
    H_f_coeffs, H_f_ops = extract_coeffs_and_ops(H_f)

    H_dict = {op: coeff for op, coeff in zip(H_ops, H_coeffs)}
    H_f_dict = {op: coeff for op, coeff in zip(H_f_ops, H_f_coeffs)}

    # Ensure both Hamiltonians have the same operators
    all_ops = set(H_ops).union(set(H_f_ops))

    # Sort the operators
    sorted_ops = sorted(all_ops, key=lambda x: (len(x), x))

    # Create vectors with matching indices
    H_vector = [H_dict.get(op, 0) for op in sorted_ops]
    H_f_vector = [H_f_dict.get(op, 0) for op in sorted_ops]

    return H_vector, H_f_vector, sorted_ops

# reconstruct bosonic operator from coefficients and operators
def reconstruct_boson_operators(coeffs, ops):
    boson_operator = BosonOperator()
    for coeff, op in zip(coeffs, ops):
        boson_operator += BosonOperator(op, coeff)
    return boson_operator

# subtract elements in two lists
def subtract_lists(list1, list2):
    return [a - b for a, b in zip(list1, list2)]

#Subtracting one Bosonic Hamitonian from another one

def subtract(H, H_f):
    # Extract coefficients and operators
    H_coeffs, H_ops = extract_coeffs_and_ops(H)
    H_f_coeffs, H_f_ops = extract_coeffs_and_ops(H_f)

    H_dict = {op: coeff for op, coeff in zip(H_ops, H_coeffs)}
    H_f_dict = {op: coeff for op, coeff in zip(H_f_ops, H_f_coeffs)}

    # Ensure both Hamiltonians have the same operators
    all_ops = set(H_ops).union(set(H_f_ops))

    # Sort the operators
    sorted_ops = sorted(all_ops, key=lambda x: (len(x), x))

    # Create vectors with matching indices
    H_vector = [H_dict.get(op, 0) for op in sorted_ops]
    H_f_vector = [H_f_dict.get(op, 0) for op in sorted_ops]

    H_new = subtract_lists(H_vector,H_f_vector)

    H = reconstruct_boson_operators(H_new, sorted_ops)

    return H


# Define the initial parameters in the exponent A=[[P,Q],[Q,P]], L and G
def initial(n):
    p = np.random.uniform(-2, 2, (n,n))
    P = p.T + p # P satifies P=P^T

    q = np.random.uniform(-2, 2, (n,n))
    Q = q.T + q # Q satifies Q=Q^T

    # Random gamma vector
    G = np.random.uniform(-1, 1, n)

    # Random lambda matrix
    L = np.random.uniform(0, 1, (n, n))

    return P,Q,G,L


# create super matrix A=[[P,Q],[Q,P]] and exponentiate it(find [[U,-V],[-V,U]]= e^A)
def super_matrix_exp(P, Q):

    # Create the super matrix A
    A = np.block([[P, Q],[Q, P]])

    expA = linalg.expm(A)

    return expA


# exatract U and V matrices from expA [[U,-V],[-V,U]]= e^A)
def extract_sub_matrices(expA,n):

    # Extract sub-matrices
    U = expA[:n, :n]
    V = -expA[:n, n:]

    return U,V

# create a vector contain upper trangular elements of a symmetric matrix
def triu_to_vector(matrix):

    upper_tri_ind = np.triu_indices_from(matrix)
    upper_tri_vector = matrix[upper_tri_ind]

    return upper_tri_vector

# recreate symmetric matrix from triangular vector of size n(n+1)/2
def vector_to_triu(vector,n):

    matrix = np.zeros((n, n))

    # upper triangular indices
    upper_tri_ind = np.triu_indices(n)

    # Adding the values from the vector to matrix
    matrix[upper_tri_ind] = vector

    # copy the upper triangular to the lower triangular
    i, j = upper_tri_ind
    matrix[j, i] = matrix[i, j]

    return matrix

# extract n(n-1)/2 off diagonal upper terms from P
def skew_sym_to_vec(P,n):

    vec = []

    for i in range(n):
        for j in range(i + 1, n):
            vec.append(P[i, j])

    return vec

# Create skew_symmetric P
def vec_to_skew_sym(vec,n):

    A = np.zeros((n, n))

    # Fill the upper triangular part
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            A[i, j] = vec[k]
            A[j, i] = -vec[k]
            k += 1

    return A

# Calculate quartic fragments from calculated L, C_b, C_bd
def compute_fragment(L, C_b, C_bd, n):

    result = BosonOperator()

    for i in range(n):
        for j in range(n):

            # Calculate the term L[i, j] * C_bd[i]^T * C_b[i] * C_bd[j]^T * C_b[j]
            term = L[i,j] * C_bd[i] * C_b[i] * C_bd[j] * C_b[j]

            result += term

    result = normal_ordered(result)

    return result

# combine necessary parmaters in P,Q,G,L in to single vector X
def flatten_matrices(P,Q,G,L,n):
    # Flatten the matrices and the vector
    P_flat = skew_sym_to_vec(P,n)
    Q_flat = triu_to_vector(Q)
    L_flat = L.flatten()

    # Concatenate all flattened arrays into a single vector
    X = np.concatenate([P_flat, Q_flat, G, L_flat])

    return X

#recreate P,Q,G,L from vector X
def recreate_matrices(X,n):
    # Calculate the sizes of the arrays
    sp = n * (n - 1) // 2
    sq = n * (n + 1) // 2
    sg = n

    # Extract the arrays from X
    P_flat = X[:sp]
    Q_flat = X[sp:sp + sq]
    G = X[sp + sq:sp + sq + sg]
    L_flat = X[sp + sq + sg:]

    # Recreate matrices
    P = vec_to_skew_sym(P_flat,n)
    Q = vector_to_triu(Q_flat,n)
    L = L_flat.reshape((n, n))

    return P,Q,G,L


# Compute quartic fragments from X
def fragments(X,n):

    P,Q,G,L = recreate_matrices(X,n)

    expA = super_matrix_exp(P, Q)

    U,V = extract_sub_matrices(expA,n)

    C_b,C_bd = tranformed_operators(U,V,G)

    H_f = compute_fragment(L,C_b,C_bd,n)

    return H_f

# Remove the fragments that have norm less than 10^-3
def filter_params(params,n):
    threshold = 1e-3
    filtered_params = [param for param in params if norm_m(fragments(param,n)) >= threshold]
    return filtered_params

# Create a vecctor for calculations
def create_bosonic_vector(n):
    return [1.0] * n



# Calculate list of bosonic terms for U b^dagger
def bosonic_BD(U):

    n=np.shape(U)[1]

    B = create_bosonic_vector(n)

    # bosonic operators
    bosonic_operators_listU = []


    for row in U:
        # Initialize an empty BosonOperator for the current row
        bosonic_operator = BosonOperator()

        # Iterate over the elements of B and the current row of U
        for coeff, index in zip(B, range(len(row))):
            # Multiply the coefficient with corresponding number in the row and create a term
            term = BosonOperator(((index,1),), coeff * row[index])

            # Add the term to the BosonOperator for the current row
            bosonic_operator += term

        # Append the resulting BosonOperator to the list
        bosonic_operators_listU.append(bosonic_operator)

    return bosonic_operators_listU



# Calculate list of bosonic terms for U b
def bosonic_B(U):

    n=np.shape(U)[1]

    B = create_bosonic_vector(n)

    # bosonic operators
    bosonic_operators_listU = []


    for row in U:
        # Initialize an empty BosonOperator for the current row
        bosonic_operator = BosonOperator()

        # Iterate over the elements of B and the current row of U
        for coeff, index in zip(B, range(len(row))):
            # Multiply the coefficient with the corresponding number in the row and create a term
            term = BosonOperator(((index,0),), coeff * row[index])

            # Add the term to the BosonOperator for the current row
            bosonic_operator += term

        # Append the resulting BosonOperator to the list
        bosonic_operators_listU.append(bosonic_operator)

    return bosonic_operators_listU



# Calculate list of bosonic terms corresponding to the constant term
def bosonic_G(G):

    # resulting terms
    iden_list = []


    for coeff in G.flat:
        # Create a term by multiplying the coefficient with the bosonic identity operator
        term = BosonOperator('', coeff)

        # Append the resulting term to the list
        iden_list.append(term)

    return iden_list



# Calculate transformed operators
def tranformed_operators(U,V,G):

    Ub = bosonic_B(U) # Calculate list of bosonic terms for U b
    Ubd = bosonic_BD(U) # Calculate list of bosonic terms for U b^dagger
    Vb = bosonic_B(V) # Calculate list of bosonic terms for V b
    Vbd = bosonic_BD(V) # Calculate list of bosonic terms for V b^dagger
    Gi = bosonic_G(G) # Calculate list of bosonic terms for I G

    # tranformed operators
    C_b = []
    C_bd = []

    # Iterate over corresponding elements of the input lists
    for inUb, inUbd, inVb, inVbd, inGi in zip(Ub, Ubd, Vb, Vbd, Gi):
        # Add corresponding elements and append it to the result list
        C_b.append(inUb - inVbd + inGi)
        C_bd.append(inUbd - inVb + inGi)

    return C_b, C_bd



# Calculate list of bosonic terms in terms of 'c' operators corresponding to the constant term
def bosonic_G_inv(G,U,V):

    # resulting terms
    iden_list = []

    G1 = U@G+V@G

    for coeff in G1.flat:
        # Create a term by multiplying the coefficient with the bosonic identity operator
        term = BosonOperator('', coeff)

        # Append the resulting term to the list
        iden_list.append(term)

    return iden_list



# Calculate transformed operators in terms of 'c' operators  # refer page 289 Bogoluibov book
def tranformed_operators_inv(U,V,G):

    Ub = bosonic_B(U.T) # Calculate list of bosonic terms for U c
    Ubd = bosonic_BD(U.T) # Calculate list of bosonic terms for U c^dagger
    Vb = bosonic_B(V.T) # Calculate list of bosonic terms for V c
    Vbd = bosonic_BD(V.T) # Calculate list of bosonic terms for V c^dagger
    Gi = bosonic_G_inv(G,U.T,V.T) # Calculate list of bosonic terms for I G

    # tranformed operators
    B_b = []
    B_bd = []

    # Iterate over corresponding elements of the input lists
    for inUb, inUbd, inVb, inVbd, inGi in zip(Ub, Ubd, Vb, Vbd, Gi):
        # Add corresponding elements and append it to the result list
        B_b.append(inUb + inVbd - inGi)
        B_bd.append(inUbd + inVb - inGi)

    return B_b, B_bd

# write original bosonic operators in terms of transformed bosonic operators
def substitute_operators(H, C_b, C_bd):
    result = BosonOperator()

    for term, coeff in H.terms.items():
        new_term = BosonOperator('', 1.0)

        for op in term:
            index, action = op

            if action == 0:  # annihilation
                new_term *= C_b[index]
            elif action == 1:  # creation
                new_term *= C_bd[index]

        result += coeff * new_term

        result = normal_ordered(result)

    return result

# Create the diagonal operator in terms of transformed bosonic operators
def create_diagonal_hamiltonian(L):
    n = L.shape[0]

    H = BosonOperator()

    for p in range(n):
        for q in range(n):
            coefficient = L[p, q]
            # Create the term c_p^† * c_p * c_q^† * c_q
            term = ((p, 1), (p, 0), (q, 1), (q, 0))
            H += BosonOperator(term, coefficient)

    return H

# Create the diagonal operator in terms of transformed bosonic operators
def create_diagonal_quadratic(L):
    n = L.shape[0]

    H = BosonOperator()

    for p in range(n):
        coefficient = L[p]
        # Create the term c_p^† * c_p * c_q^† * c_q
        term = ((p, 1), (p, 0))
        H += BosonOperator(term, coefficient)

    return H

#Compute part of the eigenspectrum of a bosonic operator using sparse methods.
def boson_eigenspectrum_sparse(operator, truncation, k):

    sparse_op = get_sparse_operator(operator, trunc=truncation)
    sparse_op = sparse_op.toarray()
    eigenvalues, eigenvectors = eigsh(sparse_op, k=k, which='SA')

    return eigenvalues, eigenvectors

def boson_eigenspectrum_full(operator, truncation):
    # Get the sparse operator
    sparse_op = get_sparse_operator(operator, trunc=truncation)

    # Convert to full matrix
    full_matrix = sparse_op.toarray()

    eigenvalues, evec = np.linalg.eigh(full_matrix)

    return eigenvalues, evec


def get_time_evolution(operator, truncation, state, t):
    sparse_op = get_sparse_operator(operator, trunc=truncation)

    full_matrix = sparse_op.toarray()

    time_op = expm(-1j * full_matrix * t)
    result = time_op @ state
    # print(result[:10])

    return result



def tensor_product(*matrices):
    result = matrices[0]
    for matrix in matrices[1:]:
        result = np.kron(result, matrix)
    return result


# Hamitonian after rotation
def rotated_hamiltonian(H,X,n):

    P,Q,G,L = recreate_matrices(X,n)

    expA = super_matrix_exp(P, Q)

    U,V = extract_sub_matrices(expA,n)

    B_b,B_bd = tranformed_operators_inv(U,V,G)

    return substitute_operators(H, B_b, B_bd)

# diagonalize Hamiltonian up to quadratic with respect to bosonic operators
def quad_diagonalization(H,n):

    Hq = H_quadatic(H)

    Hc = H_cubic_only(H)

    U,V,L,G,K = b_transform(Hq,n)

    B_b,B_bd = tranformed_operators_inv(U,V,G)

    Hqr = create_diagonal_quadratic(L.real.flatten()) + K.real.flatten()[0] * BosonOperator('')

    Hcr = substitute_operators(Hc, B_b, B_bd)

    H4 = H_quartic_only(H)

    H4r = substitute_operators(H4, B_b, B_bd)

    Hr1 = Hqr + Hcr + H4r

    return H_real(Hr1)
