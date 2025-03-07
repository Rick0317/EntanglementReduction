
import numpy as np

from openfermion.ops import BosonOperator
from openfermion.transforms import normal_ordered
from openfermion import get_sparse_operator

from scipy.sparse.linalg import eigsh, expm
from scipy import linalg


def initial_only_PQ(n):
    p = np.random.uniform(-2, 2, (n,n))
    P = p.T + p

    q = np.random.uniform(-2, 2, (n,n))
    Q = q.T + q

    return P, Q


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


def flatten_matrices_only_PQ(P, Q, n):
    # Flatten the matrices and the vector
    P_flat = skew_sym_to_vec(P,n)
    Q_flat = triu_to_vector(Q)

    # Concatenate all flattened arrays into a single vector
    X = np.concatenate([P_flat, Q_flat])

    return X


def recreate_matrices_PQ_only(X,n):
    # Calculate the sizes of the arrays
    sp = n * (n - 1) // 2
    sq = n * (n + 1) // 2

    # Extract the arrays from X
    P_flat = X[:sp]
    Q_flat = X[sp:sp + sq]

    # Recreate matrices
    P = vec_to_skew_sym(P_flat,n)
    Q = vector_to_triu(Q_flat,n)

    return P,Q


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



def transformed_op_inv_no_translation(U,V):

    Ub = bosonic_B(U.T) # Calculate list of bosonic terms for U c
    Ubd = bosonic_BD(U.T) # Calculate list of bosonic terms for U c^dagger
    Vb = bosonic_B(V.T) # Calculate list of bosonic terms for V c
    Vbd = bosonic_BD(V.T) # Calculate list of bosonic terms for V c^dagger

    # tranformed operators
    B_b = []
    B_bd = []

    # Iterate over corresponding elements of the input lists
    for inUb, inUbd, inVb, inVbd in zip(Ub, Ubd, Vb, Vbd):
        # Add corresponding elements and append it to the result list
        B_b.append(inUb + inVbd)
        B_bd.append(inUbd + inVb)

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


#Compute part of the eigenspectrum of a bosonic operator using sparse methods.
def boson_eigenspectrum_sparse(operator, truncation, k):

    sparse_op = get_sparse_operator(operator, trunc=truncation)
    sparse_op = sparse_op.toarray()
    eigenvalues, eigenvectors = eigsh(sparse_op, k=k, which='SA')

    return eigenvalues, eigenvectors
