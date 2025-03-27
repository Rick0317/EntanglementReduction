import numpy as np
from scipy.sparse.linalg import eigsh, expm
from scipy import linalg
from openfermion import BosonOperator, normal_ordered, get_boson_operator

def gen_initial_X_vector(n, squeezing = True, initial = 'id'):
    '''
    Parameters
    ----------
    n : # of modes
    squeezing : Whether not squeezing is included in Bogoliubov transform.
        DESCRIPTION. The default is True.

    intial: Guess for optimization, identity ('id') or random. ('rnd')
        Default is 'id'

    Returns
    ------
    Vector of parameters to be optimized X.
    if squeezing, len(X) = n*(n+1)/2 + n*(n-1)/2 = n**2
    if not squeexing len(X) = n*(n-1)/2
    '''

    P, Q = initial_rotation(n, squeezing = squeezing, initial = initial)
    X = flatten_matrices_only_PQ(P, Q, n, squeezing = squeezing)

    if squeezing:
        assert len(X) == n**2

    else:
        assert len(X) == n * (n-1) // 2

    return X


def flatten_matrices_only_PQ(P, Q, n, squeezing = True):
    # Modified from Sangeeth's code.

    # Flatten the matrices and the vector
    P_flat = skew_sym_to_vec(P,n)

    if squeezing:
        Q_flat = triu_to_vector(Q)

        # Concatenate all flattened arrays into a single vector
        X = np.concatenate([P_flat, Q_flat])

    else:
        X = P_flat

    return X


def triu_to_vector(matrix):
    #Taken from sangeeth's code.
    # create a vector contain upper trangular elements of a symmetric matrix
    upper_tri_ind = np.triu_indices_from(matrix)
    upper_tri_vector = matrix[upper_tri_ind]

    return upper_tri_vector

def skew_sym_to_vec(P, n):
    # Taken from Sangeeth's code.
    # extract n(n-1)/2 off diagonal upper terms from P
    vec = []

    for i in range(n):
        for j in range(i + 1, n):
            vec.append(P[i, j])

    return vec


def initial_rotation(n, squeezing = True, initial = 'id'):
    # Modified from Sangeeth's code.
    '''
    intial: Guess for optimization, identity ('id') or random. ('rnd')
        Default is 'id'
    '''

    assert initial in ['id', 'rnd']

    if initial == 'id':
        P = np.eye(n)
        Q = np.zeros((n,n))

    elif initial == 'rnd':
        p = np.random.uniform(-2, 2, (n,n))
        P = p - p.T
        # Q = 0 if no squeezing
        if squeezing:
            q = np.random.uniform(-2, 2, (n,n))
            Q = q.T + q

        else:
            Q = np.zeros((n,n))

        P *= 1e-6
        Q *= 1e-6

    return P, Q


def super_matrix_exp(P, Q):
    #Taken from Sangeeth's code.
    # create super matrix A=[[P,Q],[Q,P]] and exponentiate it(find [[U,-V],[-V,U]]= e^A)
    # Create the super matrix A
    A = np.block([[P, Q], [Q, P]])

    expA = linalg.expm(A)

    return expA


def extract_sub_matrices(expA, n):
    #Modified from Sangeeth's code
    # extract U and V matrices from expA [[U,-V],[-V,U]]= e^A)
    # Extract sub-matrices
    U = expA[:n, :n]
    V = -expA[:n, n:]

    #Define A = U - V and B = U + V.
    #These are the matrices that perform Bogoliubov transform for (x,p)
    A = U - V
    B = U + V
    S = [A, B]
    return S

def recreate_matrices_PQ_only(X, n, squeezing = True):
    #Modfied from Sangeeth's code.

    if squeezing:
        assert len(X) == n**2

        # Calculate the sizes of the arrays
        sp = n * (n - 1) // 2
        sq = n * (n + 1) // 2

        # Extract the arrays from X
        P_flat = X[:sp]
        Q_flat = X[sp:sp + sq]

        # Recreate matrices
        P = vec_to_skew_sym(P_flat,n)
        Q = vector_to_triu(Q_flat,n)

    else:
        assert len(X) == n * (n-1) // 2

        # Extract the arrays from X
        P_flat = X

        # Recreate matrices
        P = vec_to_skew_sym(P_flat,n)
        Q = np.zeros((n,n))

    return P,Q

def vec_to_skew_sym(vec, n):
    #Taken from Sangeeth's code.

    A = np.zeros((n, n))

    # Fill the upper triangular part
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            A[i, j] = vec[k]
            A[j, i] = -vec[k]
            k += 1

    return A

def vector_to_triu(vector, n):
    #Taken from Sangeeth's code.

    matrix = np.zeros((n, n))

    # upper triangular indices
    upper_tri_ind = np.triu_indices(n)

    # Adding the values from the vector to matrix
    matrix[upper_tri_ind] = vector

    # copy the upper triangular to the lower triangular
    i, j = upper_tri_ind
    matrix[j, i] = matrix[i, j]

    return matrix

def get_Smat_from_X(n, X, squeezing = True):
    '''
    Parameters
    ----------
    n : number of modes
    X : vector of optimization parameters
    squeezing: yes or no. True by default.


    Returns:
    -------
    S = [A, B]

    '''
    #Extract matrices P and Q from vector X
    P, Q = recreate_matrices_PQ_only(X, n, squeezing = squeezing)
    #Create Bog matrix M = exp([[P,Q],[Q,{}]])
    expA = super_matrix_exp(P, Q)
    #Extract matrices S = [A,B], from U anv V that make M = expA.
    S = extract_sub_matrices(expA, n)

    return S


def mode_swap(H, a, b):
    """
    Swap the mode a and b in H
    :param H: Bosonic Hamiltonian
    :param a: First mode
    :param b: Second mode
    :return:
    """
    H_bosonic = normal_ordered(get_boson_operator(H))
    result = BosonOperator()

    for term, coeff in H_bosonic.terms.items():
        new_term = BosonOperator('', 1.0)
        for op in term:
            index, action = op
            if index == a:
                new_term *= BosonOperator(((b, action),), 1.0)
            elif index == b:
                new_term *= BosonOperator(((a, action),), 1.0)
            else:
                new_term *= BosonOperator(((index, action),), 1.0)

        result += coeff * new_term

        result = normal_ordered(result)

    return result

