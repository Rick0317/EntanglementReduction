import numpy as np
from openfermion.ops import BosonOperator
from openfermion.transforms import normal_ordered
from openfermion import get_sparse_operator
from scipy.sparse.linalg import eigsh
from scipy import linalg
from scipy.linalg import eigh
from scipy.linalg import eig
from ten_network import *
from util_mutualinfo import mutual_all


# reconstruct bosonic operator from coefficients and operators
def reconstruct_boson_operators(coeffs, ops):
    boson_operator = BosonOperator()
    for coeff, op in zip(coeffs, ops):
        boson_operator += BosonOperator(op, coeff)
    return boson_operator

# Makning sure that the coefficients are all real
def H_real(H):
    Hv,ops = extract_coeffs_and_ops(H)
    Hv =[np.real(element) for element in Hv]
    H = reconstruct_boson_operator(Hv, ops)

    return H

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

# Function to extract coefficients and operators
def extract_coeffs_and_ops(boson_operator):
    coeffs = []
    ops = []
    for term, coeff in boson_operator.terms.items():
        ops.append(term)
        coeffs.append(coeff)
    return coeffs, ops

# Define the initial parameters in the exponent A=[[P,Q],[Q,P]], L and G 
def initial1(n):
   # p = np.random.uniform(-2, 2, (n,n))
    p = np.ones((n,n))
    P = p.T - p # P satifies P=-P^T
    
   # q = np.random.uniform(-2, 2, (n,n))
    q = np.ones((n,n))
    Q = q.T + q # Q satifies Q=Q^T
    
    # Random gamma vector
   # G = np.random.uniform(-1, 1, n)
    
    G = np.ones(n)
    
    # Random lambda matrix
   # L = np.random.uniform(0, 1, (n, n))
   
    L = np.ones((n,n))
      
    
    return P,Q,G,L


# Define the initial parameters in the exponent A=[[P,Q],[Q,P]], L and G 
def initial(n):
    p = np.random.uniform(-2, 2, (n,n))
    P = p.T - p # P satifies P=-P^T
    
    q = np.random.uniform(-2, 2, (n,n))
    Q = q.T + q # Q satifies Q=Q^T
    
    # Random gamma vector
    G = np.random.uniform(-1, 1, n)
    
    # Random lambda matrix
    L = np.random.uniform(0, 1, (n, n))
        
    return P,Q,G,L

def initial_no_s_d(n):
    p = np.random.uniform(-2, 2, (n, n))
    P = p.T - p
    Q = np.zeros((n, n))  # No squeezing
    G = np.zeros(n)       # No displacement
    L = np.random.uniform(0, 1, (n, n))
    return P, Q, G, L


# create super matrix A=[[P,Q],[Q,P]] and exponentiate it(find [[U,-V],[-V,U]]= e^A)
def super_matrix_exp(P, Q):
    
    # Create the super matrix A 
    A = np.block([[P, Q],[Q, P]])
    
    expA = linalg.expm(A)
       
    return expA


# exatract U and V matrices from expA [[U.T,-V.T],[-V.T,U.T]]= e^A)
def extract_sub_matrices(expA,n):
    
    # Extract sub-matrices
    U = expA[:n, :n]
    V = -expA[:n, n:]
    
    return U.T,V.T

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


# reorder so that dagger operators always comes first
def reorder_brackets(ops):
    reordered_terms = []
    for term in ops:
        first, second = term
        if first[1] < second[1]:
            reordered_terms.append((second, first))
        else:
            reordered_terms.append((first, second))
    return reordered_terms

# Create A and B matrices as in eq 2.1 Intro to Quantum stat mech Bogolubov
def create_AB(H_q,n):
    
    H_c,ops=extract_coeffs_and_ops(H_q)
    
    ops = reorder_brackets(ops) # reorder so that dagger operators always comes first

    A = np.zeros((n, n))
    B = np.zeros((n, n))

    for i, term in enumerate(ops):

        first, second = term
        
        if first[1] == 1 and second[1] == 0: # b_p' b_q                   
                A[first[0], second[0]] = H_c[i]
                A[second[0], first[0]] = H_c[i]
        elif first[1] == 1 and second[1] == 1: # b_p' b_q' 
            if first[0] == second[0]:
                B[first[0], second[0]] =  2*H_c[i]
            else:
                B[first[0], second[0]] = H_c[i]
                B[second[0], first[0]] = H_c[i]
               
    return A,B


def solve_uv_lambda(A, B, n):
    #I = np.eye(n)
    # Construct the block matrix for the eigenvalue problem
    top_block = np.hstack((A, B))
    bottom_block = np.hstack((-B, -A))
    full_matrix = np.vstack((top_block, bottom_block))

    # Solve the eigenvalue problem
    eigenvalues, eigenvectors = eig(full_matrix)
    

    '''
    positive_indices = np.real(eigenvalues) + np.imag(eigenvalues) > 0
    eigenvalues = eigenvalues[positive_indices]
    eigenvectors = eigenvectors[:, positive_indices]
    '''    
    #print(eigenvalues)
    
    # Extract the solutions for lambda, u, and v
    lambdas = []
    us = []
    vs = []

    for i in range(len(eigenvalues)):
        lambda_k = eigenvalues[i]
        
        # Split the eigenvector into u and v components
        u = eigenvectors[:n, i]
        v = eigenvectors[n:, i]
        
        lambdas.append(lambda_k)
        us.append(u)
        vs.append(v)       

    return  np.array(us).T, np.array(vs).T,np.array(lambdas)


def choose_uv(U,V,L,n):
    
    ind = []
    
    for i in range(2*n):
        
        if U[:,i].T @ U[:,i] > V[:,i].T @ V[:,i]:
        
            ind.append(i)
            
    ind = np.array(ind)
    
    U = U[:,ind]
    V = V[:,ind]
    L = L[ind]
    
    #print(L)
    
    return U,V,L.reshape(-1, 1)


# reconstruct bosonic operator from coefficients and operators
def reconstruct_boson_operator(coeffs, ops):
    boson_operator = BosonOperator()
    for coeff, op in zip(coeffs, ops):
        boson_operator += BosonOperator(op, coeff)
    return boson_operator


def scale_matrices(U, V,n):

    U_scaled = np.zeros_like(U)
    V_scaled = np.zeros_like(V)
    
    for j in range(n):
        u_j = np.dot(U[:, j], U[:, j])
        v_j = np.dot(V[:, j], V[:, j])
        
        alpha_j = 1 / np.sqrt(u_j - v_j)
        
        U_scaled[:, j] = alpha_j * U[:, j]
        V_scaled[:, j] = alpha_j * V[:, j]
    
    return U_scaled, V_scaled

def create_C(H_l,n):
    
    coeff,ops = extract_coeffs_and_ops(H_l)

    C = np.zeros((n, 1))

    for i, term in enumerate(ops):
        if term[0][1] == 1 : # b_p'                  
            C[term[0][0]] = coeff[i]
            
    C = np.array(C).reshape(-1, 1)
                
    return np.array(C)

def create_D(C,U,V): 
    return np.array(U.T@C  + V.T@C)

def create_K(L,V):
    
    sum_V = np.sum(V**2, axis=0)
    
    return -(L.T@sum_V)

def create_D_over_L(D,L): 
    
    D = np.array(D)
    
    L = np.array(L)
    
    return D/L

def create_new_K(K,L,D):
    
    D2 = D**2
    
    K_new = K - np.sum(D2/L)
    
    return K_new

# remove the cubic and quartic terms from the Hamiltonian

def H_quadatic(H):
    
    Hv,ops = extract_coeffs_and_ops(H)    

    indices_cq = [i for i, op in enumerate(ops) if len(op) < 3]

    Hvs = list()
    opss = list()

    for i in indices_cq:
        
        Hvs.append(Hv[i])
        opss.append(ops[i])
        
    H_qd = reconstruct_boson_operator(Hvs, opss)
    
    return H_qd


# separte just quadratic terms (b'b , bb, and b'b')

def H_quadatic_only(H):
    
    Hv,ops = extract_coeffs_and_ops(H)
    

    indices_cq = [i for i, op in enumerate(ops) if len(op) == 2]

    Hvs = list()
    opss = list()

    for i in indices_cq:
        
        Hvs.append(Hv[i])
        opss.append(ops[i])
        
    H_qd = reconstruct_boson_operator(Hvs, opss)
    
    return H_qd

# separte just cubic terms 

def H_cubic_only(H):
    
    Hv,ops = extract_coeffs_and_ops(H)
    

    indices_cq = [i for i, op in enumerate(ops) if len(op) == 3]

    Hvs = list()
    opss = list()

    for i in indices_cq:
        
        Hvs.append(Hv[i])
        opss.append(ops[i])
        
    H_cb = reconstruct_boson_operator(Hvs, opss)
    
    return H_cb

# separte just quadratic terms (b'b , bb, and b'b')

def H_quartic_only(H):
    
    Hv,ops = extract_coeffs_and_ops(H)
    

    indices_cq = [i for i, op in enumerate(ops) if len(op) == 4]

    Hvs = list()
    opss = list()

    for i in indices_cq:
        
        Hvs.append(Hv[i])
        opss.append(ops[i])
        
    H_qt = reconstruct_boson_operator(Hvs, opss)
    
    return H_qt

# separte just linear terms (b' and b)

def H_linear_only(H):
    
    Hv,ops = extract_coeffs_and_ops(H)
    
    indices_cq = [i for i, op in enumerate(ops) if len(op) == 1]

    Hvs = list()
    opss = list()

    for i in indices_cq:
        
        Hvs.append(Hv[i])
        opss.append(ops[i])
        
    H_l = reconstruct_boson_operator(Hvs, opss)
    
    return H_l

# separte just constant term ('')

def H_constant_only(H):
    
    Hv,ops = extract_coeffs_and_ops(H)
    
    indices_cq = [i for i, op in enumerate(ops) if len(op) == 0]
    
    Hvs = list()
    opss = list()

    for i in indices_cq:
        
        Hvs.append(Hv[i])
        opss.append(ops[i])
        
    
    return Hvs

#Performing B-Transform on quadratic terms
def b_transform(H,n):
    
    H_q = H_quadatic_only(H)
    
    A,B = create_AB(H_q,n)
    
    U,V,L = solve_uv_lambda(A, B, n)
    
    U,V,L = choose_uv(U,V,L,n)
    
    U,V = scale_matrices(U, V,n)
    
    H_l = H_linear_only(H)
    
    C = create_C(H_l,n)
    
    D = create_D(C,U,V)
    
    K = create_K(L,V)
    
    K_n = create_new_K(K,L,D)
    
    G = create_D_over_L(D,L)
    
    K_c = H_constant_only(H)
    
    K = K_n + K_c
        
    return U,V,L,G,K


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
    
    Ub = bosonic_B(U) # Calculate list of bosonic terms for U c 
    Ubd = bosonic_BD(U) # Calculate list of bosonic terms for U c^dagger
    Vb = bosonic_B(V) # Calculate list of bosonic terms for V c
    Vbd = bosonic_BD(V) # Calculate list of bosonic terms for V c^dagger
    Gi = bosonic_G_inv(G,U,V) # Calculate list of bosonic terms for I G
    
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
def create_diagonal_quadratic(L,n):
    #n = L.shape[0]
    
    H = BosonOperator()
    
    for p in range(n):
        coefficient = L[p]
        # Create the term c_p^† * c_p 
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
 #   print(f"shape full boson {full_matrix.shape}")

    
    # Calculate eigenvalues and eigenvectors using numpy's eigh for dense matrices
    eigenvalues, evec = np.linalg.eigh(full_matrix)
    
    return eigenvalues,evec

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

    U,V,L,G,K = b_transform(Hq,n)

    B_b,B_bd = tranformed_operators_inv(U,V,G)

    Hqr = create_diagonal_quadratic(L.real.flatten(),n) + K.real.flatten()[0] * BosonOperator('')
    
    Hc = H_cubic_only(H)

    Hcr = substitute_operators(Hc, B_b, B_bd)

    H4 = H_quartic_only(H)
    
    H4r = substitute_operators(H4, B_b, B_bd)

    Hr1 = Hqr + Hcr + H4r
       
    return H_real(Hr1)


