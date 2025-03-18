from openfermion.ops import BosonOperator, QuadOperator
from openfermion.transforms import normal_ordered, get_boson_operator


import numpy as np
import os


# Creating bosonic operators from file_name, divisor(coefficients of Taylors series), include_p_terms for quadratic
def process_file(file_name, divisor, include_p_terms=False):
    with open(file_name, 'r') as file:
        lines = file.readlines()
        
    operators = []
    for line in lines:
        parts = line.strip().split()
        indices = parts[:-1]
        value = float(parts[-1]) / divisor
        indices_str_q = ' '.join(f'q{int(index)-1}' for index in indices)
        operator_q = QuadOperator(indices_str_q, value)
        operators.append(operator_q)
        
        if include_p_terms:
            indices_str_p = ' '.join(f'p{int(index)-1}' for index in indices)
            operator_p = QuadOperator(indices_str_p, value)
            operators.append(operator_p)
        
    return operators

# Making sure that the coefficients are all real
def H_real(H):
    Hv,ops = extract_coeffs_and_ops(H)
    Hv =[np.real(element) for element in Hv]
    H = reconstruct_boson_operator(Hv, ops)
    
    return H

# Take molecule name and extract files from the folder named molecule and output bosonic Hamiltonian
def operator_quad(molecule):
    base_dir = 'molecules'
    files = [f'f2{molecule}.dat', f'f3{molecule}.dat', f'f4{molecule}.dat']
    divisors = {'f2': 1.0, 'f3': 1.0, 'f4': 1.0}
    
    H = QuadOperator()
    
    for file in files:
        file_path = os.path.join(base_dir, file)
        file_prefix = file[:2]
        if file_prefix in divisors:
            divisor = divisors[file_prefix]
            include_p_terms = (file_prefix == 'f2')
            operators = process_file(file_path, divisor, include_p_terms)
            for operator in operators:
                H += operator
                
    H_b = get_boson_operator(H)
    H = normal_ordered(H_b)
    
    return H_real(H)


# Function to extract coefficients and operators
def extract_coeffs_and_ops(boson_operator):
    coeffs = []
    ops = []
    for term, coeff in boson_operator.terms.items():
        ops.append(term)
        coeffs.append(coeff)
    return coeffs, ops

# Reconstruct bosonic operator from coefficients and operators
def reconstruct_boson_operator(coeffs, ops):
    boson_operator = BosonOperator()
    for coeff, op in zip(coeffs, ops):
        boson_operator += BosonOperator(op, coeff)
    return boson_operator

def H_real1(H):
    Hv,ops = extract_coeffs_and_ops(H)
    Hv =[np.real(element) for element in Hv]
    H = reconstruct_boson_operator(Hv, ops)
    
    return H



###########################################################################################
## The author of the following part is Ignacio Loaiza Ganem
# =============================================================================
# import h5py 
# from bosonic_form import harmonic_oscillators, taylor_to_bosonic
# 
# def hamil(name):
#     with h5py.File(f'data/{name}.hdf5', 'r') as f:
#         taylor_1D = f["taylor_1D"][()]
#         taylor_2D = f["taylor_2D"][()]
#         taylor_3D = f["taylor_3D"][()]
#         freqs = f["freqs"][()]
# 
#     Hharm = harmonic_oscillators(freqs)
#     Hanh = taylor_to_bosonic([taylor_1D, taylor_2D, taylor_3D])
# 
#     bosonic_hamiltonian = Hanh + Hharm
#     
#     return   219474.63068 * normal_ordered(bosonic_hamiltonian)
# =============================================================================
###########################################################################################


## Test Hamiltonians

## Test hamiltonian 3d
def test0(v):
    Hc = QuadOperator('p0 p0', v[0])
    Hc += QuadOperator('q0 q0', v[0])
    Hc += QuadOperator('p1 p1', v[1])
    Hc += QuadOperator('q1 q1', v[1])
    Hc += QuadOperator('q0 q0 q1', v[2])
    Hc += QuadOperator('q1 q1 q1', v[3])
    
    return H_real(normal_ordered(get_boson_operator(Hc)))

## Test hamiltonian 3d
def test(v):
    Hc = QuadOperator('p0 p0', v[0])
    Hc += QuadOperator('q0 q0', v[0])
    Hc += QuadOperator('p1 p1', v[1])
    Hc += QuadOperator('q1 q1', v[1])
    Hc += QuadOperator('p2 p2', v[2])
    Hc += QuadOperator('q2 q2', v[2])
    Hc += QuadOperator('q0 q0 q2', v[3])
    Hc += QuadOperator('q1 q1 q2', v[4])
    Hc += QuadOperator('q0 q1 q2', v[5])
    
    return H_real(normal_ordered(get_boson_operator(Hc)))


## Test hamiltonian 3d
def test1(v):
    Hc = QuadOperator('p0 p0', v[0])
    Hc += QuadOperator('q0 q0', v[0])
    Hc += QuadOperator('p1 p1', v[1])
    Hc += QuadOperator('q1 q1', v[1])
    Hc += QuadOperator('p2 p2', v[2])
    Hc += QuadOperator('q2 q2', v[2])
    Hc += QuadOperator('q0 q1', v[3])
    Hc += QuadOperator('q1 q2', v[4])
    Hc += QuadOperator('q0 q2', v[5])
    
    return H_real(normal_ordered(get_boson_operator(Hc)))


## Test hamiltonian 3d
def test2(v):
    Hc = QuadOperator('p0 p0', v[0])
    Hc += QuadOperator('q0 q0', v[0])
    Hc += QuadOperator('p1 p1', v[1])
    Hc += QuadOperator('q1 q1', v[1])
    Hc += QuadOperator('p2 p2', v[2])
    Hc += QuadOperator('q2 q2', v[2])
    Hc += QuadOperator('q0 q0 q1 q1', v[3])
    Hc += QuadOperator('q1 q1 q2 q2', v[4])
    Hc += QuadOperator('q0 q0 q2 q2', v[5])
    
    
    return H_real(normal_ordered(get_boson_operator(Hc)))

## Test hamiltonian4d
def test4(v):
    Hc = QuadOperator('p0 p0', v[0])
    Hc += QuadOperator('q0 q0', v[0])
    Hc += QuadOperator('p1 p1', v[1])
    Hc += QuadOperator('q1 q1', v[1])
    Hc += QuadOperator('p2 p2', v[2])
    Hc += QuadOperator('q2 q2', v[2])
    Hc += QuadOperator('p3 p3', v[3])
    Hc += QuadOperator('q3 q3', v[3])
    Hc += QuadOperator('q0 q0 q3', v[4])
    Hc += QuadOperator('q1 q1 q3', v[5])
    Hc += QuadOperator('q0 q1 q3', v[6])
    Hc += QuadOperator('q2 q2 q3', v[7])
    
    return H_real(normal_ordered(get_boson_operator(Hc)))

## Test hamiltonian 3d
def test5(v):
    Hc = QuadOperator('p0 p0', v[0])
    Hc += QuadOperator('q0 q0', v[0])
    Hc += QuadOperator('p1 p1', v[1])
    Hc += QuadOperator('q1 q1', v[1])
    Hc += QuadOperator('p2 p2', v[2])
    Hc += QuadOperator('q2 q2', v[2])
    Hc += QuadOperator('q0 q0 q1', v[3])
    Hc += QuadOperator('q0 q0 q2', v[3])
    Hc += QuadOperator('q1 q1 q0', v[4])
    Hc += QuadOperator('q1 q1 q2', v[4])
    Hc += QuadOperator('q2 q2 q0', v[3])
    Hc += QuadOperator('q2 q2 q1', v[4])
    
    return H_real(normal_ordered(get_boson_operator(Hc)))

## Test hamiltonian 3d
def test6(v):
    Hc = QuadOperator('p0 p0', v[0])
    Hc += QuadOperator('q0 q0', v[0])
    Hc += QuadOperator('p1 p1', v[1])
    Hc += QuadOperator('q1 q1', v[1])
    Hc += QuadOperator('p2 p2', v[2])
    Hc += QuadOperator('q2 q2', v[2])
    Hc += QuadOperator('q0 q1', v[3])
    Hc += QuadOperator('q1 q2', v[4])
    Hc += QuadOperator('q0 q2', v[5])
    Hc += QuadOperator('q0 q0 q1', v[8])
    Hc += QuadOperator('q0 q0 q2', v[8])
    Hc += QuadOperator('q1 q1 q0 q2', v[9])
    Hc += QuadOperator('q0', v[6])
    Hc += QuadOperator('q1', v[7])
    
    return H_real(normal_ordered(get_boson_operator(Hc)))


## Test hamiltonian 3d
def test7(v):
    Hc = QuadOperator('p0 p0', v[0])
    Hc += QuadOperator('q0 q0', v[0])
    Hc += QuadOperator('p1 p1', v[1])
    Hc += QuadOperator('q1 q1', v[1])
    Hc += QuadOperator('p2 p2', v[2])
    Hc += QuadOperator('q2 q2', v[2])
    Hc += QuadOperator('q0 q1', v[3])
    Hc += QuadOperator('q1 q2', v[4])
    Hc += QuadOperator('q0 q2', v[5])
    Hc += QuadOperator('q0 q0 q1', v[3])
    Hc += QuadOperator('q0 q0 q2', v[5])
    Hc += QuadOperator('q1 q1 q0', v[4])
    Hc += QuadOperator('q1 q1 q2', v[4])
    Hc += QuadOperator('q2 q2 q0', v[3])
    Hc += QuadOperator('q2 q2 q1', v[4])
    Hc += QuadOperator('q0 q0 q1 q1', v[3])
    Hc += QuadOperator('q1 q1 q2 q2', v[4])
    Hc += QuadOperator('q0 q0 q2 q2', v[5])
    
    return H_real(normal_ordered(get_boson_operator(Hc)))