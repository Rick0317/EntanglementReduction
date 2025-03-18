from openfermion.ops import BosonOperator
from openfermion.ops import QuadOperator
from openfermion.transforms import normal_ordered
from openfermion.transforms import get_boson_operator

import numpy as np
import os


# Making sure that the coefficients are all real
def H_real(H):
    Hv, ops = extract_coeffs_and_ops(H)
    Hv =[np.real(element) for element in Hv]
    H = reconstruct_boson_operator(Hv, ops)

    return H


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


def bilinear_two_mode_H(params):
    """
    Bi-linear coupling with three modes. Equal coefficient distribution
    over xu yz and zx
    :param params: The coefficients of each term of the Hamiltonian.
    :return:
    """
    Hc = QuadOperator('p0 p0', params[0])
    Hc += QuadOperator('q0 q0', params[0])
    Hc += QuadOperator('p1 p1', params[1])
    Hc += QuadOperator('q1 q1', params[1])
    Hc += QuadOperator('q0 q1', params[2])

    return H_real(normal_ordered(get_boson_operator(Hc)))


def un_coupled_three_mode_H(params):
    """
    Bi-linear coupling with three modes. Equal coefficient distribution
    over xu yz and zx
    :param params: The coefficients of each term of the Hamiltonian.
    :return:
    """
    Hc = QuadOperator('p0 p0', params[0])
    Hc += QuadOperator('q0 q0', params[0])
    Hc += QuadOperator('p1 p1', params[1])
    Hc += QuadOperator('q1 q1', params[1])
    Hc += QuadOperator('p2 p2', params[2])
    Hc += QuadOperator('q2 q2', params[2])

    return H_real(normal_ordered(get_boson_operator(Hc)))


def bilinear_three_mode_H(params):
    """
    Bi-linear coupling with three modes. Equal coefficient distribution
    over xu yz and zx
    :param params: The coefficients of each term of the Hamiltonian.
    :return:
    """
    Hc = QuadOperator('p0 p0', params[0])
    Hc += QuadOperator('q0 q0', params[0])
    Hc += QuadOperator('p1 p1', params[1])
    Hc += QuadOperator('q1 q1', params[1])
    Hc += QuadOperator('p2 p2', params[2])
    Hc += QuadOperator('q2 q2', params[2])
    Hc += QuadOperator('q0 q1', params[3])
    Hc += QuadOperator('q1 q2', params[4])
    Hc += QuadOperator('q0 q2', params[5])

    return H_real(normal_ordered(get_boson_operator(Hc)))


def anharmonic_two_mode_H(params):
    """
    Bi-linear coupling with three modes. Equal coefficient distribution
    over xu yz and zx
    :param params: The coefficients of each term of the Hamiltonian.
    :return:
    """
    Hc = QuadOperator('p0 p0', params[0])
    Hc += QuadOperator('q0 q0', params[0])
    Hc += QuadOperator('p1 p1', params[1])
    Hc += QuadOperator('q1 q1', params[1])
    Hc += QuadOperator('q0 q0 q1 q1', params[2])

    return H_real(normal_ordered(get_boson_operator(Hc)))


def anharmonic_three_mode_H(params):
    """
    Bi-linear coupling with three modes. Equal coefficient distribution
    over xu yz and zx
    :param params: The coefficients of each term of the Hamiltonian.
    :return:
    """
    Hc = QuadOperator('p0 p0', params[0])
    Hc += QuadOperator('q0 q0', params[0])
    Hc += QuadOperator('p1 p1', params[1])
    Hc += QuadOperator('q1 q1', params[1])
    Hc += QuadOperator('p2 p2', params[2])
    Hc += QuadOperator('q2 q2', params[2])
    Hc += QuadOperator('q0 q0 q1 q1', params[3])
    Hc += QuadOperator('q1 q1 q2 q2', params[4])
    Hc += QuadOperator('q0 q0 q2 q2', params[5])

    return H_real(normal_ordered(get_boson_operator(Hc)))

def cube_power_three_mode_H(params):
    """
    Bi-linear coupling with three modes. Equal coefficient distribution
    over xu yz and zx
    :param params: The coefficients of each term of the Hamiltonian.
    :return:
    """
    Hc = QuadOperator('p0 p0', params[0])
    Hc += QuadOperator('q0 q0', params[0])
    Hc += QuadOperator('p1 p1', params[1])
    Hc += QuadOperator('q1 q1', params[1])
    Hc += QuadOperator('p2 p2', params[2])
    Hc += QuadOperator('q2 q2', params[2])
    Hc += QuadOperator('q0 q0 q1', params[3])
    Hc += QuadOperator('q1 q1 q2', params[3])
    Hc += QuadOperator('q2 q2 q0', params[3])

    return H_real(normal_ordered(get_boson_operator(Hc)))

def four_mode_anharmonic_H(params):
    """
    Bi-linear coupling with three modes. Equal coefficient distribution
    over xu yz and zx
    :param params: The coefficients of each term of the Hamiltonian.
    :return:
    """
    Hc = QuadOperator('p0 p0', params[0])
    Hc += QuadOperator('q0 q0', params[0])
    Hc += QuadOperator('p1 p1', params[1])
    Hc += QuadOperator('q1 q1', params[1])
    Hc += QuadOperator('p2 p2', params[2])
    Hc += QuadOperator('q2 q2', params[2])
    Hc += QuadOperator('p3 p3', params[3])
    Hc += QuadOperator('q3 q3', params[3])

    Hc += QuadOperator('q0 q0 q1 q1', params[4])
    Hc += QuadOperator('q1 q1 q2 q2', params[4])
    Hc += QuadOperator('q0 q0 q2 q2', params[4])
    Hc += QuadOperator('q0 q0 q3 q3', params[4])
    Hc += QuadOperator('q1 q1 q3 q3', params[4])
    Hc += QuadOperator('q2 q2 q3 q3', params[4])

    return H_real(normal_ordered(get_boson_operator(Hc)))


def six_mode_H(params):
    """
    Bi-linear coupling with three modes. Equal coefficient distribution
    over xu yz and zx
    :param params: The coefficients of each term of the Hamiltonian.
    :return:
    """
    Hc = QuadOperator('p0 p0', params[0])
    Hc += QuadOperator('q0 q0', params[0])
    Hc += QuadOperator('p1 p1', params[1])
    Hc += QuadOperator('q1 q1', params[1])
    Hc += QuadOperator('p2 p2', params[2])
    Hc += QuadOperator('q2 q2', params[2])

    Hc += QuadOperator('p3 p3', params[3])
    Hc += QuadOperator('q3 q3', params[3])
    Hc += QuadOperator('p4 p4', params[4])
    Hc += QuadOperator('q4 q4', params[4])
    Hc += QuadOperator('p5 p5', params[5])
    Hc += QuadOperator('q5 q5', params[5])

    Hc += QuadOperator('q0 q0 q1 q1', params[6])
    Hc += QuadOperator('q1 q1 q2 q2', params[6])
    Hc += QuadOperator('q0 q0 q2 q2', params[6])

    Hc += QuadOperator('q0 q0 q3 q3', params[6])
    Hc += QuadOperator('q1 q1 q4 q4', params[6])
    Hc += QuadOperator('q0 q0 q4 q4', params[6])

    Hc += QuadOperator('q0 q0 q5 q5', params[6])
    Hc += QuadOperator('q1 q1 q3 q3', params[6])
    Hc += QuadOperator('q1 q1 q5 q5', params[6])

    Hc += QuadOperator('q2 q2 q3 q3', params[6])
    Hc += QuadOperator('q2 q2 q4 q4', params[6])
    Hc += QuadOperator('q2 q2 q5 q5', params[6])

    Hc += QuadOperator('q3 q3 q4 q4', params[6])
    Hc += QuadOperator('q3 q3 q5 q5', params[6])
    Hc += QuadOperator('q4 q4 q5 q5', params[6])

    return H_real(normal_ordered(get_boson_operator(Hc)))


if __name__ == '__main__':
    truncation = 6
    h_variables = [1 / 2, 1 / 2, 1/ 2]
    bilinearH = un_coupled_three_mode_H(h_variables)
    print(bilinearH)

    print(bilinearH)
