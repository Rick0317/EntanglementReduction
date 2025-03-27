import numpy as np
from RickVersion.utils.ham_utils import get_H_op, get_H_ground_state, get_Hmol_params, get_H_ground_state_bosonic, truncate_quad_operator
from RickVersion.utils.bog_utils import gen_initial_X_vector, get_Smat_from_X, mode_swap
from RickVersion.utils.util_tensornetwork import get_approx_bd_mps, get_mps
from RickVersion.utils.opt_utils import optimize_bog_transfrom, modified_H_params, weighted_sum_MI
from RickVersion.utils.util_mutualinfo import mutual_info_full_n_sites, mutual_info_full
from RickVersion.Graph_plots.util_visualizer import (visualize_mutual_information,
                                   visualize_minimization_steps)
from RickVersion.utils.util_gfro import quad_diagonalization
from openfermion import get_boson_operator, normal_ordered

if __name__ == '__main__':
    au_to_cm = 219474.63068
    mol_list = ['H2S', 'CO2', 'CH2O']
    nmax_list = [7, 5, 5]
    method = "COBYLA"
    for idx, mol in enumerate(mol_list):
        #Read in molecular Hamiltonian
        n_modes, H_params = get_Hmol_params(mol)
        nmax = nmax_list[idx]
        error_threshold_frac = 1e-4

        #Choose cost function:
        cost_func = 'MI'  #Options are 'MI', 'energy'
        #Choose whether or not to perform squeezing:
        squeezing = True

        #Get H as open fermion quad-operator
        H = get_H_op(H_params)

        #Get H ground state eigenvector and eigenvalue

        ed_ground_state_energy, ed_ground_state = get_H_ground_state(H, nmax)
        reshaped_gs = ed_ground_state.reshape(*(nmax,) * n_modes)

        MI_list_original = mutual_info_full_n_sites(reshaped_gs, nmax, n_modes)

        error_threshold = abs(ed_ground_state_energy * error_threshold_frac)
        exact_bd, data_original = get_approx_bd_mps(reshaped_gs, threshold= error_threshold)

        print(f"Almost Exact BD for error thershold {error_threshold_frac}: {exact_bd}")
        print(f"Ground state energy in cm: {ed_ground_state_energy * au_to_cm}")

        #Get initial guess for optimization vector X
        X = gen_initial_X_vector(n_modes, squeezing = squeezing, initial = 'id')

        #Select options for optimization
        options = {'maxiter': 100, 'disp': False} #, 'gtol': 1e-20}
        bond_dimension = 2

        res, intermediate_data = optimize_bog_transfrom(X, H_params, nmax, bond_dimension, n_modes, options, cost_function=cost_func, tol=1e-100, squeezing=squeezing, mol=mol)
        S = get_Smat_from_X(n_modes, res.x, squeezing=squeezing)
        Ht_params = modified_H_params(H_params, S)

        #Get Ht as open fermion operator
        Ht = get_H_op(Ht_params)

        #Get Ht ground state eigenvector and eigenvalue
        gst_energy, gst = get_H_ground_state(Ht, nmax)

        reshaped_gst = gst.reshape(*(nmax,) * n_modes)

        vmax = intermediate_data[0]

        visualize_mutual_information(MI_list_original, n_modes, vmax, file_name=f"{mol}_{cost_func}_{squeezing}_original.png")
        MI_list = mutual_info_full_n_sites(reshaped_gst, nmax, n_modes)
        visualize_mutual_information(MI_list, n_modes, vmax, file_name=f"{mol}_{cost_func}_{squeezing}_optimized.png")

        bd_optim, data_optim = get_approx_bd_mps(reshaped_gst, threshold = error_threshold)

        energy_change = abs(gst_energy - ed_ground_state_energy)

        filter_threshold = max(intermediate_data)
        visualize_minimization_steps(filter_threshold, intermediate_data)

        print(f"Almost exact BD for error thershold {error_threshold_frac} after optimization: {bd_optim}")
        print(f"Ground state energy after optimization: {gst_energy * au_to_cm}")
        print(f"Ground state energy change: {energy_change * au_to_cm}")

        file_name = 'results.txt'
        with open(file_name, 'a') as f:
            f.write(
                f'\n\n--- New Entry ---\n')  # Optional separator for clarity
            f.write(f'Molecule: {mol}\n')
            f.write(f'Cost Function: {cost_func}\n')
            f.write(f'Squeezing: {squeezing}\n')
            f.write(f'Truncation in the optimization: {nmax}\n')
            f.write(f'Optimization option: {options}\n')
            f.write(f'Optimization method: {method}\n')
            f.write(f'Error threshold fraction: {error_threshold_frac}\n')
            f.write(f'Approximate bond dimension: {bond_dimension}\n')
            f.write(f'Exact Bond Dimension: {exact_bd}\n')
            f.write(f'Optimized Bond Dimension: {bd_optim}\n')
            f.write(f'Original Cost fn Value Data: {data_original}\n')
            f.write(f'Optimized Cost fn Value Data: {data_optim}\n')
            f.write(f'Ground state energy original: {ed_ground_state_energy}\n')
            f.write(f'Ground state energy optimized: {gst_energy}\n')

        print(f'Results appended to {file_name}')



