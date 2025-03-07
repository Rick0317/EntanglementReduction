from utils.util_gfro import get_boson_eigenspectrum_full
from utils.util_hamil import bilinear_three_mode_H
from utils.util_tensornetwork import get_exact_bd_mps

if __name__ == "__main__":
    truncation = 6
    h_variables = [1 / 2, 1 / 2, 1 / 2, 0.4]
    bilinearH = bilinear_three_mode_H(h_variables)
    eig_values, eig_vecs = get_boson_eigenspectrum_full(bilinearH, truncation)

    ground_state = eig_vecs[0]

    reshaped_gs = ground_state.reshape(6, 6, 6)

    exact_bd = get_exact_bd_mps(reshaped_gs)

    print(f"Exact BD: {exact_bd}")
