import numpy as np
import tensorly as tl
from tensorly.decomposition import tensor_train
from tensorly.tt_tensor import tt_to_tensor

def get_mps(tensor, rank):
    tt_cores = tensor_train(tensor, rank=rank)

    # Convert TT cores back to the full tensor
    reconstructed_tensor = tt_to_tensor(tt_cores)

    # Check the reconstruction error
    error = tl.norm(tensor - reconstructed_tensor)

    return error, tt_cores


def get_approx_tensor(tensor, rank):
    tt_cores = tensor_train(tensor, rank=rank)

    reconstructed_tensor = tt_to_tensor(tt_cores)

    return reconstructed_tensor


def get_exact_bd_mps(tensor):
    """
    Get the exact bond dimension for the given tensor
    :param tensor:
    :return:
    """
    bd = 1
    error, tt_cores = get_mps(tensor, rank=bd)
    while error > 1e-10:
        bd += 1
        print(f"Bond dimension: {bd}")
        error, tt_cores = get_mps(tensor, rank=bd)

    return bd


def get_approx_bd_mps(tensor, threshold):
    """
    Get the approximate bond dimension for the given tensor
    :param tensor:
    :param threshold:
    :return:
    """
    bd = 1
    error, tt_cores = get_mps(tensor, rank=bd)
    while error > threshold:
        bd += 1
        print(f"Bond dimension: {bd}")
        error, tt_cores = get_mps(tensor, rank=bd)

    return bd
