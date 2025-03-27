import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def visualize_mutual_information(MI, n, vmax, file_name='mutual_information_heatmap.png'):
    """
    Visualize the mutual information between n modes.
    Normalized

    :param MI: List of mutual information values in upper triangular order.
    :param n: Number of modes.
    """
    # Initialize an empty mutual information matrix
    mutual_info_matrix = np.zeros((n, n))

    # Fill the upper triangular part of the matrix
    index = 0
    for i in range(n):
        for j in range(i + 1, n):
            mutual_info_matrix[i, j] = MI[index]
            mutual_info_matrix[j, i] = MI[index]  # Symmetric assignment
            index += 1

    # Plot the heatmap for the normalized matrix
    sns.heatmap(mutual_info_matrix, annot=True, fmt=".5f", cmap="Blues", vmin=0,
                vmax=vmax,
                xticklabels=[str(i + 1) for i in range(n)],
                yticklabels=[str(i + 1) for i in range(n)])

    plt.savefig(file_name, format='png')
    print(f"Plot saved as {file_name}")

    plt.show()


def visualize_minimization_steps(filter_threshold, intermediate_values):
    """
    Visualize the minimization steps.
    Parameters
    ----------
    filter_threshold
    intermediate_values

    Returns
    -------

    """
    # Create an index list (0, 1, 2, ...)
    filtered_values = [val if val < filter_threshold else np.nan for val in
                       intermediate_values]

    # Create an index list
    indices = range(len(filtered_values))

    # Plot
    plt.plot(indices, filtered_values, marker='o', linestyle='-')

    # Labels and title
    plt.xlabel("Index")
    plt.ylabel("Intermediate Values (Filtered)")
    plt.title("Plot with Outliers Removed")

    # Show the plot
    plt.show()
