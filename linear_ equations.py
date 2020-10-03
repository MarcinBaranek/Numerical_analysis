from householder_methodes import householder_algorithm
import numpy as np

def linear_equations_with_householder_algorithm(matrix: np.ndarray, bias: np.ndarray) -> np.ndarray:
    dim = matrix.shape[0]
    bias = np.reshape(bias, newshape=(dim, 1))
    q_matrix, r_matrix = householder_algorithm(matrix)
    bias = np.dot(np.transpose(q_matrix), bias)
    solve = np.array(bias[-1, 0] / r_matrix[-1, -1])
    solve = np.reshape(solve, newshape=(-1, 1))
    for i in range(1, dim, 1):
        next_solve_value = (bias[dim - i - 1, 0] - np.dot(r_matrix[dim - i - 1, dim - i:],
                                                          solve))\
                           / r_matrix[dim - i - 1, dim - i - 1]
        solve = np.append(next_solve_value, solve)
        solve = np.reshape(solve, newshape=(-1, 1))
    return solve
    
