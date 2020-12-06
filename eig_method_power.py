import numpy as np


def eig_value_power_method(matrix: np.array, init_vector=None, max_iter=None):
    if max_iter is None:
        max_iter = 5 * matrix.shape[0]
    if init_vector is None:
        init_vector = np.ones(shape=(matrix.shape[0],))

    for i in range(max_iter):
        init_vector = np.dot(matrix, init_vector) / np.sqrt(np.sum(init_vector ** 2))

    index = np.unravel_index(np.argmax(np.dot(matrix, init_vector), axis=None), init_vector.shape)
    return np.dot(matrix, init_vector)[index] / init_vector[index], init_vector / np.sqrt(np.sum(init_vector ** 2))
