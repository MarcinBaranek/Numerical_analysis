import numpy as np
import math


def init_identity_matrix(dim: int):
    identity_matrix = np.zeros(shape=(dim, dim))
    for i in range(dim):
        identity_matrix[i, i] = 1
    return identity_matrix


def householder_transformation_matrix(vector: np.ndarray, new_vector: np.ndarray):
    if not (isinstance(vector, np.ndarray) and isinstance(new_vector, np.ndarray)):
        print("Error in householder_transformation: input vector is not a numpy.ndarray object")
        return False
    vector = np.reshape(vector, newshape=(-1, 1))
    new_vector = np.reshape(new_vector, newshape=(-1, 1))
    if vector.shape != new_vector.shape:
        print("Error in householder_transformation: wrong shape of inputs vectors")
        return False
    dimensional = vector.shape[0]
    cos_of_angel_between_vector_and_new_vector = np.dot(np.transpose(vector), new_vector)
    #for numerical stability we chose a vector of more norm
    if cos_of_angel_between_vector_and_new_vector >= 0:
        householder_vector = vector + math.sqrt((vector**2).sum()) * new_vector
    else:
        householder_vector = vector - math.sqrt((vector ** 2).sum()) * new_vector
    #compute the specular reflection
    specular_reflection = 2 / np.dot(np.transpose(householder_vector), householder_vector)
    #create identity matrix
    identity_matrix = init_identity_matrix(dimensional)
    #create householder matrix
    householder_matrix = specular_reflection * np.dot(householder_vector, np.transpose(householder_vector))
    householder_matrix = identity_matrix - householder_matrix
    return householder_matrix


def improvement_r_matrix(matrix: np.ndarray, iteration: int):
    j = 1
    for i in range(iteration):
        matrix[j:, i] = np.zeros(shape=matrix[i, j:].shape)
        j += 1
    return matrix


def householder_algorithm(matrix: np.ndarray):
    if not isinstance(matrix, np.ndarray):
        print("Error in householder_algorithm: input matrix is not a numpy.ndarray object")
        return False
    first_dimensional = matrix.shape[0]
    # create identity matrix
    identity_matrix = init_identity_matrix(first_dimensional)
    # create P, Q and R matrix
    # Q matrix will be transpose be for end of function
    p_matrix = identity_matrix
    q_matrix = np.zeros(shape=p_matrix.shape)
    r_matrix = matrix
    # create unit vector
    unit_vector = np.zeros(shape=(first_dimensional, 1)).astype(int)
    unit_vector[0, 0] = 1
    for i in range(first_dimensional):
        householder_matrix = householder_transformation_matrix(r_matrix[i:, i], unit_vector)
        # preparing unit vector to next iteration
        unit_vector = np.reshape(np.delete(unit_vector, -1), newshape=(-1, 1))
        # update P matrix
        p_matrix[i:, i:] = householder_matrix
        if i == 0:
            q_matrix = p_matrix
        r_matrix = np.dot(p_matrix, r_matrix)
        r_matrix = improvement_r_matrix(r_matrix, i)
        if i != 0:
            q_matrix = np.dot(p_matrix, q_matrix)
        p_matrix = init_identity_matrix(first_dimensional)
        q_matrix = np.transpose(q_matrix)
    return q_matrix, r_matrix
