import numpy as np
import math


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
    specular_reflection = 0.5 * np.dot(np.transpose(householder_vector), householder_vector)
    #create identity matrix
    identity_matrix = np.zeros(shape=(dimensional, dimensional))
    for i in range(dimensional):
        identity_matrix[i, i] = 1
    #create householder matrix
    householder_matrix = specular_reflection * np.dot(householder_vector, np.transpose(householder_vector))
    householder_matrix = identity_matrix - householder_matrix
    return householder_matrix
