import numpy as np
import math
import time


def householder_transformation_matrix(vector: np.ndarray, new_vector: np.ndarray,
                                      time_of_execution=False) -> np.ndarray:

    print(f"compute transformation matrix for:             {vector}\n"
          f"in new direction:                              {new_vector}")
    if time_of_execution:
        now = time.time()

    vector = np.reshape(vector, newshape=(-1, 1))
    new_vector = np.reshape(new_vector, newshape=(-1, 1))

    dimensional = vector.shape[0]

    cos_of_angel_between_vector_and_new_vector = np.dot(np.transpose(vector), new_vector)
    # for numerical stability we chose a vector of more norm
    if cos_of_angel_between_vector_and_new_vector > 0:
        householder_vector = vector + math.sqrt((vector ** 2).sum()) * new_vector
    else:
        householder_vector = vector - math.sqrt((vector ** 2).sum()) * new_vector

    specular_reflection = np.dot(np.transpose(householder_vector), householder_vector)
    if specular_reflection != 0:
        specular_reflection = 2 / specular_reflection
    else:
        #vector and direction vector have the same direction so return identity matrix
        if time_of_execution:
            print(f"time of execution householder_transformation_matrix is: {time.time() - now} s")
        return np.eye(dimensional)

    identity_matrix = np.eye(dimensional)

    #create householder matrix
    householder_matrix = specular_reflection * np.dot(householder_vector, np.transpose(householder_vector))
    householder_matrix = identity_matrix - householder_matrix

    if time_of_execution:
        print(f"time of execution householder_transformation_matrix is: {time.time() - now} s")

    print(f"transformation matrix:\n"
          f"{householder_matrix}")

    return householder_matrix


#for numerical stability
#inserts 0 where they sure are (for upper triangular matrix)
def improvement_r_matrix(matrix: np.ndarray, iteration: int) -> np.ndarray:
    j = 1
    for i in range(iteration):
        matrix[j:, i] = np.zeros(shape=matrix[j:, i].shape)
        j += 1
    return matrix


#for numerical stability
def improvement_matrix(matrix: np.ndarray, epsilon=1.e-7):
    matrix = matrix.astype(dtype=float)
    location = np.where(abs(matrix - 0) < epsilon)
    matrix[location] = epsilon
    return matrix


def householder_algorithm(matrix: np.ndarray,
                          improve_matrix=True,
                          improve_r_matrix=True,
                          time_of_execution=False) -> (np.ndarray, np.ndarray):

    print(f"compute matrix decomposition for\n"
          f"{matrix}")

    if time_of_execution:
        now = time.time()

    if improve_matrix:
        matrix = improvement_matrix(matrix)

    first_dimensional = matrix.shape[0]

    if first_dimensional > matrix.shape[1]:
        print("The matrix has the first dimension greater than the second,\n"
              "therefore the algorithm was run for the transposed matrix")
        first_dimensional = matrix.shape[1]
        matrix = np.transpose(matrix)

    identity_matrix = np.eye(first_dimensional)

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

        if not householder_matrix.all():
            print("something went wrong")
            print("check the assumptions")
            return False, False

        # preparing unit vector to next iteration
        if i < first_dimensional - 1:
            unit_vector = np.reshape(np.delete(unit_vector, -1), newshape=(-1, 1))

        # update P matrix
        if i < first_dimensional - 1:
            p_matrix[i:, i:] = householder_matrix

        print(f"current P matrix:\n"
              f"{p_matrix}")

        # update R matrix
        r_matrix = np.dot(p_matrix, r_matrix)
        if improve_r_matrix:
            r_matrix = improvement_r_matrix(r_matrix, i)

        print(f"current R matrix:\n"
              f"{r_matrix}")

        # update Q matrix
        if i == 0:
            q_matrix = p_matrix
        else:
            q_matrix = np.dot(p_matrix, q_matrix)

        print(f"current Q matrix:\n"
              f"{q_matrix}")

        # preparing P matrix to next iteration
        p_matrix = np.eye(first_dimensional)

    q_matrix = np.transpose(q_matrix)

    if time_of_execution:
        print(f"time of execution householder_algorithm is: {time.time() - now} s")

    print(f"function return Q matrix:\n"
          f"{q_matrix}\n"
          f"and R matrix:\n"
          f"{r_matrix}")

    return q_matrix, r_matrix


def linear_equations_with_householder_algorithm(matrix: np.ndarray,
                                                bias: np.ndarray,
                                                time_of_execution=False) -> np.ndarray:

    print(f"compute linear equations with matrix:\n"
          f"{matrix}\n"
          f"with bias:                                      {bias}")

    if time_of_execution:
        now = time.time()

    first_dimensional = matrix.shape[0]

    if matrix.shape[0] != matrix.shape[1]:
        print("Error in linear_equations_with_householder_algorithm:\n"
              "The input matrix has wrong shape")
        return np.array(False)

    bias = np.reshape(bias, newshape=(-1, 1))

    if bias.shape[0] != first_dimensional:
        print("Error in linear_equations_with_householder_algorithm:\n"
              "the bias has the wrong shape")
        return np.array(False)

    q_matrix, r_matrix = householder_algorithm(matrix, improve_matrix=True, improve_r_matrix=True)
    bias = np.dot(np.transpose(q_matrix), bias)
    solve = np.array(False)

    for i in range(1, first_dimensional + 1, 1):
        if r_matrix[-i, - i] == 0:
            print("Error in linear_equations_with_householder_algorithm:\n"
                  "the system is contradictory")
            return np.array(False)

        if i == 1:
            solve = np.array(bias[-i, 0] / r_matrix[-i, -i])
        else:
            next_solve_value = (bias[-i, 0] + np.dot(r_matrix[- i, 1 - i:], solve)) / r_matrix[- i, - i]
            solve = np.concatenate((np.reshape(next_solve_value, newshape=(-1, 1)),
                                    np.reshape(solve, newshape=(-1, 1))), axis=0)
            solve = np.reshape(solve, newshape=(-1, 1))

        print(f"current solve:                              {solve}")

    if time_of_execution:
        print(f"time of execution linear_equations_with_householder_algorithm is: {time.time() - now} s")

    return solve
  
def mserorr(matrix_1: np.ndarray, matrix_1: np.ndarray):
    return ((matrix_1-matrix_2)**2).sum()


if __name__ == '__main__':
    a = np.array([[12, -51, 4], [6, 167, -68], [-4, 24, -41]])
    b = np.array([12, 6, -4])
    linear_equations_with_householder_algorithm(a, b)
