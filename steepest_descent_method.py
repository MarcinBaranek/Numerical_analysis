import numpy as np


def l2_norm(vector):
    """
    compute l2 norm of vector
    :param vector: numpy.array with shape (n,)
    :return: float scalar, l2 norm of vector
    """
    return (vector**2).sum()


def compute_quadratic_form(matrix, vector):
    """
    Compute quadratic form: x^t A x
    :param matrix: numpy.array with shape (n, n)
    :param vector: numpy.array with shape (n,)
    :return: scalar with type as arguments
    """
    return np.dot(np.transpose(vector), np.dot(matrix, vector))


def compute_residual(matrix, point, bias):
    """
    Compute residual: Ax-b
    :param matrix: numpy.array with shape (n, n)
    :param point: numpy.array with shape (n,)
    :param bias: numpy.array with shape (n,)
    :return: numpy.array with shape (n,)
    """
    return np.dot(matrix, point) - bias


def compute_learning_rate(matrix, residual):
    """
    compute learning rate: (r^t r) / (r^t A r), for steepest_descent_method
    :param matrix: numpy.array with shape (n, n), matrix of system
    :param residual: numpy.array with shape (n,), residual of point
    :return: float scalar, learning rate for steepest_descent_method
    """
    return (residual**2).sum() / compute_quadratic_form(matrix, residual)


def step_of_steepest_descent_method(matrix, point, residual):
    """
    one iteration of steepest descent method
    :param matrix: numpy.array with shape (n, n)
    :param point: numpy.array with shape (n,)
    :param residual: numpy.array with shape (n,)
    :return: numpy.array with shape (n,), is a new point
    """
    lr = compute_learning_rate(matrix, residual)
    return point - lr * residual


def steepest_descent_method(matrix, bias, start_point=None, tolerance=1.e-8, max_iter=100, norm=l2_norm):
    """
    approximates the solution of Ax=b with a positive symmetric matrix.
    The algorithm ends when the Ax-b error in the L2 norm is at the tolerance level
    or after performing the maximum number of iterations.
    :param matrix: numpy.array with shape (n, n), is a matrix of system (A)
    :param bias: numpy.array with shape (n,), is a bias of system (b)
    :param start_point: numpy.array with shape (n,), is a start point.
        The method is convergent regardless of the choice of the starting point
        as long as the assumptions are met.
    :param tolerance: scalar, is a tolerance level
    :param max_iter: int, is a maximum number of iteration
    :param norm: norm function
    :return: numpy.array with shape (n,), is a approximation of solution the system Ax=b
    """
    if not start_point:
        point = np.zeros(shape=bias.shape)
    else:
        point = start_point
    residual = compute_residual(matrix, point, bias)
    iterator = 0
    while norm(residual) > tolerance and iterator < max_iter:
        point = step_of_steepest_descent_method(matrix, point, residual)
        residual = compute_residual(matrix, point, bias)
        iterator += 1
    return point


A = np.array([[1, 3], [3, 5]])
b = np.array([7, 13])
print(steepest_descent_method(A, b))
