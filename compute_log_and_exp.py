import matplotlib.pyplot as plt


""" Hyper parameters """

argument_to_log = 0.5
argument_to_exp = 1
list_with_precisions = [0, 2.2250738585072014e-308, 1.175494351e-38, 2 ** (-64), 2 ** (-56), 2 ** (-55), 2 ** (-32)]
list_with_iterations = [5, 10, 15, 20, 40, 60, 80, 100]
iteration_grid_to_plots = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

""" End Hyper parameters"""


def sum_version_count_log(x=argument_to_log):
    counter = 1
    component = x ** counter / counter
    last_component = 1

    while last_component + component != last_component:
        counter += 1
        component = x ** counter / counter

    return counter


def total_sum_version_log(x=argument_to_log):
    component = 1
    counter = 1
    total_sum = 0

    while total_sum + component != total_sum:
        counter += 1
        total_sum += component
        component = x ** counter / counter

    return counter


def count_essential_components_log(x, precision=2.2250738585072014e-308):
    counter = 1
    component = x ** counter / counter

    while component > precision:
        counter += 1
        component = x ** counter / counter
    return counter


def factorial(n: int):
    if n > 1:
        return n * factorial(n - 1)
    else:
        return 1


def count_essential_components_exp(x, precision=2.2250738585072014e-308):
    counter = 1
    component = x ** counter / factorial(counter)

    while component > precision:
        counter += 1
        component = x ** counter / factorial(counter)

    return counter


def sum_version_count_exp(x=argument_to_log):
    counter = 1
    component = x ** counter / factorial(counter)
    last_component = 1

    while last_component + component != last_component:
        counter += 1
        component = x ** counter / factorial(counter)

    return counter


def total_sum_version_exp(x=argument_to_exp):
    component = 1
    counter = 1
    total_sum = 0

    while total_sum + component != total_sum:
        counter += 1
        total_sum += component
        component = x ** counter / factorial(counter)

    return counter


def compute_log(x, precision=2.2250738585072014e-308, number_of_components=None):
    if number_of_components is None:
        iterator = count_essential_components_exp(x, precision)
    else:
        iterator = number_of_components

    results_plus = 0
    results_minus = 0

    while iterator > 0:

        if not iterator % 2:
            results_minus += (x ** iterator) / iterator

        else:
            results_plus += (x ** iterator) / iterator

        iterator += -1

    results = results_plus - results_minus

    return results


def compute_exp(x, precision=2.2250738585072014e-308, number_of_components=None):
    if number_of_components is None:
        iterator = count_essential_components_exp(x, precision)
    else:
        iterator = number_of_components

    results = x ** iterator / factorial(iterator)

    while iterator > 0:
        results += x ** iterator / factorial(iterator)
        iterator += -1

    return results


def print_results(log_arg=argument_to_log, exp_arg=argument_to_exp, precision_array=None, numbers_of_components=None):
    if precision_array is None:
        #enter interesting precision in the array
        precision_array = list_with_precisions

    if numbers_of_components is None:
        numbers_of_components = list_with_iterations

    for precision in precision_array:
        print(f'For precision =               {precision}\n'
              f'needed components to log:     {count_essential_components_log(log_arg, precision=precision)}\n'
              f'needed components to exp:     {count_essential_components_exp(exp_arg, precision=precision)}\n'
              f'log({1 + log_arg}) is equal:            {"{:.56f}".format(compute_log(log_arg, precision=precision))}\n'
              f'exp({exp_arg}) is equal:              {"{:.56f}".format(compute_exp(exp_arg, precision=precision))}\n'
              f'===========================================================================================')

    for number in numbers_of_components:
        print(f'number of components: {number}\n'
              f'log({1 + log_arg}) is equal:    {"{:.56f}".format(compute_log(log_arg, number_of_components=number))}\n'
              f'exp({exp_arg}) is equal:      {"{:.56f}".format(compute_exp(exp_arg, number_of_components=number))}\n'
              f'===========================================================================================')

    print(f'needed components to log (sum version):        {sum_version_count_log(log_arg)}\n'
          f'needed components to exp (sum version):        {sum_version_count_exp(exp_arg)}\n'
          f'=============================================================================================')

    print(f'needed components to log (total sum version):  {total_sum_version_log(log_arg)}\n'
          f'needed components to exp (total sum version):  {total_sum_version_exp(exp_arg)}\n'
          f'=============================================================================================')

    return None


def plots_for_experiment(log_arg=argument_to_log, exp_arg=argument_to_exp, numbers_of_components=None):
    if numbers_of_components is None:
        numbers_of_components = list_with_iterations

    log_array = []
    exp_array = []

    for iterator in numbers_of_components:
        log_array.append(compute_log(log_arg, number_of_components=iterator))
        exp_array.append(compute_exp(exp_arg, number_of_components=iterator))

    plt.plot(numbers_of_components, log_array)
    plt.title(f"Log({1 + log_arg})")
    plt.xlabel("numbers of components")
    plt.ylabel("value")
    plt.show()
    plt.plot(numbers_of_components, exp_array)
    plt.title(f"Exp({exp_arg})")
    plt.xlabel("numbers of components")
    plt.ylabel("value")
    plt.show()

    log_array = []
    exp_array = []

    for grid in iteration_grid_to_plots:
        log_array.append(compute_log(log_arg, number_of_components=grid))
        exp_array.append(compute_exp(exp_arg, number_of_components=grid))

    plt.plot(iteration_grid_to_plots, log_array)
    plt.title(f"Log({1 + log_arg})")
    plt.xlabel("numbers of components")
    plt.ylabel("value")
    plt.show()
    plt.plot(iteration_grid_to_plots, exp_array)
    plt.title(f"Exp({exp_arg})")
    plt.xlabel("numbers of components")
    plt.ylabel("value")
    plt.show()

    return None


if __name__ == '__main__':
    print_results()
    plots_for_experiment()
