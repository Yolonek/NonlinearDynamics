import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp


def n_components_of_function(function, variable, n=1):
    return tuple(sp.Function(f'{sp.latex(function)[0]}_{i}')(variable) for i in range(n))


def orders_of_perturbation(terms, perturbation):
    equation = terms[0]
    for i, term in enumerate(terms[1:]):
        equation += perturbation**(i + 1) * term
    return equation


def evaluate_function(equation, variable, x_values, threshold=None):
    lambdified_eq = sp.lambdify(variable, equation, 'numpy')
    y_values = lambdified_eq(x_values)
    if threshold is not None:
        y_values[y_values < threshold[0]] = np.nan
        y_values[y_values > threshold[1]] = np.nan
    return y_values


def solve_second_order_differential_equation(equation, x, f, lower_bound, upper_bound, initial_x, initial_y_dy,
                                             output_size):
    v = sp.Function('t')(x)
    f_prime = v
    v_prime = sp.solve(equation, f.diff(x, 2))[0].subs(f.diff(x), v)

    lambdified_eq = sp.lambdify((f, v, x), [f_prime, v_prime], modules='numpy')

    def second_order_equation(t, yv):
        return lambdified_eq(*yv, t)

    params = second_order_equation, lower_bound, upper_bound, initial_x, initial_y_dy, output_size
    return solve_differential_equation(*params)


def solve_differential_equation(function, lower_bound, upper_bound, initial_x, initial_y_list, output_size):
    if lower_bound < initial_x < upper_bound:
        span_backward = (initial_x, lower_bound)
        span_forward = (initial_x, upper_bound)
        x_range_backward, sol_backward = solve_equation_with_scipy(function, span_backward, initial_y_list, output_size // 2)
        x_range_forward, sol_forward = solve_equation_with_scipy(function, span_forward, initial_y_list, output_size // 2)
        x_range = np.concatenate((x_range_backward[::-1], x_range_forward), axis=0)
        sol = np.concatenate((sol_backward[::, ::-1], sol_forward), axis=1)
    elif initial_x <= lower_bound:
        span = (initial_x, upper_bound)
        x_range, sol = solve_equation_with_scipy(function, span, initial_y_list, output_size)
    elif initial_x >= upper_bound:
        span = (initial_x, lower_bound)
        x_range, sol = solve_equation_with_scipy(function, span, initial_y_list, output_size)
    else:
        x_range, sol = np.array([]), np.array([])
    return x_range, sol


def solve_equation_with_scipy(function, span, initial_y_list, output_size):
    x_range = np.linspace(*span, output_size)
    solution = solve_ivp(function, span, initial_y_list, t_eval=x_range, dense_output=True)
    return x_range, solution.y


def plot_numeric_solutions(diff_equation, x, f, initial_conditions_y, initial_x, span, size, axes, threshold=None):
    lambdified_eq = sp.lambdify((x, f), diff_equation, 'numpy')

    def first_order_equation(x, f):
        return lambdified_eq(x, f)

    x_range, sol = solve_differential_equation(first_order_equation, *span, initial_x, initial_conditions_y, size)
    for index, y0 in enumerate(initial_conditions_y):
        y_values = sol[index]
        if threshold is not None:
            y_values[y_values < threshold[0]] = np.nan
            y_values[y_values > threshold[1]] = np.nan
        axes.plot(x_range, y_values, label=f'$y({initial_x}) = {y0}$')


def convert_equations_to_meshgrid(equations, variables, params, normalize=False):
    equation_x, equation_v = equations
    x_var, v_var = variables
    x_0, x_k, x_step, v_0, v_k, v_step = params
    lambdified_x = sp.lambdify([x_var, v_var], equation_x.rhs, 'numpy')
    lambdified_v = sp.lambdify([x_var, v_var], equation_v.rhs, 'numpy')
    X, V = np.meshgrid(np.arange(x_0, x_k + x_step, x_step), np.arange(v_0, v_k + v_step, v_step))
    dx = lambdified_x(X, V)
    dv = lambdified_v(X, V)
    if normalize:
        norm = np.sqrt(dx ** 2 + dv ** 2)
        norm[norm == 0] = 1
        dx = dx / norm
        dv = dv / norm
    return X, V, dx, dv


def phase_portrait(equations, variables, params, axes, normalize=False):
    meshgrid = convert_equations_to_meshgrid(equations, variables, params, normalize=normalize)
    axes.quiver(*meshgrid, pivot='mid')
    axes.set(xlabel=f'${sp.latex(variables[0])}$', ylabel=f'${sp.latex(variables[1])}$')


def phase_trajectory(diff_solution, t, params, threshold=None, second_solution=None):
    t_0, t_k, quality = params
    t_values = np.linspace(t_0, t_k, quality)
    lambdified_x = sp.lambdify(t, diff_solution.rhs, 'numpy')
    if second_solution is None:
        lambdified_v = sp.lambdify(t, diff_solution.rhs.diff(t), 'numpy')
    else:
        lambdified_v = sp.lambdify(t, second_solution.rhs, 'numpy')
    x_values, v_values = lambdified_x(t_values), lambdified_v(t_values)
    if threshold is not None:
        x_values[x_values < threshold[0]] = np.nan
        x_values[x_values > threshold[1]] = np.nan
        v_values[v_values < threshold[2]] = np.nan
        v_values[v_values > threshold[3]] = np.nan
    return t_values, x_values, v_values
