import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt


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


def solve_numerically_system_of_equations(expressions, functions, t, boundaries, init_t, init_func_list, output_size):
    lower_bound, upper_bound = boundaries
    lamdified_eq = sp.lambdify((t, functions), expressions, modules='numpy')

    def system_of_equations(t, y):
        return lamdified_eq(t, y)

    params = system_of_equations, lower_bound, upper_bound, init_t, init_func_list, output_size
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
    solution = solve_ivp(function, span, initial_y_list, t_eval=np.linspace(*span, output_size), dense_output=True)
    return solution.t, solution.y


def calculate_numerically_list_of_trajectories(expressions, functions, t, t_bound, t_0, boundaries,
                                               inits, output_size, with_t_array=False):
    trajectories = []
    for initial_point in inits:
        t_range, trajectory = solve_numerically_system_of_equations(
            expressions, functions, t, t_bound, t_0, initial_point, output_size
        )
        trajectory_axes = [t_range] if with_t_array else []
        for index, (lower_bound, upper_bound) in enumerate(boundaries):
            trajectory_axis = trajectory[index]
            if lower_bound is not None:
                trajectory_axis[trajectory_axis < lower_bound] = np.nan
            if upper_bound is not None:
                trajectory_axis[trajectory_axis > upper_bound] = np.nan
            trajectory_axes.append(trajectory_axis)
        trajectories.append(tuple(trajectory_axes))
    return trajectories


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


def convert_equations_to_meshgrid(equations, variables, params, normalize=False, damping_factor=0):
    equation_x, equation_v = equations
    x_var, v_var = variables
    x_0, x_k, x_step, v_0, v_k, v_step = params
    lambdified_x = sp.lambdify([x_var, v_var], equation_x.rhs, 'numpy')
    lambdified_v = sp.lambdify([x_var, v_var], equation_v.rhs, 'numpy')
    X, V = np.meshgrid(np.arange(x_0, x_k + x_step, x_step), np.arange(v_0, v_k + v_step, v_step))
    dx = lambdified_x(X, V)
    dv = lambdified_v(X, V)
    norm = np.sqrt(dx ** 2 + dv ** 2)
    norm[norm < 1] = 1
    if normalize:
        dx = dx / norm
        dv = dv / norm
    else:
        denominator = ((norm / norm.mean()) ** damping_factor)
        dx = dx / denominator
        dv = dv / denominator
    return X, V, dx, dv


def phase_portrait(equations, variables, params, axes, normalize=False, damping_factor=0):
    meshgrid = convert_equations_to_meshgrid(equations, variables, params,
                                             normalize=normalize,
                                             damping_factor=damping_factor)
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


def find_fixed_point_abcd(P, Q, x, y):
    a = P.rhs.diff(x)
    b = P.rhs.diff(y)
    c = Q.rhs.diff(x)
    d = Q.rhs.diff(y)
    return a, b, c, d


def find_fixed_point_parameters(P, Q, x, y, fixed_point=None):
    a, b, c, d = find_fixed_point_abcd(P, Q, x, y)
    q = a*d - b*c
    p = -(a + d)
    r = p**2 - 4*q
    if fixed_point is not None:
        x_0, y_0 = fixed_point
        q = q.subs([(x, x_0), (y, y_0)]).evalf()
        p = p.subs([(x, x_0), (y, y_0)]).evalf()
        r = r.subs([(x, x_0), (y, y_0)]).evalf()
    return q, p, r


def classify_fixed_point(q, p, r):
    if q < 0 < r:
        return 'saddle'
    elif q == 0 and r >= 0:
        return 'higher order'
    elif q > 0:
        if p > 0 > r:
            return 'stable focal'
        elif p > 0 and r >= 0:
            return 'stable nodal'
        elif p == 0 and r < 0:
            return 'vortex/focal'
        elif p < 0 and r < 0:
            return 'unstable focal'
        elif p < 0 <= r:
            return 'unstable nodal'
        else:
            return 'unclassified'
    else:
        return 'unclassified'


def colormap_for_array(array, colormap_type='viridis'):
    cmap = plt.get_cmap(colormap_type)
    norm = plt.Normalize(array.min(), array.max())
    return cmap, norm


def plot_colored_line3d(x, y, z, array, axes, colormap_type='viridis', return_cmap=False):
    cmap, norm = colormap_for_array(array, colormap_type=colormap_type)
    for t in range(len(array) - 1):
        axes.plot(x[t:t+2], y[t:t+2], z[t:t+2], color=cmap(norm(array[t])))
    if return_cmap:
        return cmap, norm
