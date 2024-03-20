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


def evaluate_function(expression, variable, variable_values, threshold=None):
    lambdified_eq = sp.lambdify(variable, expression, 'numpy')
    function_values = lambdified_eq(variable_values)
    if threshold is not None:
        function_values[function_values < threshold[0]] = np.nan
        function_values[function_values > threshold[1]] = np.nan
    return function_values


def solve_equation_with_scipy(function, span, init_func_list, t_eval, method='RK45'):
    solution = solve_ivp(function, span, init_func_list, t_eval=t_eval, dense_output=True, method=method)
    return solution.t, solution.y


def solve_numerically_first_order_ode(function, t_values, init_t, init_func_list, method='RK45'):
    lower_bound, upper_bound = t_values.min(), t_values.max()
    if lower_bound < init_t < upper_bound:
        split_index = np.searchsorted(t_values, init_t)
        t_backward = t_values[:split_index][::-1]
        t_forward = t_values[split_index:]
        span_backward = (init_t, lower_bound)
        span_forward = (init_t, upper_bound)
        t_range_backward, sol_backward = solve_equation_with_scipy(
            function, span_backward, init_func_list, t_backward, method=method)
        t_range_forward, sol_forward = solve_equation_with_scipy(
            function, span_forward, init_func_list, t_forward, method=method)
        t_range = np.concatenate((t_range_backward[::-1], t_range_forward), axis=0)
        sol = np.concatenate((sol_backward[::, ::-1], sol_forward), axis=1)
    elif init_t <= lower_bound:
        span = (init_t, upper_bound)
        t_range, sol = solve_equation_with_scipy(function, span, init_func_list, t_values, method=method)
    elif init_t >= upper_bound:
        span = (init_t, lower_bound)
        t_range, sol = solve_equation_with_scipy(function, span, init_func_list, t_values, method=method)
    else:
        t_range, sol = np.array([]), np.array([])
    return t_range, sol


def solve_numerically_system_of_equations(expressions, functions, t, t_values, init_t, init_func_list, method='RK45'):
    lamdified_eq = sp.lambdify((t, functions), expressions, modules='numpy')

    def system_of_equations(t, y):
        return lamdified_eq(t, y)

    params = system_of_equations, t_values, init_t, init_func_list
    return solve_numerically_first_order_ode(*params, method=method)


def solve_numerically_second_order_ode(diff_equation, x, t, t_values, init_t, init_func_list, method='RK45'):
    v = sp.Function('v')(t)
    x_diff_eq = v
    v_diff_eq = sp.solve(diff_equation, x.diff(t, 2)[0]).subs(x.diff(t), v)
    params = [x_diff_eq, v_diff_eq], [x, v], t, t_values, init_t, init_func_list
    return solve_numerically_system_of_equations(*params, method=method)


def calculate_numerically_list_of_trajectories(expressions, functions, t, t_values, init_t, axis_thresholds, inits,
                                               with_t_array=False, method='RK45'):
    trajectories = []
    for initial_point in inits:
        t_range, trajectory = solve_numerically_system_of_equations(
            expressions, functions, t, t_values, init_t, initial_point, method=method
        )
        trajectory_axes = [t_range] if with_t_array else []
        for index, (lower_bound, upper_bound) in enumerate(axis_thresholds):
            trajectory_axis = trajectory[index]
            if lower_bound is not None:
                trajectory_axis[trajectory_axis < lower_bound] = np.nan
            if upper_bound is not None:
                trajectory_axis[trajectory_axis > upper_bound] = np.nan
            trajectory_axes.append(trajectory_axis)
        trajectories.append(tuple(trajectory_axes))
    return trajectories


def convert_equations_to_meshgrid(expressions, functions, params, normalize=False, damping_factor=0):
    expression_x, expression_v = expressions
    x_var, v_var = functions
    x_0, x_k, x_step, v_0, v_k, v_step = params
    lambdified_x = sp.lambdify([x_var, v_var], expression_x, 'numpy')
    lambdified_v = sp.lambdify([x_var, v_var], expression_v, 'numpy')
    X, V = np.meshgrid(np.arange(x_0, x_k + x_step, x_step), np.arange(v_0, v_k + v_step, v_step))
    dx = lambdified_x(X, V)
    dv = lambdified_v(X, V)
    norm = np.sqrt(dx ** 2 + dv ** 2)
    if normalize:
        norm[norm == 0] = 1
        dx = dx / norm
        dv = dv / norm
    else:
        norm[norm < 1] = 1
        denominator = ((norm / norm.mean()) ** damping_factor)
        dx = dx / denominator
        dv = dv / denominator
    return X, V, dx, dv


def phase_portrait(expressions, functions, params, axes, normalize=False, damping_factor=0):
    meshgrid = convert_equations_to_meshgrid(expressions, functions, params,
                                             normalize=normalize,
                                             damping_factor=damping_factor)
    axes.quiver(*meshgrid, pivot='mid')
    axes.set(xlabel=f'${sp.latex(functions[0])}$', ylabel=f'${sp.latex(functions[1])}$')


def phase_trajectory(expression, t, t_values, axis_thresholds=None, second_expression=None):
    lambdified_x = sp.lambdify(t, expression, 'numpy')
    if second_expression is None:
        lambdified_v = sp.lambdify(t, expression.diff(t), 'numpy')
    else:
        lambdified_v = sp.lambdify(t, second_expression, 'numpy')
    x_values, v_values = lambdified_x(t_values), lambdified_v(t_values)
    if axis_thresholds is not None:
        for trajectory_axis, (lower_bound, upper_bound) in zip([x_values, v_values], axis_thresholds):
            if lower_bound is not None:
                trajectory_axis[trajectory_axis < lower_bound] = np.nan
            if upper_bound is not None:
                trajectory_axis[trajectory_axis > upper_bound] = np.nan
    return x_values, v_values


def find_fixed_point_abcd(P, Q, x, y):
    a = P.diff(x)
    b = P.diff(y)
    c = Q.diff(x)
    d = Q.diff(y)
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


def evaluate_list_of_equation_solutions(solutions):
    return [tuple(component.evalf() for component in point) for point in solutions]
