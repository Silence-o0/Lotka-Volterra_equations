from sympy import *
import numpy as np
from matplotlib import pyplot as plt


def plot_function(x_arr, y_arr, t):
    plt.plot(t, x_arr, label='Preys')
    plt.plot(t, y_arr, label='Predators')
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.legend()
    plt.ylim(ymin=0)
    plt.show()


def eq_for_Euler_method(F_subs, func, var, current_val):
    var_ = current_val + h * F_subs
    var_eq = current_val + (h / 2) * (F_subs + func.subs(var, var_))
    return var_eq


def Euler_method(F_for_x, F_for_y, x_arr, y_arr, t_arr):
    n = 0
    while t_arr[n] < b:
        F_for_x_subs = F_for_x.subs({x: x_arr[n], y: y_arr[n]})
        x_eq = eq_for_Euler_method(F_for_x_subs, F_for_x, x, x_arr[n])

        F_for_y_subs = F_for_y.subs({x:x_arr[n], y: y_arr[n]})
        y_eq = eq_for_Euler_method(F_for_y_subs, F_for_y, y, y_arr[n])

        solutions = solve([x_eq - x, y_eq - y], x, y, dict=False)

        t_arr = np.append(t_arr, t_arr[n]+h)
        x_arr = np.append(x_arr, solutions[x])
        y_arr = np.append(y_arr, solutions[y])

        n = n + 1

    return x_arr, y_arr, t_arr


def val_for_Runge_Kutta_method(func, target_val_n, target_var, another_val_n, another_var):
    k1 = h * func.subs({another_var: another_val_n, target_var: target_val_n})
    k2 = h * func.subs({another_var: another_val_n + h/2, target_var: target_val_n + k1/2})
    k3 = h * func.subs({another_var: another_val_n + h/2, target_var: target_val_n + k2/2})
    k4 = h * func.subs({another_var: another_val_n + h, target_var: target_val_n + k3})

    value = target_val_n + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return value


def Runge_Kutta_method(F_for_x, F_for_y, x_arr, y_arr, t_arr):
    n = 0
    while t_arr[n] < b:
        x_val = val_for_Runge_Kutta_method(F_for_x, x_arr[n], x, y_arr[n], y)
        y_val = val_for_Runge_Kutta_method(F_for_y, y_arr[n], y, x_arr[n], x)

        t_arr = np.append(t_arr, t_arr[n]+h)
        x_arr = np.append(x_arr, x_val)
        y_arr = np.append(y_arr, y_val)

        n = n + 1

    return x_arr, y_arr, t_arr


if __name__ == '__main__':
    init_printing(use_unicode=False, wrap_line=False)
    x = Symbol('x')
    y = Symbol('y')

    # parameters
    a = 0.02   # probability of prey birth
    b = 0.0005   # probability of prey death
    c = 0.05   # probability of predator death
    d = 0.0005   # probability of predator birth

    F_for_x = (a - b * y) * x
    F_for_y = (-c + d * x) * y

    print("System of Lotka–Volterra equations:")
    print(F_for_x)
    print(F_for_y)
    print()

    a = 0
    b = 200
    h = 0.5

    #start value
    t0 = a
    x0 = 100
    y0 = 25

    if (a > b or a < 0):
        print("Wrong limits")
        exit()

    t_res = np.array([])
    x_res = np.array([])
    y_res = np.array([])

    t_res = np.append(t_res, t0)
    x_res = np.append(x_res, x0)
    y_res = np.append(y_res, y0)

    x_res, y_res, t_res = Euler_method(F_for_x, F_for_y, x_res, y_res, t_res)
    # x_res, y_res, t_res = Runge_Kutta_method(F_for_x, F_for_y, x_res, y_res, t_res)

    print("Result:")
    print("Time: ", t_res)
    print("Preys: ", x_res)
    print("Predators: ", y_res)

    plot_function(x_res, y_res, t_res)
