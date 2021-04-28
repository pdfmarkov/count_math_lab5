import warnings
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import genfromtxt
from tabulate import tabulate
from termcolor import cprint

warnings.filterwarnings("ignore", category=RuntimeWarning)


##############################
#       APPROXIMATION        #
##############################

# Printing matrix in the comfortable view
def print_matrix(input_matrix):
    cprint(tabulate(input_matrix,
                    tablefmt="fancy_grid", floatfmt="2.5f"), 'cyan')


# Getting x and f(x) from the file by filename
def get_xy_from_file():
    cprint('Please, write the name of your file!', 'yellow')
    filename = input()
    with open(filename) as f:
        this_xy = [list(map(float, row.split())) for row in f.readlines()]
        this_xy[0].insert(0, "X")
        this_xy[1].insert(0, "Y")
    cprint('\nFunction:', 'cyan', attrs=['bold'])
    print_matrix(this_xy, 5)
    this_xy[0].pop(0)
    this_xy[1].pop(0)
    return this_xy


def write_number(s='Write a number:', integer=False, check=None):
    if check is None:
        check = [False, 0, 0]
    flag = True
    while flag:
        flag = False
        try:
            if integer:
                val = int(input(s))
            else:
                val = float(input(s))
            if check[0] and (val < check[1] or val > check[2]):
                raise ValueError
        except ValueError:
            flag = True
            if check[0]:
                cprint(f'\nPlease, try again! Your input should be between [{check[1]}; {check[2]}]\n', 'yellow',
                       attrs=['bold'])
            else:
                cprint('Please, try again!\n', 'yellow', attrs=['bold'])
    return val


def parse():
    flag = True
    while flag:
        print()
        cprint('Write your path:\n', 'green', attrs=['bold'])
        path = input().strip()
        try:
            a = genfromtxt(path, delimiter=',')
            if True in np.isnan(a) or a.shape[0] != 2:
                raise ValueError
            return a
        except ValueError:
            cprint('It is necessary to have two rows in the file!\n', 'red', attrs=['bold'])
        except OSError:
            cprint('I can\'t find this file ;(\n', 'red', attrs=['bold'])
        cprint('Please, try again!\n', 'yellow', attrs=['bold'])


def write_values():
    print()
    n = write_number(s='Write the number of values: ', integer=True)
    print()
    a = []
    for i in range(int(n)):
        a.append([write_number('x'), write_number('y')])
        print()
    return np.array(a).transpose()


def check_and_draw(x, y, approximate_function, title, point):
    fig, ax = plt.subplots()
    xnew = np.linspace(np.min(x), np.max(x), 100)
    ynew = [approximate_function(x, y, i)[0] for i in xnew]
    plt.plot(x, y, 'o', color='r', label='input data')
    plt.plot(xnew, ynew, color='b', label='approximate function')
    plt.plot(point[0], point[1], '*', color='g', markersize=12, label='answer')
    plt.title(title)
    ax.legend()
    plt.grid(True)
    plt.show()


##############################
#          LAGRANGE          #
##############################
def lagrange(array_x, array_y, cur_x):
    array_x.astype(float)
    array_y.astype(float)
    lag = 0
    lagrangians = []
    for j in range(len(array_y)):
        multiplying = 1
        for i in range(len(array_x)):
            if i != j:
                multiplying *= (cur_x - array_x[i]) / (array_x[j] - array_x[i])
        lagrangians.append(array_y[j] * multiplying)
        lag += array_y[j] * multiplying
    return lag, lagrangians


##############################
#           NEWTON           #
##############################
def count_coef(array_x, array_y):
    m = len(array_x)

    array_x.astype(float)
    array_y.astype(float)
    array_x = np.copy(array_x)
    array_y = np.copy(array_y)
    for k in range(1, m):
        array_y[k:m] = (array_y[k:m] - array_y[k - 1]) / (array_x[k:m] - array_x[k - 1])
    return array_y


def newton_polynomial(array_x, array_y, cur_x):
    coef = count_coef(array_x, array_y)

    n = len(array_x) - 1  # Degree of polynomial
    cur_y = coef[n]

    for k in range(1, n + 1):
        cur_y = coef[n - k] + (cur_x - array_x[n - k]) * cur_y
    return cur_y, None


def main_func():
    again = True
    cprint('! WELCOME TO THE INTERPOLATION CALCULATOR !\n', 'green', attrs=['bold'])
    while again:
        again = False
        cprint('Please, write...\n\tk - if u want to put data from keyboard\n\tf - if u want from file\n', 'green',
               attrs=['bold'])
        in_type = input()
        if in_type.strip() == 'k':
            data = write_values()
        elif in_type.strip() == 'f':
            data = parse()
        else:
            print('Oh, your data is broken ;(\n Please, try again!')
            again = True
    cprint('\nData:', 'cyan', attrs=['bold'])
    print_matrix(data)

    cprint('\nPlease, write a number, which u want to interpolate: \n', 'green', attrs=['bold'])
    cur_x = write_number('', check=[True, min(data[0]), max(data[0])])

    lag, lagrangians = lagrange(data[0], data[1], cur_x)
    cprint(f'\nLAGRANGE', 'cyan', attrs=['bold'])
    cprint(f'ANSWER: {lag}', 'cyan')
    check_and_draw(data[0], data[1], lagrange, 'LAGRANGE', [cur_x, lag])
    print()

    newtone_answer = newton_polynomial(data[0], data[1], cur_x)[0]
    cprint(f'\nNEWTON', 'cyan', attrs=['bold'])
    cprint(f'ANSWER: {newtone_answer}', 'cyan')
    check_and_draw(data[0], data[1], newton_polynomial, 'NEWTON', [cur_x, newtone_answer])


try:
    main_func()
except Exception as ex:
    template = "Oh, you've got an exception! What a pity! So, the problem is...\n{1!r}"
    message = template.format(type(ex).__name__, ex.args)
    print(message)
