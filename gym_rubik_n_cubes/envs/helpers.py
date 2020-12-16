import numpy as np
from scipy.spatial.transform import Rotation


def add(u, v):
    return tuple(sum(x) for x in zip(u, v))


def multiply(v, m):
    return tuple(x * m for x in v)


def rotate(v, along, direction=1):
    return tuple(Rotation.from_rotvec(multiply(along, direction * np.pi / 2)).apply(v))


def dot(u, v):
    return sum(a * b for a, b in zip(u, v))


def print_coloured_square(colour):
    if colour == -1:  # none
        pass
    elif colour == 0:  # white
        print('\x1b[1;107m', end='')
    elif colour == 1:  # yellow
        print('\x1b[1;103m', end='')
    elif colour == 2:  # orange
        print('\x1b[1;44m', end='')
    elif colour == 3:  # blue
        print('\x1b[1;42m', end='')
    elif colour == 4:  # red
        print('\x1b[0;43m', end='')
    elif colour == 5:  # green
        print('\x1b[1;41m', end='')
    else:
        assert False, 'Colour must be in range [-1, 5]'

    print('  \x1b[1;0m', end='')
