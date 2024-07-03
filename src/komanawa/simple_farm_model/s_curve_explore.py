"""
created matt_dumont 
on: 7/3/24
"""

import numpy as np
import matplotlib.pyplot as plt


def s_curve(x, s, a, b, c):
    """
    S-curve function
    :param x: x value
    :param s: scale - the maximum value of the curve (if s=1, the maximum value is 1)
    :param a: steepness - smoothing parameter as a increases the curve becomes steeper and the inflection point moves to the right
    :param b: steepness about the inflection point
    :param c: inflection point (if a=1, c is the x value at which y=0.5)
    :return:
    """
    return s * (1 / (1 + np.exp(-b * (x - c)))) ** a


def a_var():
    fig, ax = plt.subplots(figsize=(10, 10))
    for a in np.arange(-5, 2, 0.5):
        a = 10 ** a

        x = np.linspace(-50, 50, 100)
        y = s_curve(x, 1, a, 1, 10)
        ax.plot(x, y, label=f'a={a}')
    ax.legend()


def b_var():
    fig, ax = plt.subplots(figsize=(10, 10))
    for b in np.arange(-2, 3, 0.5):
        b = 10 ** b

        x = np.linspace(-50, 50, 100)
        y = s_curve(x, 1, 1, b, 10)
        ax.plot(x, y, label=f'b={b}')
    ax.legend()


def example_feed_scarcity_cost():
    x = np.linspace(0, 100, 100)
    c=33
    y = s_curve(x, s=5, a=.85, b=0.33, c=c)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(x, y)
    ax.axvline(14, ls='--', color='r')
    ax.axvline(28, ls='--', color='r')
    ax.axhline(0, ls='--', color='r')
    ax.axvline(c, ls='--', color='r')


if __name__ == '__main__':
    example_feed_scarcity_cost()
    plt.show()
    a_var()
    b_var()
    plt.show()
