# -*- coding:utf-8 -*-
# /usr/bin/python

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math


# calculate Y according to X
def func(a):
    if a < 1e-6:
        return 0
    last = a
    c = a / 2
    while math.fabs(c - last) > 1e-6:
        last = c
        c = (c + a/c) / 2
    return c


if __name__ == '__main__':
    # set picture paraameter
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    # return ndarray: evenly sapced number over a specified interval
    # endPoint = True, retstep=False, dtype=None, axis=0
    x = np.linspace(start=0, stop=30, num=50)
    # [0.0, 0.61, ..., 29.39, 30.0], 50 number in total
    x.tolist()
    # take a arbitary Python function and returns a NumPy func
    # nin  = 1, number of input parameters is 1
    # nout = 1, number of returned objects is 1
    func_ = np.frompyfunc(func, 1, 1)
    # the same as "y = np.sqrt(x)“
    y = func_(x)
    # [0, 0.78, ..., 5.48], 50 number in total
    y.tolist()
    # matplotlib.pyplot.figure 
    # - figsize: width, height
    # - facecolor: the figure patch facecolor
    # for other parameter, query the DOC of matplotlib.pyplot.figure
    plt.figure(figsize=(10, 5), facecolor='w')
    # matplotlib.pyplot.plot
    #   Plot y versus x as lines and/or markers
    #   x: x numpy.ndarray
    #   y: y numpy.ndarray
    #   fmt: 'ro-', red, line and marker
    #   lw: line width
    #   markersize: size of the marker
    plt.plot(x, y, 'ro-', lw=2, markersize=6)
    # matplotlib.pyplot.grid
    #   b: True, show the grid
    #   ls: line style 
    plt.grid(b=True, ls=':')
    # X,Y label, Title
    plt.xlabel('X', fontsize=16)
    plt.ylabel('Y', fontsize=16)
    plt.title('What is this code doing?', fontsize=18)
    # show the plot
    plt.show()
