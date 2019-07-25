#! /usr/bin/env python

import matplotlib
matplotlib.use('agg')

from glob import glob
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess

import shapely.geometry as sg


OFILE = 'images/samplegrid.png'
def main():
    NROWS = 40
    NCOLS = 20
    WIDTH = 2.66 # in
    HEIGHT = 3.74 # in
    
    column = [ (i, 0) for i in range(NROWS) ]
    row = [ (10, i) for i in range(NCOLS) ]
    square = []
    for i in range(20, 30):
        for j in range(5, 15):
            square.append((i, j))
    spec = column + row + square
    
   
    fig = plt.figure()
    fig.set_size_inches(WIDTH, HEIGHT, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    #ax.set_axis_off()
    fig.add_axes(ax)
    
    for r,c in spec:
        bbox = sg.box(c, r, c+1, r+1)
        ax.plot(*bbox.exterior.xy, color='black')
    
    plt.gca().set_aspect('equal')
    plt.savefig(OFILE, facecolor='gray')

if __name__ == '__main__':
    main()
