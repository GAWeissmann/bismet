#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
import numpy as np
import matplotlib.pyplot as plt
from schimpy.schism_mesh import read_mesh, write_mesh
from schimpy.split_quad import calculate_internal_angles


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--skewness', type=float, default=0.02)
    parser.add_argument('--plot', type=bool, default=True)
    return parser


def calculate_skewness(mesh):
    """ Assuming all triangular
    """
    angles = []
    for elem in mesh.elems:
        nodes = mesh.nodes[elem]
        angles.append(calculate_internal_angles(nodes))
    angles = np.array(angles)
    theta_max = np.max(angles, axis=1)
    theta_min = np.min(angles, axis=1)
    theta_e = np.pi / 3.
    skewness = 1. - np.maximum((theta_max - theta_e) / (np.pi - theta_e), 1. -  theta_min / theta_e)
    return skewness


def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    mesh = read_mesh(args.input)
    skewness = calculate_skewness(mesh)
    degenerate = np.argwhere(skewness < args.skewness)
    print("Elements to remove: {}".format(degenerate))
    for i in degenerate:
        mesh.mark_elem_deleted(i)
    mesh.renumber()
    write_mesh(mesh, args.output)
    if args.plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(skewness, bins=20, log=True)
        ax.set_xlabel('Skewness')
        ax.set_ylabel('Count of cells')
        plt.show()
    

if __name__ == '__main__':
    main()
