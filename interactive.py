# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from energymin import *
import numpy as np
from scipy.special import binom

import matplotlib.pyplot as plt

from matplotlib.lines import Line2D


class ControlPolygon(object):
    def __init__(self, x_data=None, y_data=None, distance_threshold=0.025):
        self.points = [[x, y] for x, y in zip(x_data, y_data)] if (x_data is not None and y_data is not None) else []
        self.known_indices = []
        self.num_points = 0
        self.distance_threshold = distance_threshold

    @property
    def xy_data(self):
        x = [point[0] for point in self.points]
        y = [point[1] for point in self.points]
        return x, y

    def add_point(self, x, y):
        self.points.append([x, y])
        self.known_indices.append(self.num_points)
        self.num_points += 1

    def get_nearest_point(self, x_in, y_in):
        nearest_index = None
        nearest_distance = float('inf')
        for i, (x, y) in enumerate(self.points):
            distance = abs(x - x_in) + abs(y - y_in)
            if distance < nearest_distance:
                nearest_index = i
                nearest_distance = distance

        return nearest_index if nearest_distance < self.distance_threshold else None

    def update_control_point(self, index, x_in, y_in):
        self.points[index][0] = x_in
        self.points[index][1] = y_in

    def remove_control_point(self, nearest_index):
        self.points.pop(nearest_index)
        new_known_indices = []
        for index in self.known_indices:
            if index < nearest_index:
                new_known_indices.append(index)
            elif index > nearest_index:
                new_known_indices.append(index-1)
        self.known_indices = new_known_indices
        self.num_points -= 1

    def is_index_first_or_last(self, index):
        return index in (0, self.num_points-1)


class BezierBuilder(object):
    """Bézier curve interactive builder.
    """

    def __init__(self, control_polygon, ax_energy):
        """Constructor.
        Receives the initial control polygon of the curve.
        """
        self.control_polygon_handler = control_polygon
        self.control_polygon = ControlPolygon(x_data=control_polygon.get_xdata(), y_data=control_polygon.get_ydata())
        # self.xp = list(control_polygon.get_xdata())
        # self.yp = list(control_polygon.get_ydata())
        self.known_matrix = None
        # self.known_indices = []
        # self.known_points = None
        # self.unknown_matrix = None
        # self.update_known_point_matrix()
        self.canvas = control_polygon.figure.canvas
        self.ax_main = control_polygon.axes
        self.ax_energy = ax_energy
        self.degree = 1
        # self.num_points = 0
        self.is_generated_point = False

        # Event handler for mouse clicking
        self.press = self.canvas.mpl_connect('button_press_event', self.on_press)
        self.motion = self.canvas.mpl_connect('motion_notify_event',self.on_motion)
        self.release = self.canvas.mpl_connect('button_release_event', self.on_release)
        self.key_event = self.canvas.mpl_connect('key_press_event', self.handle_key_event)

        # Variables to know when we really need to add a point (when
        # there's no mouse movement between button press and release)
        self.moved_before_release = False
        self.pressed = False

        self.nearest_point_index = None
        # self.distance_threshold = 0.025

        # Create Bézier curve
        line_bezier = Line2D([], [],
                             c=control_polygon.get_markeredgecolor())
        self.bezier_curve = self.ax_main.add_line(line_bezier)

    def on_press(self, event):
        # Ignore clicks outside axes
        if event.inaxes != self.control_polygon_handler.axes:
            return
        else:
            self.pressed = True
            if self.control_polygon.num_points > 0:
                self.nearest_point_index = self.control_polygon.get_nearest_point(event.xdata, event.ydata)
                # if nearest_distance <= self.distance_threshold:
                #     self.nearest_point_index = nearest_index
                # else:
                #     self.nearest_point_index = None

    def on_motion(self, _event_):
        # We need to handle events only when there's no movement
        # between button press and button release. If there's some
        # movement in between, it means the user is zooming or panning
        if self.pressed:
            self.moved_before_release = True

    def on_release(self, event):
        if self.pressed:
            if not self.moved_before_release:
                # Add point
                self.control_polygon.add_point(event.xdata, event.ydata)
            else:
                if self.nearest_point_index is not None:
                    self.control_polygon.update_control_point(self.nearest_point_index, event.xdata, event.ydata)
            self.update_curves()
        self.pressed = False
        self.moved_before_release = False

    # def get_nearest_point(self, x, y):
    #     nearest_index = None
    #     nearest_distance = float('inf')
    #     for i, (point_x, point_y) in enumerate(zip(self.xp, self.yp)):
    #         distance = abs(point_x - x) + abs(point_y - y)
    #         if distance < nearest_distance:
    #             nearest_index = i
    #             nearest_distance = distance
    #     return nearest_index, nearest_distance

    # def add_control_point(self, x, y):
    #     self.xp.append(x)
    #     self.yp.append(y)
    #     self.known_indices.append(self.num_points)  # add before number increases to avoid off-by-one
    #     print('known indices: {}'.format(self.known_indices))
    #     self.update_known_point_matrix()
    #     self.num_points += 1
    #
    # def update_control_point(self, index, x, y):
    #     self.xp[index] = x
    #     self.yp[index] = y
    #     self.update_known_point_matrix()
    #
    # def remove_control_point(self, nearest_index):
    #     self.xp.pop(nearest_index)
    #     self.yp.pop(nearest_index)
    #     new_known_indices = []
    #     for point in self.known_indices:
    #         if point < nearest_index:
    #             new_known_indices.append(point)
    #         elif point > nearest_index:
    #             new_known_indices.append(point-1)
    #     self.known_indices = new_known_indices
    #     print('known indices: {}'.format(self.known_indices))
    #     self.update_curves()
    #     self.update_known_point_matrix()
    #     self.num_points -= 1

    def known_to_unknown_point(self, index):
        self.known_indices = [x for x in self.known_indices if x != index]
        self.known_points = np.array([[x, y] for i, (x, y) in enumerate(zip(self.xp, self.yp)) if i not in self.known_indices])
        unknown_matrix = self._construct_unknown_energy_min_matrix()
        known_matrix = self._construct_known_energy_min_matrix()
        print('unknown: {}'.format(unknown_matrix))
        print('  known: {}'.format(known_matrix))
        result = np.linalg.solve(unknown_matrix, known_matrix)
        print('B: {}'.format(result))
        unknown_inds = sorted(list(set(range(self.num_points)) - set(self.known_indices)))
        for i, unknown_ind in enumerate(unknown_inds):
            b_i = result[i]
            print('{0} -> {1}'.format(i, b_i))
            self.xp[unknown_ind] = b_i[0]
            self.yp[unknown_ind] = b_i[1]
        self.update_curves()

    def unknown_to_known_point(self, index):
        if index not in self.known_indices:
            self.known_indices.append(index)
            self.known_indices.sort()

    def update_known_point_matrix(self):
        self.known_points = np.array([[x, y] for x, y in zip(self.xp, self.yp)])

    def _construct_unknown_energy_min_matrix(self):
        size = self.num_points - len(self.known_indices)
        n_matrix = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                n_matrix[i][j] = inner_sum(i+1, j+1, self.num_points, self.degree)
        return n_matrix

    def _construct_known_energy_min_matrix(self):
        size = len(self.known_points)
        n_matrix = np.zeros((size, 2))
        for i, known_index in enumerate(self.known_indices):
            print(self.known_points)
            print(self.known_indices)
            n_matrix[i] = -1.*sum(inner_sum(known_index, j, self.num_points, self.degree)*known_point
                                  for known_point, j in zip(self.known_points, self.known_indices))
        return n_matrix

    def handle_key_event(self, event):
        if event.key == 'd':
            nearest_index = self.control_polygon.get_nearest_point(event.xdata, event.ydata)
            if nearest_index is not None:
                self.control_polygon.remove_control_point(nearest_index)
                self.update_curves()

        elif event.key == 'e':
            nearest_index = self.control_polygon.get_nearest_point(event.xdata, event.ydata)
            if nearest_index is not None and self.control_polygon.is_index_first_or_last(nearest_index):
                # self.known_to_unknown_point(nearest_index)
                print(self.known_indices)

        elif event.key in ('1', '2', '3'):
            self.degree = int(event.key)

    def update_curves(self):
        self.control_polygon_handler.set_data(*self.control_polygon.xy_data)

        # Rebuild Bézier curve and update canvas
        bezier_x, bezier_y, norm = self._build_energy_curve()
        self.bezier_curve.set_data(bezier_x, bezier_y)
        self._update_energy(norm)
        self._update_bezier()

    def _build_energy_curve(self):
        x, y = bezier(self.control_polygon.points).T
        t = np.linspace(0, 1, num=200)
        diffx = np.gradient(x, t[1] - t[0])
        diffy = np.gradient(x, t[1] - t[0])
        norm = np.zeros(np.size(diffx))
        for i in range(len(norm)):
            norm[i] = diffx[i]*diffx[i] + diffy[i]*diffy[i]
        return x, y, norm

    def _update_bezier(self):
        self.canvas.draw()

    def _update_energy(self, norm):
        n = self.control_polygon.num_points - 1
        t = np.linspace(0, 1, num=200)

        ax = self.ax_energy
        ax.clear()
        for k in range(n + 1):
            ax.plot(t, norm)
        ax.set_title("Energy, N = {0}, degree = {1}".format(self.control_polygon.num_points, self.degree))
        ax.set_xlabel('t')
        ax.set_ylabel('Energy')


def bernstein(n, k):
    """Bernstein polynomial.
    """
    coefficient = binom(n, k)

    def _bernstein_polynomial(x):
        return coefficient * x ** k * (1 - x) ** (n - k)

    return _bernstein_polynomial


def bezier(points, num=200):
    """Build Bézier curve from points.
    """
    n = len(points)
    t = np.linspace(0, 1, num=num)
    curve = np.zeros((num, 2))
    for i in range(n):
        curve += np.outer(bernstein(n - 1, i)(t), points[i])
    return curve


if __name__ == '__main__':
    # Initial setup
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Empty line
    line = Line2D([], [], ls='--', c='#666666',
                  marker='x', mew=2, mec='#204a87')
    ax1.add_line(line)

    # Canvas limits
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_title("Bézier curve")

    # Bernstein plot
    ax2.set_title("Energy degree = 1")
    ax2.set_xlabel('t')
    ax2.set_ylabel('Energy')

    # Create BezierBuilder
    bezier_builder = BezierBuilder(line, ax2)

    plt.show()