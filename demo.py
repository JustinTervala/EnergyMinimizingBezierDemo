# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.interpolate import UnivariateSpline
from scipy.special import binom

from helpers import inner_sum


class ControlPolygon(object):
    def __init__(self, x_data=None, y_data=None, distance_threshold=0.025, degree=1):
        self.points = [[x, y] for x, y in zip(x_data, y_data)] if (x_data is not None and y_data is not None) else []
        self.num_points = len(self.points)
        self.known_indices = range(self.num_points)
        self.distance_threshold = distance_threshold
        self._degree = degree

    @property
    def xy_data(self):
        x = [point[0] for point in self.points]
        y = [point[1] for point in self.points]
        return x, y

    @property
    def known_xy_data(self):
        known_points = [self.points[i] for i in self.known_indices]
        x = [point[0] for point in known_points]
        y = [point[1] for point in known_points]
        return x, y

    @property
    def unknown_xy_data(self):
        known_points = [self.points[i] for i in self.unknown_indices]
        x = [point[0] for point in known_points]
        y = [point[1] for point in known_points]
        return x, y

    @property
    def unknown_indices(self):
        return sorted(list(set(range(self.num_points)) - set(self.known_indices)))

    @property
    def degree(self):
        return self._degree

    @degree.setter
    def degree(self, degree):
        self._degree = degree
        self._update_points_from_known_unknown_matrix()

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
        if index not in self.known_indices:
            self.known_indices.append(index)
            self.known_indices.sort()
        self.points[index][0] = x_in
        self.points[index][1] = y_in

    def remove_control_point(self, nearest_index):
        self.points.pop(nearest_index)
        new_known_indices = []
        for index in self.known_indices:
            if index < nearest_index:
                new_known_indices.append(index)
            elif index > nearest_index:
                new_known_indices.append(index - 1)
        self.known_indices = new_known_indices
        self.num_points -= 1

    def is_index_first_or_last(self, index):
        return index in (0, self.num_points - 1)

    def known_to_unknown_point(self, index):
        self.known_indices = [x for x in self.known_indices if x != index]
        self._update_points_from_known_unknown_matrix()

    def _update_points_from_known_unknown_matrix(self):
        if len(self.known_indices) != self.num_points:
            unknown_matrix = self._construct_unknown_energy_min_matrix()
            known_matrix = self._construct_known_energy_min_matrix()
            result = np.linalg.solve(unknown_matrix, known_matrix)
            for i, unknown_index in enumerate(self.unknown_indices):
                self.points[unknown_index] = list(result[i])

    def _construct_unknown_energy_min_matrix(self):
        size = self.num_points - len(self.known_indices)
        n_matrix = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                n_matrix[i][j] = inner_sum(i + 1, j + 1, self.num_points - 1, self._degree)

        return n_matrix

    def _construct_known_energy_min_matrix(self):
        size = self.num_points - len(self.known_indices)
        n_matrix = np.zeros((size, 2))
        for i, unknown_index in enumerate(self.unknown_indices):
            n_matrix[i] = -1. * sum(
                inner_sum(unknown_index, known_index, self.num_points - 1, self._degree) * np.array(
                    self.points[known_index])
                for known_index in self.known_indices)
        return n_matrix


class EnergyMinimizingBezierBuilder(object):

    def __init__(self, bezier_axis, energy_axis):
        self.control_polygon = ControlPolygon()
        # Empty line
        self.control_polygon_line = Line2D([], [], ls='--', c='#666666', mew=2, mec='#204a87')
        # Create Bézier curve
        self.bezier_curve = Line2D([], [], c=self.control_polygon_line.get_markeredgecolor())
        self.energy_curve = Line2D([], [], c=self.control_polygon_line.get_markeredgecolor())
        self.drag_line = Line2D([], [], c='#e67f7f', linestyle='--')
        self.known_scatter = None
        self.unknown_scatter = None
        self.bezier_axis = bezier_axis
        self.energy_axis = energy_axis
        self.setup_bezier_axis()
        self.setup_energy_axis()
        self.canvas = self.control_polygon_line.figure.canvas
        # Event handler for mouse clicking
        self.press = self.canvas.mpl_connect('button_press_event', self.on_press)
        self.motion = self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.release = self.canvas.mpl_connect('button_release_event', self.on_release)
        self.key_event = self.canvas.mpl_connect('key_press_event', self.handle_key_event)

        self.is_generated_point = False

        self.moved_before_release = False
        self.pressed = False
        self.starting_mouse_position = None

        self.nearest_point_index = None

    def setup_bezier_axis(self):
        self.bezier_axis.add_line(self.control_polygon_line)
        self.bezier_axis.add_line(self.bezier_curve)
        self.bezier_axis.add_line(self.drag_line)
        self.known_scatter = self.bezier_axis.scatter([], [], c=self.control_polygon_line.get_markeredgecolor())
        self.unknown_scatter = self.bezier_axis.scatter([], [], c='r')
        self.bezier_axis.set_xlim(0, 1)
        self.bezier_axis.set_ylim(0, 1)
        self.bezier_axis.set_title('Bézier curve')
        self.bezier_axis.set_facecolor('#d5e4f4')
        plt.setp(self.bezier_axis.get_xticklabels()[-1], visible=False)
        EnergyMinimizingBezierBuilder.add_minor_ticks_and_grid(self.bezier_axis)

    def setup_energy_axis(self):
        self.energy_axis.set_title('Energy')
        self.energy_axis.set_xlabel('t')
        self.format_energy_y_label()
        self.energy_axis.add_line(self.energy_curve)
        self.energy_axis.set_facecolor('#d5e4f4')
        EnergyMinimizingBezierBuilder.add_minor_ticks_and_grid(self.energy_axis)

    @staticmethod
    def add_minor_ticks_and_grid(axis):
        axis.grid(b=True, which='major', color='b', linestyle='-', alpha=0.15)
        axis.grid(b=True, which='minor', color='b', linestyle='-', alpha=0.075)
        axis.minorticks_on()

    def on_press(self, event):
        if event.inaxes != self.control_polygon_line.axes:
            return
        else:
            self.pressed = True
            if self.control_polygon.num_points > 0:
                self.nearest_point_index = self.control_polygon.get_nearest_point(event.xdata, event.ydata)
                self.starting_mouse_position = self.control_polygon.points[self.nearest_point_index]

    def on_motion(self, event):
        if self.pressed:
            self.moved_before_release = True
            self.drag_line.set_data([self.starting_mouse_position[0], event.xdata],
                                    [self.starting_mouse_position[1], event.ydata])
            self.canvas.draw()

    def on_release(self, event):
        if self.pressed:
            if not self.moved_before_release:
                self.control_polygon.add_point(event.xdata, event.ydata)
            else:
                if self.nearest_point_index is not None:
                    self.control_polygon.update_control_point(self.nearest_point_index, event.xdata, event.ydata)
                    self.drag_line.set_data([], [])
            self.update_curves()
        self.pressed = False
        self.moved_before_release = False

    def handle_key_event(self, event):
        if event.key == 'd':
            nearest_index = self.control_polygon.get_nearest_point(event.xdata, event.ydata)
            if nearest_index is not None:
                self.control_polygon.remove_control_point(nearest_index)
                self.update_curves()

        elif event.key == 'e':
            nearest_index = self.control_polygon.get_nearest_point(event.xdata, event.ydata)
            if nearest_index is not None and not self.control_polygon.is_index_first_or_last(nearest_index):
                self.control_polygon.known_to_unknown_point(nearest_index)
                self.update_curves()

        elif event.key in ('1', '2', '3'):
            self.control_polygon.degree = int(event.key)
            self.update_curves()

    def update_curves(self):
        self.control_polygon_line.set_data(*self.control_polygon.xy_data)
        self.update_control_polygon_scatter()
        if self.control_polygon.num_points > 2:
            bezier_x, bezier_y, norm, energy = self._build_energy_curve()
            self.bezier_curve.set_data(bezier_x, bezier_y)
            self._update_energy(norm, energy)
        self._update_bezier()

    def update_control_polygon_scatter(self):
        known_x, known_y = self.control_polygon.known_xy_data
        known_offsets = np.column_stack([known_x, known_y])
        self.known_scatter.set_offsets(known_offsets)

        unknown_x, unknown_y = self.control_polygon.unknown_xy_data
        unknown_offsets = np.column_stack([unknown_x, unknown_y])
        self.unknown_scatter.set_offsets(unknown_offsets)

    def _build_energy_curve(self):
        x, y = bezier(self.control_polygon.points).T
        t = np.linspace(0, 1, num=200)
        bezier_spline_x = UnivariateSpline(t, x, k=5)
        bezier_spline_y = UnivariateSpline(t, y, k=5)
        dxdt = bezier_spline_x.derivative(n=self.control_polygon.degree)(t)
        dydt = bezier_spline_y.derivative(n=self.control_polygon.degree)(t)
        norm = dxdt * dxdt + dydt * dydt

        norm_spline = UnivariateSpline(t, norm, k=5)
        energy = norm_spline.integral(0, 1)
        return x, y, norm, energy

    def _update_bezier(self):
        self.canvas.draw()

    def _update_energy(self, norm, energy):
        t = np.linspace(0, 1, num=200)
        self.energy_axis.clear()
        self.energy_axis.plot(t, norm)
        self.energy_axis.fill_between(t, 0, norm, facecolor='r', alpha=0.25)
        self.energy_axis.set_title('Energy')
        self.energy_axis.set_xlabel('t')
        self.format_energy_y_label()
        self.energy_axis.autoscale(tight=True)
        self.energy_axis.set_facecolor('#d5e4f4')
        self.add_energy_text_box(energy)

        EnergyMinimizingBezierBuilder.add_minor_ticks_and_grid(self.energy_axis)

    def format_energy_y_label(self):
        self.energy_axis.yaxis.tick_right()
        label = r'Energy $\Vert \mathbf{p}^{(%d)}(t)\Vert$' % self.control_polygon.degree
        self.energy_axis.set_ylabel(label)
        self.energy_axis.yaxis.set_label_position('right')

    def add_energy_text_box(self, energy):
        energy_text = '$N=%d$\n$m=%d$\n$\int{Energy}=%.3f$' % (
            self.control_polygon.num_points, self.control_polygon.degree, round(energy, 3))
        box_properties = {'boxstyle': 'round', 'facecolor': 'blue', 'alpha': 0.3}
        self.energy_axis.text(0.95, 0.95, energy_text, transform=self.energy_axis.transAxes, verticalalignment='top',
                              horizontalalignment='right',
                              bbox=box_properties, )


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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.set_facecolor('#cfdae6')
    plt.subplots_adjust(wspace=0.01, left=0.05, bottom=0.1, right=0.925, top=0.85)
    fig.suptitle('Energy-minimizing Bézier Curves', fontsize=16)
    bezier_builder = EnergyMinimizingBezierBuilder(ax1, ax2)
    plt.show()
