# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import numpy as np
from scipy.special import binom

import matplotlib.pyplot as plt

from matplotlib.lines import Line2D


class BezierBuilder(object):
    """Bézier curve interactive builder.
    """

    def __init__(self, control_polygon, ax_bernstein):
        """Constructor.
        Receives the initial control polygon of the curve.
        """
        self.control_polygon = control_polygon
        self.xp = list(control_polygon.get_xdata())
        self.yp = list(control_polygon.get_ydata())
        self.canvas = control_polygon.figure.canvas
        self.ax_main = control_polygon.axes
        self.ax_bernstein = ax_bernstein

        # Event handler for mouse clicking
        self.press = self.canvas.mpl_connect('button_press_event', self.on_press)
        self.motion = self.canvas.mpl_connect('motion_notify_event',self.on_motion)
        self.release = self.canvas.mpl_connect('button_release_event', self.on_release)
        self.delete_point = self.canvas.mpl_connect('key_press_event', self.remove_control_point)
        # Variables to know when we really need to add a point (when
        # there's no mouse movement between button press and release)
        self.moved_before_release = False
        self.pressed = False
        self.nearest_point = None
        self.distance_threshold = 0.025

        # Create Bézier curve
        line_bezier = Line2D([], [],
                             c=control_polygon.get_markeredgecolor())
        self.bezier_curve = self.ax_main.add_line(line_bezier)

    def on_press(self, event):
        # Ignore clicks outside axes
        if event.inaxes != self.control_polygon.axes:
            return
        else:
            self.pressed = True
            if self.xp:
                nearest_index, nearest_distance = self.get_nearest_point(event.xdata, event.ydata)
                if nearest_distance <= self.distance_threshold:
                    self.nearest_point = nearest_index
                else:
                    self.nearest_point = None

    def on_motion(self, event):
        # We need to handle events only when there's no movement
        # between button press and button release. If there's some
        # movement in between, it means the user is zooming or panning
        if self.pressed:
            self.moved_before_release = True

    def on_release(self, event):
        if self.pressed:
            if not self.moved_before_release:
                # Add point
                self.add_control_point(event.xdata, event.ydata)
            else:
                if self.nearest_point is not None:
                    self.update_control_point(self.nearest_point, event.xdata, event.ydata)
            self.update_curves()
        self.pressed = False
        self.moved_before_release = False

    def get_nearest_point(self, x, y):
        nearest_index = None
        nearest_distance = float('inf')
        for i, (point_x, point_y) in enumerate(zip(self.xp, self.yp)):
            distance = abs(point_x - x) + abs(point_y - y)
            if distance < nearest_distance:
                nearest_index = i
                nearest_distance = distance
        return nearest_index, nearest_distance

    def add_control_point(self, x, y):
        self.xp.append(x)
        self.yp.append(y)

    def update_control_point(self, index, x, y):
        self.xp[index] = x
        self.yp[index] = y

    def remove_control_point(self, event):
        if event.key == 'd':
            nearest_index, nearest_distance = self.get_nearest_point(event.xdata, event.ydata)
            if nearest_distance <= self.distance_threshold:
                self.xp.pop(nearest_index)
                self.yp.pop(nearest_index)
                self.update_curves()

    def update_curves(self):
        self.control_polygon.set_data(self.xp, self.yp)

        # Rebuild Bézier curve and update canvas
        bezier_x, bezier_y, norm = self._build_bezier()
        self.bezier_curve.set_data(bezier_x, bezier_y)
        self._update_bernstein(norm)
        self._update_bezier()

    def _build_bezier(self):
        x, y = bezier(list(zip(self.xp, self.yp))).T
        t = np.linspace(0, 1, num=200)
        diffx = np.gradient(x, t[1] - t[0])
        diffy = np.gradient(x, t[1] - t[0])
        norm = np.zeros(np.size(diffx))
        for i in range(len(norm)):
            norm[i] = diffx[i]*diffx[i] + diffy[i]*diffy[i]
        return x, y, norm

    def _update_bezier(self):
        self.canvas.draw()

    def _update_bernstein(self, norm):
        n = len(self.xp) - 1
        t = np.linspace(0, 1, num=200)

        ax = self.ax_bernstein
        ax.clear()
        for k in range(n + 1):
            ax.plot(t, norm)
        ax.set_title("Energy, N = {}".format(n))
        # ax.set_xlim(0, 1)
        # ax.set_ylim(0, 1)


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
    ax2.set_title("Energy")

    # Create BezierBuilder
    bezier_builder = BezierBuilder(line, ax2)

    plt.show()