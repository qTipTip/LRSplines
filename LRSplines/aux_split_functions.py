import typing

import numpy as np

from LRSplines.b_spline import BSpline
from LRSplines.meshline import Meshline


def _split_weights(knots: np.ndarray, x: float) -> typing.Tuple[float, float]:
    """
    Returns the weights corresponding to knot insertion of x into `knots`

    :param knots: initial knot vector
    :param x: knot to be inserted
    :return: knot insertion weights alpha1 and alpha2
    """
    if knots[-2] <= x <= knots[-1]:
        alpha_1 = 1
    else:
        alpha_1 = (x - knots[0]) / (knots[-2] - knots[0])
    if knots[0] <= x <= knots[1]:
        alpha_2 = 1
    else:
        alpha_2 = (knots[-1] - x) / (knots[-1] - knots[1])

    return alpha_1, alpha_2


def _split(alpha_1: float, alpha_2: float, b: BSpline, m: Meshline, new_knots: np.ndarray) -> typing.Tuple[
    BSpline, BSpline]:
    """
    Given two split weights, a b spline marked for splitting by the given meshline and a set of new knots,
    return the two resulting BSplines

    :param alpha_1: left split weight
    :param alpha_2: right split weight
    :param b: old BSpline
    :param m: meshline to split by
    :param new_knots: knots with inserted knot
    :return: two resulting b-splines
    """
    if m.axis == 0:  # vertical split
        b1 = BSpline(b.degree_u, b.degree_v, new_knots[:-1], b.knots_v, b.weight * alpha_1)
        b2 = BSpline(b.degree_u, b.degree_v, new_knots[1:], b.knots_v, b.weight * alpha_2)

        b1.end_v = b.end_v
        b2.end_v = b.end_v

        b1.north = b.north
        b2.north = b.north

        b1.south = b.south
        b2.south = b.south

        if b.end_u:
            b1.end_u = False
            b2.end_u = True

        if b.east:
            b2.east = True
        if b.west:
            b1.west = True

    elif m.axis == 1:  # horizontal split
        b1 = BSpline(b.degree_u, b.degree_v, b.knots_u, new_knots[:-1], b.weight * alpha_1)
        b2 = BSpline(b.degree_u, b.degree_v, b.knots_u, new_knots[1:], b.weight * alpha_2)

        b1.end_u = b.end_u
        b2.end_u = b.end_u

        b1.east = b.east
        b2.east = b.east

        b1.west = b.west
        b2.west = b.west

        if b.end_v:
            b1.end_v = False
            b2.end_v = True

        if b.end_u:
            b1.end_u = False
            b2.end_u = True

        if b.north:
            b2.north = True
        if b.south:
            b1.south = True

    return b1, b2


def split_single_basis_function(m: Meshline, b: BSpline, return_weights=False) -> typing.Union[
    typing.Tuple[BSpline, BSpline, float, float], typing.Tuple[BSpline, BSpline]]:
    """
    Splits a single basis function according to the provided meshline and updates the set of basis functions.

    :param m: meshline to split basis func by
    :param b: basis function to split
    """
    # raise NotImplementedError('LRSpline.{} is not implemented yet'.format(self._split_single_basis_function.__name__))

    new_knot = m.constant_value

    if m.axis == 0:
        knots_to_split = b.knots_u
    elif m.axis == 1:
        knots_to_split = b.knots_v

    alpha_1, alpha_2 = _split_weights(knots_to_split, new_knot)

    new_knot_index = np.searchsorted(knots_to_split, new_knot, side='right')
    new_knots = np.insert(knots_to_split, new_knot_index, new_knot)

    b1, b2 = _split(alpha_1, alpha_2, b, m, new_knots)

    if return_weights:
        return b1, b2, alpha_1, alpha_2

    return b1, b2
