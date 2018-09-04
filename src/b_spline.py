import typing

import numpy as np

from src.element import Element

Vector = typing.List['float']


def intersects(b_spline: "BSpline", element: "Element") -> bool:
    """
    Returns true if the support of b_spline intersects the element with positive area.
    :param b_spline: b_spline whose support is to be checked
    :param element: element whose domain is to be checked
    :return: true or false
    """

    intersection_u = min(b_spline.knots_u[-1], element.u_max) - max(b_spline.knots_u[0], element.u_min)
    intersection_v = min(b_spline.knots_v[-1], element.v_max) - max(b_spline.knots_v[0], element.v_min)

    if intersection_u <= 0 or intersection_v <= 0:
        return False
    else:
        return True


class BSpline(object):
    """
    Represents a single weighted tensor product B-spline with associated methods and fields.
    """

    def __init__(self, degree_u: int, degree_v: int, knots_u: Vector, knots_v: Vector, weight: float = 1) -> None:
        """
        Initialize a BSpline with bidegree (degree_u, degree_v) over associated knot vectors
        knots_u and knots_v
        :param degree_u: degree in u_direction
        :param degree_v: degree in v_direction
        :param knots_u:  knot vector in u_direction
        :param knots_v:  knot_vector in v_direction
        :param weight: B-spline weight
        """
        self.degree_u = degree_u
        self.degree_v = degree_v
        self.knots_u = np.array(knots_u, dtype=np.float64)
        self.knots_v = np.array(knots_v, dtype=np.float64)
        self.weight = weight

    def __call__(self, u: float, v: float) -> float:
        """
        Evaluates the BSpline at the parametric point (u, v).
        :param u: u component
        :param v: v component
        :return: B(u, v)
        """
        raise NotImplementedError('Evaluation is not implemented yet')

    def _evaluate_univariate_b_spline(self, x: float, knots: Vector, degree: int) -> float:
        """
        Evaluates a univariate BSpline corresponding to the given knot vector and polynomial degree at the point x.
        :param x: point of evaluation
        :param knots: knot vector
        :param degree: polynomial degree
        :return: B(x)
        """
        raise NotImplementedError('Univariate Evaluation is not Implemented yet')

    def add_to_support_if_intersects(self, element: "Element") -> bool:
        """
        Returns true if the given element intersects the support of this BSpline, and
        adds element to the list of elements of support.
        :param element: element in consideration
        :return: true or false
        """
