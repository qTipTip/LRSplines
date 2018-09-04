import typing

import numpy as np

Vector = typing.List['float']


class BSpline(object):
    """
    Represents a single tensor product B-spline with associated methods and fields.
    """

    def __init__(self, degree_u: int, degree_v: int, knots_u: Vector, knots_v: Vector) -> None:
        """
        Initialize a BSpline with bidegree (degree_u, degree_v) over associated knot vectors
        knots_u and knots_v
        :param degree_u: degree in u_direction
        :param degree_v: degree in v_direction
        :param knots_u:  knot vector in u_direction
        :param knots_v:  knot_vector in v_direction
        """
        self.degree_u = degree_u
        self.degree_v = degree_v
        self.knots_u = np.array(knots_u, dtype=np.float64)
        self.knots_v = np.array(knots_v, dtype=np.float64)
