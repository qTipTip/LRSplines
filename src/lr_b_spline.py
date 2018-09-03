import typing

import numpy as np

from src.element import Element

Vector = typing.List[float]


class LRBSpline(object):
    """
    Represents an LRBSpline in terms of a given bidegree and local knot vectors.
    Contains methods for evaluation.
    """

    def __init__(self, degree_p: int, degree_q: int, knots_u: Vector, knots_v: Vector, weight: float = 1,
                 coefficient: float = 1) -> None:
        """
        Initializes an LRBSpline with bidegree (degree_p, degree_q) and associated local knot vectors
        knots_u and knots_v.
        :param degree_p: degree in u_direction
        :param degree_q: degree in v_direction
        :param knots_u: knots in u_direction
        :param knots_v: knots in v_direction
        """

        self.degree_p = degree_q
        self.degree_p = degree_p

        self.coefficient = coefficient
        self.weight = weight

        self.knots_u = np.array(knots_u, dtype=np.float64)
        self.knots_v = np.array(knots_v, dtype=np.float64)

        self.elements_of_support: list = []

    def __call__(self, u: float, v: float) -> float:
        """
        Evaluates the LRBspline at the parametric point (u, v)
        :param u: u_component of point of evaluation
        :param v: v_component of point of evaluation
        :return: B(u, v)
        """
        pass

    def add_support(self, element: 'Element') -> None:
        """
        Adds an element to the list of elements which support this LRBspline
        :param element: element to add
        """

        pass

    def remove_support(self, element: 'Element') -> None:
        """
        Removes an element from the list of elements which support this LRBSpline.
        :param element: element to remove
        """

        pass

    def set_support(self, elements: typing.List['Element']) -> None:
        """
        Overwrites the list of element supports with given list.
        :param elements: List of elements.
        """

        pass

    def has_support(self, element: Element) -> bool:
        """
        Returns True if given element is present in list of supported elements.
        :param element: element of interest
        :return: True or False
        """

        pass
