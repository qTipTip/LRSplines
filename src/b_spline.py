import typing

import numpy as np

if False:
    from src.element import Element

Vector = typing.List['float']
ElementVector = typing.List['Element']


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


def _evaluate_univariate_b_spline(x: float, knots: typing.Union[Vector, np.ndarray], degree: int) -> float:
    """
    Evaluates a univariate BSpline corresponding to the given knot vector and polynomial degree at the point x.
    :param x: point of evaluation
    :param knots: knot vector
    :param degree: polynomial degree
    :return: B(x)
    """
    t = _augment_knots(knots, degree)
    i = _find_knot_interval(x, t)

    if i <= degree:
        return 0

    c = np.zeros(len(t) - degree - 1)
    c[degree + 1] = 1
    c = c[i - degree: i + 1]

    for k in range(degree, 0, -1):
        t1 = t[i - k + 1: i + 1]
        t2 = t[i + 1: i + k + 1]
        omega = np.divide((x - t1), (t2 - t1), out=np.zeros_like(t1, dtype=np.float64), where=(t2 - t1) != 0)

        a = np.multiply((1 - omega), c[:-1])
        b = np.multiply(omega, c[1:])
        c = a + b

    return c


def _augment_knots(knots: Vector, degree: int) -> np.ndarray:
    """
    Adds degree + 1 values to either end of the knot vector, in order to facilitate matrix based evaluation.
    :param knots: knot vector
    :param degree: polynomial degree
    :return: padded knot vector
    """
    return np.pad(knots, (degree + 1, degree + 1), 'constant', constant_values=(knots[0] - 1, knots[-1] + 1))


def _find_knot_interval(x: float, knots: np.ndarray) -> int:
    """
    Finds the index i such that knots[i] <= x < knots[i+1]
    :param x: point of interest
    :param knots: knot vector
    :return: index i
    """
    if x < knots[0] or x >= knots[-1]:
        return -1
    return np.max(np.argmax(knots > x) - 1, 0)


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
        self.coefficient = 1

        self.elements_of_support: ElementVector = []

    def __call__(self, u: float, v: float) -> float:
        """
        Evaluates the BSpline at the parametric point (u, v).
        :param u: u component
        :param v: v component
        :return: B(u, v)
        """

        return self.weight * _evaluate_univariate_b_spline(u, self.knots_u,
                                                           self.degree_u) * _evaluate_univariate_b_spline(v,
                                                                                                          self.knots_v,
                                                                                                          self.degree_v)

    def add_to_support_if_intersects(self, element: "Element") -> bool:
        """
        Returns true if the given element intersects the support of this BSpline, and
        adds element to the list of elements of support.
        :param element: element in consideration
        :return: true or false
        """

        if intersects(self, element):
            self.elements_of_support.append(element)
            return True
        else:
            return False

    def remove_from_support(self, element: "Element") -> bool:
        """
        Removes given element from the list of elements with support.
        Returns true if element is found and removed, false otherwise.
        :param element: element to remove
        :return: true or false
        """

        if element in self.elements_of_support:
            self.elements_of_support.remove(element)
            return True
        else:
            return False