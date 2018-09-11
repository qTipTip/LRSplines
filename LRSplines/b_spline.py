import typing
from functools import lru_cache

import numpy as np

if False:
    from LRSplines.element import Element

Vector = typing.List['float']
ElementVector = typing.List['Element']




def _evaluate_univariate_b_spline(x: float, knots: typing.Union[Vector, np.ndarray], degree: int,
                                  endpoint=False) -> float:
    """
    Evaluates a univariate BSpline corresponding to the given knot vector and polynomial degree at the point x.

    :param endpoint:
    :param x: point of evaluation
    :param knots: knot vector
    :param degree: polynomial degree
    :return: B(x)
    """
    knots = np.array(knots)
    i = _find_knot_interval(x, knots, endpoint=endpoint)
    if i == -1:
        return 0
    t = _augment_knots(knots, degree)
    i += degree + 1

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


def _find_knot_interval(x: float, knots: np.ndarray, endpoint=False) -> int:
    """
    Finds the index i such that knots[i] <= x < knots[i+1]

    :param endpoint:
    :param x: point of interest
    :param knots: knot vector
    :return: index i
    """

    # if we have requested end point, and are at the end, return corresponding index.
    if endpoint and (knots[-2] <= x <= knots[-1]):
        i = max(np.argmax(knots < x) - 1, 0)
        return len(knots) - i - 2

    # if we are utside the domain, return -1
    if x < knots[0] or x >= knots[-1]:
        return -1
    # otherwise, return the corresponding index

    return np.max(np.argmax(knots > x) - 1, 0)


def cached_univariate(degree: int, knots: typing.Union[typing.List[float], np.ndarray],
                      endpoint: bool = False) -> typing.Callable:
    """
    Creates a cached version of the _evaluate_univariate_b_spline functions, as in a tensor product structure
    this yields a significant speedup.

    :param endpoint:
    :param degree: polynomial degree
    :param knots: knot vector
    :return: cached univariate evaluation.
    """

    @lru_cache(maxsize=128)
    def cached_evaluation(x):
        return _evaluate_univariate_b_spline(x, knots, degree, endpoint=endpoint)

    return cached_evaluation


class BSpline(object):
    """
    Represents a single weighted tensor product B-spline with associated methods and fields.
    """
    coefficient: float

    def __init__(self, degree_u: int, degree_v: int, knots_u: Vector, knots_v: Vector, weight: float = 1, end_u=False,
                 end_v=False) -> None:
        """
        Initialize a BSpline with bidegree (degree_u, degree_v) over associated knot vectors
        knots_u and knots_v

        :param end_u:
        :param end_v:
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
        self.end_u = end_u
        self.end_v = end_v
        self.elements_of_support: ElementVector = []

        # for cached calls
        self._univariate_u = cached_univariate(degree_u, knots_u, endpoint=end_u)
        self._univariate_v = cached_univariate(degree_v, knots_v, endpoint=end_v)

    def __call__(self, u: float, v: float) -> float:
        """
        Evaluates the BSpline at the parametric point (u, v).

        :param u: u component
        :param v: v component
        :return: B(u, v)
        """

        return self.weight * self._univariate_u(u) * self._univariate_v(v)

    def add_to_support_if_intersects(self, element: "Element") -> bool:
        """
        Returns true if the given element intersects the support of this BSpline, and
        adds element to the list of elements of support.

        :param element: element in consideration
        :return: true or false
        """

        if self.intersects(element):
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

    def __eq__(self, other: object) -> bool:
        """
        Implements == operator for BSplines. Two BSplines are assumed to be equal if their knot vectors are equal
        within a tolerance.

        :param other: BSpline to compare against
        :return: true or false
        """
        if not isinstance(other, BSpline):
            return False
        return np.allclose(self.knots_u, other.knots_u, atol=1.0e-14) and np.allclose(self.knots_v, other.knots_v,
                                                                                      atol=1.0e-14)

    def intersects(self, element: "Element") -> bool:
        """
        Returns true if the support of b_spline intersects the element with positive area.

        :param b_spline: b_spline whose support is to be checked
        :param element: element whose domain is to be checked
        :return: true or false
        """

        intersection_u = min(self.knots_u[-1], element.u_max) - max(self.knots_u[0], element.u_min)
        intersection_v = min(self.knots_v[-1], element.v_max) - max(self.knots_v[0], element.v_min)

        if intersection_u <= 0 or intersection_v <= 0:
            return False
        else:
            return True

    @property
    def knot_average(self) -> typing.Tuple[float, float]:
        """
        Returns the knot average for this BSpline (the Greville point).

        :return: the knot average (u, v).
        """

        return np.average(self.knots_u), np.average(self.knots_v)

    def update_weights(self, other: "BSpline") -> None:
        """
        Updates the weights during splitting.
        """
        w = self.weight + other.weight
        self.coefficient = (self.coefficient * self.weight + other.coefficient * other.weight) / w
        self.weight = w

    def __hash__(self):
        return hash(tuple(self.knots_v)) * self.degree_u + hash(tuple(self.knots_u)) * self.degree_v
