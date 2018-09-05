import typing
from typing import List

import numpy as np

from meshline import Meshline
from src.b_spline import BSpline
from src.element import Element

Vector = typing.Union[typing.List['float'], np.ndarray]


def init_tensor_product_LR_spline(d1: int, d2: int, ku: Vector, kv: Vector) -> 'LRSpline':
    """
    Initializes an LR spline at the tensor product level of bidegree (d1, d2).
    :param d1: first component degree
    :param d2: second component degree
    :param ku: knots in u_direction
    :param kv: knots in v_direction
    :return: corresponding LR_spline
    """
    elements = []
    basis = []
    for i in range(len(ku) - 1):
        for j in range(len(kv) - 1):
            elements.append(Element(ku[i], kv[j], ku[i + 1], kv[j + 1]))

    for i in range(len(ku) - d1 - 1):
        for j in range(len(kv) - d2 - 1):
            basis.append(BSpline(d1, d2, ku[i: i + d1 + 2], kv[j: j + d2 + 2]))

    for b in basis:
        for e in elements:
            if b.add_to_support_if_intersects(e):
                e.add_supported_b_spline(b)

    return LRSpline(elements, basis)


class LRSpline(object):
    """
    Represents a LRSpline, which is a tuple (M, S), where M is a mesh and S is a set of basis functions
    defined on M.
    """

    def __init__(self, mesh: List['Element'], basis: List['BSpline']) -> None:
        """
        Initialize an LR Spline with associated set of elements, and set of basis functions.
        :param mesh: elements constituting the LR mesh
        :param basis: basis functions defined over the LR mesh
        """
        self.M = mesh
        self.S = basis

    def refine_by_element_full(self, e: Element) -> None:
        """
        Refines the LRSpline by finding and inserting a meshline that ensures that all supported BSplines on the
        given element will be split by the refinement.
        :param e: element to refine
        """
        raise NotImplementedError('LRSpline.{} is not implemented yet'.format(self.refine_by_element_full.__name__))

    def refine_by_element_minimal(self, e: Element) -> None:
        """
        Refines the LRSpline by finding and inserting the smallest possible meshline that splits the support of at least one
        BSpline.
        :param e: element to refine
        """
        raise NotImplementedError('LRSpline.{} is not implemented yet'.format(self.refine_by_element_minimal.__name__))

    def get_minimal_span_meshline(self, e: Element, axis=0) -> Meshline:
        """
        Finds the shortest possible meshline in direction prescribed by axis that splits at least one supported B-spline on the element.
        :param e: element to refine by
        :param axis: direction to look for split, 0 vertical, 1 horizontal
        :return: minimal span meshline
        """

        smallest_start = None
        smallest_stop = None
        basis: BSpline
        for basis in e.supported_b_splines:
            k = basis.knots_v if axis == 0 else basis.knots_u
            current_length = abs(k[-1] - k[0])
            if smallest_start is None or smallest_stop is None or current_length < (smallest_stop - smallest_start):
                smallest_start = k[0]
                smallest_stop = k[-1]

        constant_value = e.midpoint[axis]

        return Meshline(smallest_start, smallest_stop, constant_value, axis)
