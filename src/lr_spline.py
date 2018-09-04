from typing import List

from src.b_spline import BSpline
from src.element import Element
from src.meshline import Meshline


class LRSpline(object):
    """
    Represents a LRSpline, which is a tuple (M, S), where M is a mesh and S is a set of basis functions
    defined on M.
    """

    def __init__(self, mesh: List['Element'], basis: List['BSpline']):
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

    def _split_single_basis_function(self, m: Meshline, b: BSpline) -> None:
        """
        Splits a single basis function according to the provided meshline and updates the set of basis functions.
        :param m: meshline to split basis func by
        :param b: basis function to split
        """
        raise NotImplementedError(
            'LRSpline.{} is not implemented yet'.format(self._split_single_basis_function.__name__))
