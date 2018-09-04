from typing import List

from src.element import Element


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
