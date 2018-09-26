import typing

import numpy as np

from LRSplines.b_spline import BSpline
from LRSplines.element import Element


def hierarchichal_meshline_rectangle(min_u, min_v, max_u, max_v, step=0.5):
    """
    Produces a uniform grid of meshlines over the given rectangle with given knot interval lenght.
    For use in quick sketching of hierarchichal meshes.
    :param min_u:
    :param min_v:
    :param max_u:
    :param max_v:
    :param step:
    :return:
    """
    meshlines = [
                    Meshline(start=min_u, stop=max_u, constant_value=c, axis=1)
                    for c in [min_u + step + step * i for i in range(int((max_u - min_u) / step) - 1)]
                ] + [
                    Meshline(start=min_v, stop=max_v, constant_value=c, axis=0)
                    for c in [min_v + step + step * i for i in range(int((max_v - min_v) / step) - 1)]
                ]
    return meshlines


class Meshline(object):
    """
    Represents a meshline (knotline) in given direction with designated endpoints.
    """

    def __init__(self, start: float, stop: float, constant_value: float, axis: int, multiplicity: int = 1) -> None:
        """
        Initialize a mesh line from start to stop in direction `axis` with given constant value.
        :param start: start of line
        :param stop:  end of line
        :param constant_value: constant value in other parametric direction
        :param axis: direction of line, 0 is vertical, 1 is horizontal
        :param multiplicity: multiplicty of line, 1 by default.
        """
        self.start = start
        self.stop = stop
        self.constant_value = constant_value
        self.axis = axis
        self.multiplicity = multiplicity

    def splits_element(self, element: Element) -> bool:
        """
        Returns true whether this meshline traverses the interior of given element.
        :param element: element to check split against
        :return: true or false
        """

        if self.axis == 0:  # vertical split
            return element.u_min < self.constant_value < element.u_max and self.start <= element.v_min and \
                   self.stop >= element.v_max
        elif self.axis == 1:  # horizontal split
            return element.v_min < self.constant_value < element.v_max and self.start <= element.u_min and \
                   self.stop >= element.u_max
        return False

    def splits_basis(self, basis: BSpline) -> bool:
        """
        Returns true whether this mesh line traverses the interior of the support of the given basis function.
        :param basis: basis function to check split against
        :return: true or false
        """

        if self.axis == 0:  # vertical split
            return basis.knots_u[0] < self.constant_value < basis.knots_u[-1] and \
                   self.start <= basis.knots_v[0] and self.stop >= basis.knots_v[-1]
        elif self.axis == 1:  # horizontal split
            return basis.knots_v[0] < self.constant_value < basis.knots_v[-1] and \
                   self.start <= basis.knots_u[0] and self.stop >= basis.knots_u[-1]
        return False

    def number_of_knots_contained(self, basis: BSpline) -> int:
        """
        Returns the number of knots of given BSpline that lies on this meshline.
        :param basis: BSpline
        :return: number of knots of BSpline that lies on this meshline.
        """
        knots = basis.knots_u if self.axis == 0 else basis.knots_v

        return self._number_of_knots_contained_helper(knots)

    def _number_of_knots_contained_helper(self, knot_vector) -> int:
        tol = 1.0e-14

        return int(np.sum(np.abs(knot_vector - self.constant_value) < tol))

    def set_multiplicity(self, knots) -> None:
        """
        Sets the multiplicity of the mesh line according to how many knots in the knot vector overlaps with this constant value.
        :param knots: knot vector
        """
        self.multiplicity = self._number_of_knots_contained_helper(knots)

    @property
    def midpoint(self) -> typing.Tuple[float, float]:
        """
        Returns the midpoint of the meshline.

        :return: midpoint of the mesh line.
        """

        a = (self.stop - self.start) / 2 + self.start
        b = self.constant_value

        if self.axis == 0:
            return b, a
        else:
            return a, b

    def _similar(self, other: "Meshline") -> bool:
        """
        Returns true if the two meshlines have the same direction and the same constant_value, but not neccesarily
        the same endpoints or multiplicity.

        :param other: meshline to compare against
        :return: true or false
        """
        tol = 1.0e-14
        return self.axis == other.axis and abs(self.constant_value - other.constant_value) < tol

    def __eq__(self, other: "Meshline") -> bool:
        """
        Returns true if the two meshlines have the same direction, same constant value, and same endpoints and multiplicity.

        :param other: meshline to compare against
        :return: true or false
        """
        tol = 1.0e-14
        return self._similar(other) and abs(self.start - other.start) < tol and abs(
            self.stop - other.stop) < tol and self.multiplicity == other.multiplicity

    def contains(self, other: "Meshline") -> bool:
        """
        Returns true if meshline is completely contained in this meshline.

        :param other: meshline to check if is contained
        :return: true if other is contained, false otherwise
        """

        if not self._similar(other):
            return False

        return self.start <= other.start and self.stop >= other.stop

    def overlaps(self, other: "Meshline") -> bool:
        """
        Returns true if the two meshlines overlap.

        :param other: meshline to check for overlap
        :return: true if the meshlines overlap, false otherwise
        """
        if not self._similar(other):
            return False

        return not (other.stop < self.start or self.stop < other.start)

    def __repr__(self):
        return "Meshline(start={}, stop={}, constant_value={}, axis={})".format(self.start, self.stop,
                                                                                self.constant_value, self.axis)
