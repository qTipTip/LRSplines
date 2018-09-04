from src.b_spline import BSpline
from src.element import Element


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
