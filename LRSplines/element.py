"""
This class represents a single mesh element (i.e., a rectangle), represented in terms
of its lower left and upper right corner.
"""
import typing

from LRSplines.b_spline import BSpline

BasisFunctions = typing.List[BSpline]


class Element(object):

    def __init__(self, u_min: float, v_min: float, u_max: float, v_max: float) -> None:
        """
        Initialize an Element (a rectangle) with lower left corner (u_min, v_min)
        and upper right corner (u_max, v_max)

        :param u_min: lower left u_component
        :param v_min: lower left v_component
        :param u_max: upper right u_component
        :param v_max: upper right v_component
        """

        self.u_min = u_min
        self.v_min = v_min
        self.u_max = u_max
        self.v_max = v_max

        self.supported_b_splines: BasisFunctions = []

    def fetch_neighbours(self):
        """
        Returns a list of neighbouring elements based on supported B-splines.

        :return:
        """
        pass

    def contains(self, u: float, v: float) -> bool:
        """
        Returns True if this element contains the point (u, v)

        :param u: u_component
        :param v: v_component
        :return:
        """

        return self.u_min <= u <= self.u_max and self.v_min <= v <= self.v_max

    def get_supported_b_spline(self, i: int):
        """
        Returns the i-th supported B-spline.

        :param i: index of supported B-spline
        :return: b-spline i
        """
        return self.supported_b_splines[i]

    def add_supported_b_spline(self, b_spline):
        """
        Adds a B-spline to the list of supported B-splines.

        :param b_spline: B-spline to add
        """
        self.supported_b_splines.append(b_spline)

    def remove_supported_b_spline(self, b_spline):
        """
        Removes a B-spline from the list of supported B-splines.

        :param b_spline: B-spline to remove
        """
        self.supported_b_splines.remove(b_spline)

    def has_supported_b_spline(self, b_spline) -> bool:
        """
        Returns True if given b_spline is among the list of supported b-splines.

        :param b_spline: B-spline to check
        :return: True or False
        """
        return b_spline in self.supported_b_splines

    def update_supported_basis(self, b_splines: BasisFunctions) -> None:
        """
        Updates the list of supported basis functions.

        :param b_splines: list of BSpline functions
        """

        raise NotImplementedError('Updating of supported basis functions is not implemented yet')

    def split(self, axis: int, split_value: float) -> "Element":
        """
        Splits the element into two, resizing into the left half, and returning the right half.

        :return: right half of element.
        """

        new_element = None
        if axis == 0:  # vertical split
            if not self.u_min < split_value < self.u_max:
                return None
            new_element = Element(split_value, self.v_min, self.u_max, self.v_max)
            self.u_max = split_value

        elif axis == 1:  # horizontal split
            if not self.v_min < split_value < self.v_max:
                return None
            new_element = Element(self.u_min, split_value, self.u_max, self.v_max)
            self.v_max = split_value

        return new_element

    @property
    def midpoint(self) -> typing.Tuple[float, float]:
        """
        Returns the midpoint of the element.

        :return: midpoint of the element
        """
        return (self.u_max - self.u_min) * 0.5 + self.u_min, (self.v_max - self.v_min) * 0.5 + self.v_min

    @property
    def area(self) -> float:
        """
        Returns the area of the element.

        :return: area of the element
        """
        return (self.u_max - self.u_min) * (self.v_max - self.v_min)

    def intersects(self, other: 'Element') -> bool:
        """
        Returns true if this element intersects the other element with *positive* area.

        :param other: the element to check intersection with.
        :return: true or false
        """
        intersection_u = min(other.u_max, self.u_max) - max(other.u_min, self.u_min)
        intersection_v = min(other.v_max, self.v_max) - max(other.v_min, self.v_min)

        if intersection_u <= 0 or intersection_v <= 0:
            return False
        else:
            return True

    def __eq__(self, other: 'Element') -> bool:
        """
        Checks whether the two elements are equal within a tolerance.

        :param other: element to compare
        :return: true or false
        """

        tol = 1.0e-14
        return abs(self.u_min - other.u_min) < tol and abs(self.u_max - other.u_max) < tol and abs(
            self.v_min - other.v_min) < tol and abs(self.v_max - other.v_max) < tol

    def __repr__(self):
        return "Element({}, {}, {}, {})".format(self.u_min, self.v_min, self.u_max, self.v_max)

    def is_overloaded(self) -> bool:
        """
        Returns true if the number of supported B-splines on this element is greater than (d1 + 1)*(d2 + 1).

        :return: true if overloaded, false otherwise
        """

        b = self.supported_b_splines[0]

        return len(self.supported_b_splines) > (b.degree_u + 1) * (b.degree_v + 1)

    # TODO: Now that I think about it, this hash may change during the lifetime of the object, due to the
    # Element.split method.
    def __hash__(self):
        return hash(tuple([self.u_min, self.u_max, self.v_min, self.v_max]))