"""
This class represents a single mesh element (i.e., a rectangle), represented in terms
of its lower left and upper right corner.
"""


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

        self.supported_b_splines: list = []

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

    def split(self, axis: int, split_value: float):
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

    def midpoint(self) -> (float, float):
        """
        Returns the midpoint of the element.
        :return: midpoint of the element
        """
        return (self.u_max - self.u_min) * 0.5, (self.v_max - self.v_min) * 0.5
