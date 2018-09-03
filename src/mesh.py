import typing

Vector = typing.List[float]


class MeshRectangle(object):
    """
    Represents a line segment (for 2D LR-Splines)
    """

    def __init__(self, index: int, multiplicity: int) -> None:
        self.index = index
        self.multiplicity = multiplicity


class Mesh(object):

    def __init__(self, knots_u: list, knots_v: list) -> None:
        """
        Initializes a mesh as a full tensor product mesh given by the knot vectors knots_u and knots_v.
        :param knots_u: knots in u parametric direction
        :param knots_v: knots in v parametric direction
        """

        # knots with multiplicity
        self.knots_u = knots_u
        self.knots_v = knots_v

        # knots without multiplicity
        self.unique_knots_u: Vector = list(set(knots_u))
        self.unique_knots_v: Vector = list(set(knots_v))

        # mesh rectangles
        self.horizontal_meshlines: typing.List['MeshRectangle'] = []
        self.vertical_meshlines: typing.List['MeshRectangle'] = []

    def insert_meshline(self, axis: int, constant_value: float, multiplicity: int) -> None:
        """
        Inserts a new meshline along specified axis with given multiplicity.
        :param axis: 0 is a vertical and 1 is a horizontal line.
        :param constant_value: the constant value of the line
        :param multiplicity: the line multiplicity
        :return: None
        """

        pass

    def visualize_mesh(self) -> None:
        """
        Plots the mesh using Matplotlib.
        :return: None
        """

        pass
